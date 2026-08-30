"""评测入口：把 yaml、数据、模型、判分串起来。

跑一遍就会在 results/ 留下 config、generation、metrics，
后面 SFT/GRPO 也用同一套来比，不换尺子。

  python scripts/eval.py
  python scripts/eval.py --max_samples 100
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

# 允许作为脚本直接运行：python scripts/eval.py
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prompts import apply_chat_template, build_gsm8k_prompt
from src.utils import ensure_dir, load_yaml, save_json, save_jsonl, set_seed
from src.verifier import extract_boxed, extract_final_answer, extract_ground_truth, has_boxed, is_correct


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase0 Baseline Eval")
    p.add_argument("--config", type=str, default="configs/eval.yaml", help="eval yaml path")
    p.add_argument("--max_samples", type=int, default=None, help="override max_samples (debug)")
    p.add_argument("--output_dir", type=str, default=None, help="override output_dir")
    p.add_argument("--batch_size", type=int, default=None, help="override batch_size")
    p.add_argument("--model_path", type=str, default=None, help="override model path")
    return p.parse_args()


def resolve_dtype(dtype_str: str):
    import torch

    s = dtype_str.lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("float32", "fp32"):
        return torch.float32
    if s == "4bit":
        return "4bit"  # 特殊标记，走 BitsAndBytes
    return torch.bfloat16


def load_model_and_tokenizer(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", "tmp_models/Qwen/Qwen2___5-1___5B-Instruct")
    dtype_str = model_cfg.get("dtype", "bfloat16")
    device_map = model_cfg.get("device_map", "auto")
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    print(f"[eval] loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",  # batch generate 需 left padding
    )
    # Qwen 无显式 pad_token 时用 eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 确保 pad_token_id 存在
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = resolve_dtype(dtype_str)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }

    if torch_dtype == "4bit":
        from transformers import BitsAndBytesConfig

        bnb_dtype_str = model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        bnb_dtype = resolve_dtype(bnb_dtype_str)
        if isinstance(bnb_dtype, str):
            bnb_dtype = torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_dtype,
        )
        print(f"[eval] 4bit quantization enabled (compute_dtype={bnb_dtype})")
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    print(f"[eval] loading model: {model_path} (dtype={dtype_str}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    print(f"[eval] model loaded. dtype={model.dtype if hasattr(model, 'dtype') else torch_dtype}")

    return model, tokenizer


def build_inputs(questions: list[str], tokenizer, gen_cfg: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """对整批 question 套 prompt+chat_template，返回渲染后文本列表与 tokenized batch 信息。"""
    max_input_length = gen_cfg.get("max_input_length", 1024)
    add_generation_prompt = gen_cfg.get("add_generation_prompt", True)

    prompts = [build_gsm8k_prompt(q) for q in questions]
    rendered = [apply_chat_template(tokenizer, p, add_generation_prompt=add_generation_prompt) for p in prompts]
    return rendered, {"max_input_length": max_input_length, "rendered": rendered}


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] * (c - k) + s[c] * (k - f))


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    # CLI 覆盖
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.output_dir is not None:
        cfg["eval"]["output_dir"] = args.output_dir
    if args.batch_size is not None:
        cfg["generation"]["batch_size"] = args.batch_size
    if args.model_path is not None:
        cfg["model"]["path"] = args.model_path

    data_cfg = cfg.get("data", {})
    gen_cfg = cfg.get("generation", {})
    eval_cfg = cfg.get("eval", {})

    seed = eval_cfg.get("seed", 42)
    set_seed(seed)
    print(f"[eval] seed={seed}")

    # ---- 输出目录 ----
    output_dir = Path(eval_cfg.get("output_dir", "results/eval_baseline"))
    ensure_dir(output_dir)
    # 保存配置快照（AGENTS §6）
    if eval_cfg.get("save_config_snapshot", True):
        save_json(output_dir / "config_snapshot.json", cfg)
        print(f"[eval] config snapshot -> {output_dir / 'config_snapshot.json'}")

    # ---- 读数据 ----
    test_path = data_cfg.get("test_path", "data/gsm8k/test.parquet")
    max_samples = data_cfg.get("max_samples", None)
    shuffle = data_cfg.get("shuffle", False)

    print(f"[eval] loading data: {test_path}")
    df = pd.read_parquet(test_path)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).reset_index(drop=True)
    n = len(df)
    print(f"[eval] dataset: {n} samples (max_samples={max_samples}, shuffle={shuffle})")
    if n == 0:
        raise ValueError("empty dataset after filtering")

    # ---- 模型 ----
    import torch

    model, tokenizer = load_model_and_tokenizer(cfg)

    batch_size = int(gen_cfg.get("batch_size", 4))
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))
    do_sample = bool(gen_cfg.get("do_sample", False))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_p = float(gen_cfg.get("top_p", 1.0))
    top_k = int(gen_cfg.get("top_k", -1))
    max_input_length = int(gen_cfg.get("max_input_length", 1024))

    print(
        f"[eval] generation: max_new_tokens={max_new_tokens} do_sample={do_sample} "
        f"temp={temperature} top_p={top_p} top_k={top_k} batch={batch_size} max_input={max_input_length}"
    )

    # ---- 批量生成 ----
    questions: list[str] = df["question"].astype(str).tolist()
    answers_gt_raw: list[str] = df["answer"].astype(str).tolist()

    rendered_texts, _ = build_inputs(questions, tokenizer, gen_cfg)

    generations: list[str] = []
    prompt_token_lens: list[int] = []
    gen_token_lens: list[int] = []
    latencies: list[float] = []

    log_interval = int(eval_cfg.get("log_interval", 10))
    print_samples = int(eval_cfg.get("print_samples", 2))

    # 分批
    total_start = time.perf_counter()
    num_batches = (n + batch_size - 1) // batch_size

    for bi in range(num_batches):
        lo = bi * batch_size
        hi = min(lo + batch_size, n)
        batch_texts = rendered_texts[lo:hi]

        # tokenize
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # 移到模型设备（device_map=auto 时模型已在 GPU，输入需放 cuda）
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        prompt_lens = attention_mask.sum(dim=1).tolist()
        prompt_token_lens.extend([int(x) for x in prompt_lens])
        # 左 padding 场景：input_ids 已 pad 到 batch 内最长，生成起点是 padded 长度
        padded_input_len = int(input_ids.shape[1])

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        t0 = time.perf_counter()
        with torch.no_grad():
            # autocast 若为 bf16 可提速；device_map 已定 dtype，此处不强制
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        t1 = time.perf_counter()
        # 均摊到 batch 内各样本
        per_sample_lat = (t1 - t0) / (hi - lo)
        latencies.extend([per_sample_lat] * (hi - lo))

        # decode：只取新生成部分（左 padding 需用 padded 长度切，否则会切出 prompt 尾巴）
        for j in range(outputs.shape[0]):
            gen_ids = outputs[j, padded_input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            generations.append(text)
            # 统计生成 token 数：去 pad 后计数
            if gen_ids.numel() > 0:
                # 截掉因 batch 内最长生成而产生的 trailing pad
                # HF generate 用 pad_token_id 填充短序列尾部
                valid = (gen_ids != tokenizer.pad_token_id)
                # 但 eos 后的 pad 亦为 pad_token，此计数即为有效生成长度
                # 若 pad_token==eos，需额外处理；Qwen pad=<|endoftext|> != eos=<|im_end|>，可直接计非 pad
                gen_token_lens.append(int(valid.sum().item()))
            else:
                gen_token_lens.append(0)

        if (bi + 1) % max(1, log_interval // batch_size) == 0 or bi == num_batches - 1:
            print(f"[eval] batch {bi+1}/{num_batches} ({hi}/{n})  latency_batch={(t1-t0):.2f}s")

        # 显存友好：及时清理
        del enc, input_ids, attention_mask, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - total_start
    print(f"[eval] generation done: {n} samples in {total_time:.1f}s ({total_time/max(n,1):.2f}s/sample)")

    # ---- Verifier 判分 ----
    rows: list[dict[str, Any]] = []
    correct_flags: list[int] = []
    boxed_flags: list[int] = []
    pred_list: list[str | None] = []

    for i in range(n):
        q = questions[i]
        gt_raw = answers_gt_raw[i]
        gen = generations[i] if i < len(generations) else ""
        gt = extract_ground_truth(gt_raw)
        pred = extract_final_answer(gen)
        boxed = extract_boxed(gen)
        ok = is_correct(pred, gt)
        is_box = has_boxed(gen)

        correct_flags.append(1 if ok else 0)
        boxed_flags.append(1 if is_box else 0)
        pred_list.append(pred)

        rows.append(
            {
                "id": i,
                "question": q,
                "ground_truth": gt,
                "ground_truth_raw": gt_raw,
                "generation": gen,
                "pred": pred if pred is not None else "",
                "boxed": boxed if boxed is not None else "",
                "has_boxed": bool(is_box),
                "is_correct": bool(ok),
                "prompt_tokens": int(prompt_token_lens[i]) if i < len(prompt_token_lens) else 0,
                "generated_tokens": int(gen_token_lens[i]) if i < len(gen_token_lens) else 0,
                "latency_sec": float(latencies[i]) if i < len(latencies) else 0.0,
            }
        )

    # ---- 打印 Sample 0/1 抽取与正误（验证标准） ----
    for idx in range(min(print_samples, n)):
        r = rows[idx]
        print(f"\n--- Sample {idx} ---")
        print(f"Q: {r['question'][:200]}...")
        print(f"GT: {r['ground_truth']!r}  | raw tail: ...{r['ground_truth_raw'][-60:]}")
        print(f"Gen (first 400): {r['generation'][:400]}")
        print(f"pred={r['pred']!r} boxed={r['boxed']!r} has_boxed={r['has_boxed']} correct={r['is_correct']}")

    # ---- 指标（AGENTS §5） ----
    accuracy = sum(correct_flags) / max(len(correct_flags), 1)
    boxed_rate = sum(boxed_flags) / max(len(boxed_flags), 1)
    # format_success_rate 与 boxed_rate 同义（指令要求 \\boxed{}）
    format_success_rate = boxed_rate

    avg_gen_tokens = sum(gen_token_lens) / max(len(gen_token_lens), 1)
    median_gen_tokens = percentile([float(x) for x in gen_token_lens], 50)
    p90_gen_tokens = percentile([float(x) for x in gen_token_lens], 90)
    avg_prompt_tokens = sum(prompt_token_lens) / max(len(prompt_token_lens), 1)
    avg_latency = sum(latencies) / max(len(latencies), 1)
    tokens_per_sec = (sum(gen_token_lens) / total_time) if total_time > 0 else 0.0

    metrics = {
        "n": n,
        "accuracy": round(float(accuracy), 6),
        "pass_at_1": round(float(accuracy), 6),
        "format_success_rate": round(float(format_success_rate), 6),
        "boxed_rate": round(float(boxed_rate), 6),
        "avg_generated_tokens": round(float(avg_gen_tokens), 2),
        "median_generated_tokens": round(float(median_gen_tokens), 2),
        "p90_generated_tokens": round(float(p90_gen_tokens), 2),
        "avg_prompt_tokens": round(float(avg_prompt_tokens), 2),
        "total_latency_sec": round(float(total_time), 2),
        "avg_latency_sec_per_sample": round(float(avg_latency), 4),
        "tokens_per_sec": round(float(tokens_per_sec), 2),
        "config": {
            "model_path": cfg.get("model", {}).get("path"),
            "dtype": cfg.get("model", {}).get("dtype"),
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "batch_size": batch_size,
            "seed": seed,
        },
    }

    print("\n[eval] metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- 落盘 ----
    if eval_cfg.get("save_generations", True):
        save_jsonl(output_dir / "generations.jsonl", rows)
        print(f"[eval] -> {output_dir / 'generations.jsonl'} ({len(rows)} rows)")

    if eval_cfg.get("save_metrics", True):
        save_json(output_dir / "metrics.json", metrics)
        print(f"[eval] -> {output_dir / 'metrics.json'}")

    if eval_cfg.get("save_csv", True):
        csv_path = output_dir / "generations.csv"
        # 用 csv 模块避免 pandas 依赖过重时的编码问题
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "question",
                    "ground_truth",
                    "pred",
                    "boxed",
                    "has_boxed",
                    "is_correct",
                    "prompt_tokens",
                    "generated_tokens",
                    "latency_sec",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        print(f"[eval] -> {csv_path}")

    print(f"\n[eval] done. results in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
