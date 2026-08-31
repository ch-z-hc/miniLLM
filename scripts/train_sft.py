"""Phase 1 SFT 训练：Qwen2.5-1.5B-Instruct + LoRA 在 GSM8K 上学 \\boxed{} 推理格式。

管线位置：
  configs/sft.yaml + src/data.py(text 列)  --->  本脚本(PEFT LoRA + HF Trainer)  --->  outputs/ 适配器
  产出的适配器随后由 scripts/eval.py 复评，与 Phase0 baseline 对比。

关键设计：
  1. 不依赖 trl（本机未安装），直接用 transformers.Trainer + peft
  2. Loss 只算 assistant 段：在 ChatML 文本中按 "<|im_start|>assistant\\n" 切分，
     prompt 部分的 label 置 -100（AGENTS §1：SFT 的 CrossEntropy 只算答案部分）
  3. 训练用 right padding（与 eval 的 left padding 区分）
  4. 显存自适应：--preset 4060 单卡回退 per_device1 + accum16 + gradient_checkpointing
  5. 全程结构化记录：config_snapshot.json / training_log.jsonl / metrics.json 落 results/

用法：
  # 冒烟（4060 8G，约 1~2 分钟）
  python scripts/train_sft.py --smoke --preset 4060
  # 正式（单卡 4060）
  python scripts/train_sft.py --preset 4060
  # 正式（4×3090 DDP）
  accelerate launch --num_processes 4 scripts/train_sft.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_sft_records, load_parquet_df
from src.utils import ensure_dir, load_yaml, save_json, set_seed

# ChatML 中 assistant 段起点；fallback 纯文本模板用第二个
ASSISTANT_MARKERS = ["<|im_start|>assistant\n", "\n\nAssistant:"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1 SFT (LoRA) for GSM8K")
    p.add_argument("--config", type=str, default="configs/sft.yaml")
    p.add_argument("--preset", type=str, default="none", choices=["none", "4060", "3090"],
                   help="显存预设：4060=单卡8G(per_device1/accum16/ckpt) 3090=yaml 默认")
    p.add_argument("--smoke", action="store_true", help="冒烟测试：小数据+少步数，验证管线")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None, help="指标/日志目录，默认 results/<output_dir 名>")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true", default=None)
    p.add_argument("--load_in_4bit", action="store_true", default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """把 preset / smoke / CLI 覆盖写回 cfg，保证 config_snapshot 与实际运行一致。"""
    m, d, t = cfg.setdefault("model", {}), cfg.setdefault("data", {}), cfg.setdefault("training", {})

    # ---- 显存预设 ----
    if args.preset == "4060":
        t["per_device_train_batch_size"] = 1
        t["per_device_eval_batch_size"] = 1
        t["gradient_accumulation_steps"] = 16   # eff batch 16
        t["gradient_checkpointing"] = True
        t["dataloader_num_workers"] = 0
        cfg.setdefault("_runtime", {})["preset"] = "4060(single-gpu-8G)"
    elif args.preset == "3090":
        cfg.setdefault("_runtime", {})["preset"] = "3090(yaml default)"

    # ---- 冒烟：只验证管线能跑通 ----
    if args.smoke:
        d["max_train_samples"] = 64
        d["max_eval_samples"] = 16
        t["max_steps"] = 4
        t["num_train_epochs"] = 1
        t["logging_steps"] = 1
        t["eval_steps"] = 2
        t["save_strategy"] = "no"
        t["warmup_ratio"] = 0.0
        if args.preset != "4060":
            t["per_device_train_batch_size"] = min(2, int(t.get("per_device_train_batch_size", 2)))
            t["gradient_accumulation_steps"] = 2
        else:
            t["gradient_accumulation_steps"] = 2
        t["output_dir"] = t.get("output_dir", "outputs/sft") + "-smoke"
        cfg.setdefault("_runtime", {})["smoke"] = True

    # ---- 逐项 CLI 覆盖 ----
    if args.model_path is not None:
        m["path"] = args.model_path
    if args.max_seq_length is not None:
        m["max_seq_length"] = args.max_seq_length
    if args.load_in_4bit:
        m["dtype"] = "4bit"
        m["load_in_4bit"] = True
    if args.max_train_samples is not None:
        d["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        d["max_eval_samples"] = args.max_eval_samples
    if args.output_dir is not None:
        t["output_dir"] = args.output_dir
    if args.num_train_epochs is not None:
        t["num_train_epochs"] = args.num_train_epochs
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        t["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        t["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        t["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.gradient_checkpointing:
        t["gradient_checkpointing"] = True
    return cfg


# --------------------------------------------------------------------------------------
# 模型 / tokenizer
# --------------------------------------------------------------------------------------
def resolve_dtype(dtype_str: str):
    import torch

    s = str(dtype_str).lower()
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("float32", "fp32"):
        return torch.float32
    if s == "4bit":
        return "4bit"
    return torch.bfloat16


def load_tokenizer(cfg: dict[str, Any]):
    from transformers import AutoTokenizer

    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(
        m["path"],
        trust_remote_code=bool(m.get("trust_remote_code", True)),
        padding_side=m.get("padding_side", "right"),  # 训练 right padding
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_model(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM

    m = cfg["model"]
    t = cfg["training"]
    dtype = resolve_dtype(m.get("dtype", "bfloat16"))
    kwargs: dict[str, Any] = {"trust_remote_code": bool(m.get("trust_remote_code", True))}

    if dtype == "4bit":
        from transformers import BitsAndBytesConfig

        compute_dtype = resolve_dtype(m.get("bnb_4bit_compute_dtype", "bfloat16"))
        if isinstance(compute_dtype, str):
            compute_dtype = torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # DDP 下每个进程各自占一张卡，不能用 device_map="auto"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        kwargs["device_map"] = {"": local_rank}
    else:
        kwargs["torch_dtype"] = dtype
        # 训练阶段不使用 device_map="auto"（会与 Trainer/DDP 的设备管理冲突），
        # 让 Trainer 自行 .to(device)

    print(f"[sft] loading model: {m['path']} (dtype={m.get('dtype')})")
    model = AutoModelForCausalLM.from_pretrained(m["path"], **kwargs)
    model.config.use_cache = False  # 与 gradient_checkpointing 冲突，训练期一律关闭

    if dtype == "4bit":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(t.get("gradient_checkpointing", False))
        )
    return model


def wrap_lora(model, cfg: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    lo = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=int(lo.get("r", 16)),
        lora_alpha=int(lo.get("alpha", 32)),
        lora_dropout=float(lo.get("dropout", 0.05)),
        bias=lo.get("bias", "none"),
        task_type=lo.get("task_type", "CAUSAL_LM"),
        target_modules=lo.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)
    # gradient_checkpointing + LoRA：冻结的 embedding 输出需 require_grad 才能回传
    if cfg["training"].get("gradient_checkpointing", False):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, peft_cfg


# --------------------------------------------------------------------------------------
# 数据：text -> input_ids / labels(prompt 段 -100)
# --------------------------------------------------------------------------------------
def split_prompt_completion(text: str) -> tuple[str, str]:
    """按 assistant 起始标记切开 prompt / completion；找不到标记则整段都算监督。"""
    for marker in ASSISTANT_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            cut = idx + len(marker)
            return text[:cut], text[cut:]
    return "", text


def make_tokenize_fn(tokenizer, max_seq_length: int):
    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = example["text"]
        prompt, _ = split_prompt_completion(text)
        full_ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_seq_length)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"] if prompt else []
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = -100
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "n_prompt_tokens": n_prompt,
            "n_total_tokens": len(full_ids),
        }

    return _tokenize


class PromptMaskedCollator:
    """right padding；input_ids 用 pad_token_id 补，labels 用 -100 补。"""

    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.pad_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids, attn, labels = [], [], []
        for f in features:
            ids = list(f["input_ids"])
            lab = list(f["labels"])
            am = list(f.get("attention_mask", [1] * len(ids)))
            pad_n = max_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_n)
            attn.append(am + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_datasets(cfg: dict[str, Any], tokenizer):
    """按 cfg（已含 CLI 覆盖）读 parquet -> src.data 转 ChatML text -> tokenize + mask。"""
    from datasets import Dataset

    d, m = cfg["data"], cfg["model"]
    max_seq_length = int(m.get("max_seq_length", 1024))
    use_chat_template = bool(m.get("use_chat_template", True))

    train_df = load_parquet_df(
        d.get("train_path", "data/gsm8k/train.parquet"),
        max_samples=d.get("max_train_samples"),
        shuffle=bool(d.get("shuffle", True)),
        seed=int(d.get("seed", 42)),
    )
    eval_df = load_parquet_df(
        d.get("test_path", "data/gsm8k/test.parquet"),
        max_samples=d.get("max_eval_samples", 200),
        shuffle=False,
        seed=int(d.get("seed", 42)),
    )
    print(f"[sft] raw data: train={len(train_df)} eval={len(eval_df)}")

    kw = dict(tokenizer=tokenizer, instruction=d.get("instruction"),
              use_chat_template=use_chat_template, text_column=d.get("text_column", "text"))
    train_ds = Dataset.from_list(build_sft_records(train_df, **kw))
    eval_ds = Dataset.from_list(build_sft_records(eval_df, **kw))

    tok_fn = make_tokenize_fn(tokenizer, max_seq_length)
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="tokenize train")
    eval_tok = eval_ds.map(tok_fn, remove_columns=eval_ds.column_names, desc="tokenize eval")

    # 统计：长度分布 + 截断比例 + mask 是否合理
    lens = train_tok["n_total_tokens"]
    n_trunc = sum(1 for x in lens if x >= max_seq_length)
    n_empty_label = sum(1 for a, b in zip(train_tok["n_prompt_tokens"], lens) if a >= b)
    print(f"[sft] token len: max={max(lens)} mean={sum(lens)/len(lens):.1f} "
          f"truncated(>={max_seq_length})={n_trunc} empty_label={n_empty_label}")

    stats = {"train_n": len(train_tok), "eval_n": len(eval_tok), "max_len": int(max(lens)),
             "mean_len": round(sum(lens) / len(lens), 2), "truncated": int(n_trunc),
             "empty_label": int(n_empty_label), "max_seq_length": max_seq_length}

    # 打印 1 条被监督的内容，肉眼确认只有 answer 段进 loss
    sample = train_tok[0]
    sup_ids = [i for i, l in zip(sample["input_ids"], sample["labels"]) if l != -100]
    print("\n--- masking check (sample 0) ---")
    print(f"prompt_tokens={sample['n_prompt_tokens']} total={sample['n_total_tokens']} supervised={len(sup_ids)}")
    print(f"supervised text: {tokenizer.decode(sup_ids)[:300]!r}")
    print("-------------------------------\n")

    keep = ["input_ids", "attention_mask", "labels"]
    return train_tok.select_columns(keep), eval_tok.select_columns(keep), stats


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------
def build_training_args(cfg: dict[str, Any]):
    from transformers import TrainingArguments

    t = cfg["training"]
    kwargs: dict[str, Any] = dict(
        output_dir=t.get("output_dir", "outputs/sft"),
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(t.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(t.get("num_train_epochs", 2)),
        max_steps=int(t.get("max_steps", -1)),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        optim=t.get("optimizer", "adamw_torch"),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        bf16=bool(t.get("bf16", True)),
        fp16=bool(t.get("fp16", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", False)),
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        logging_steps=int(t.get("logging_steps", 10)),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=int(t.get("eval_steps", 100)),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=int(t.get("save_steps", 100)),
        save_total_limit=int(t.get("save_total_limit", 2)),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", False)),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(t.get("greater_is_better", False)),
        seed=int(t.get("seed", 42)),
        report_to=t.get("report_to", "none"),
        remove_unused_columns=False,
    )
    if kwargs["gradient_checkpointing"]:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    # 老版本 transformers 用 evaluation_strategy，做一次兼容降级
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def make_log_callback(log_path: Path):
    from transformers import TrainerCallback

    class JsonlLogCallback(TrainerCallback):
        """把 Trainer 的每次 log（train loss / eval loss / lr）追加进 jsonl，供画曲线用。"""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or not state.is_world_process_zero:
                return
            row = {"step": int(state.global_step), "epoch": round(float(state.epoch or 0), 4), **logs}
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return JsonlLogCallback()


def main():
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args)

    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)

    output_dir = Path(cfg["training"]["output_dir"])
    results_dir = Path(args.results_dir) if args.results_dir else Path("results") / output_dir.name
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    print(f"[sft] seed={seed} output_dir={output_dir} results_dir={results_dir}")

    tokenizer = load_tokenizer(cfg)
    train_ds, eval_ds, data_stats = build_datasets(cfg, tokenizer)

    model = load_model(cfg)
    model, peft_cfg = wrap_lora(model, cfg)

    training_args = build_training_args(cfg)
    eff_batch = (training_args.per_device_train_batch_size
                 * training_args.gradient_accumulation_steps
                 * max(1, training_args.world_size))
    est_steps = (training_args.max_steps if training_args.max_steps > 0
                 else math.ceil(len(train_ds) / eff_batch * training_args.num_train_epochs))
    print(f"[sft] eff_batch={eff_batch} (per_device={training_args.per_device_train_batch_size}"
          f" x accum={training_args.gradient_accumulation_steps} x world={training_args.world_size})"
          f"  est_steps={est_steps}")

    # 配置快照（AGENTS §6）：yaml + CLI 覆盖 + 运行时信息
    cfg_snapshot = dict(cfg)
    cfg_snapshot["_runtime"] = {
        **cfg.get("_runtime", {}),
        "cli": vars(args),
        "eff_batch": eff_batch,
        "est_steps": est_steps,
        "data_stats": data_stats,
        "world_size": training_args.world_size,
    }
    if training_args.process_index == 0:
        save_json(results_dir / "config_snapshot.json", cfg_snapshot)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PromptMaskedCollator(tokenizer),
        callbacks=[make_log_callback(results_dir / "training_log.jsonl")],
    )

    t0 = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_sec = time.perf_counter() - t0
    print(f"[sft] train done in {train_sec/60:.1f} min")

    eval_metrics = trainer.evaluate()
    print(f"[sft] final eval: {eval_metrics}")

    if training_args.process_index == 0:
        # 只存 LoRA adapter（几十 MB），推理时 base + adapter 加载
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        metrics = {
            "train_runtime_sec": round(train_sec, 2),
            "train_loss": round(float(train_result.metrics.get("train_loss", float("nan"))), 6),
            "eval_loss": round(float(eval_metrics.get("eval_loss", float("nan"))), 6),
            "eval_perplexity": round(float(math.exp(min(20.0, eval_metrics.get("eval_loss", 20.0)))), 4),
            "global_step": int(train_result.global_step),
            "eff_batch": eff_batch,
            "data_stats": data_stats,
            "lora": {"r": peft_cfg.r, "alpha": peft_cfg.lora_alpha, "dropout": peft_cfg.lora_dropout,
                     "target_modules": list(peft_cfg.target_modules)},
            "config": {
                "model_path": cfg["model"]["path"],
                "dtype": cfg["model"].get("dtype"),
                "max_seq_length": cfg["model"].get("max_seq_length"),
                "learning_rate": training_args.learning_rate,
                "num_train_epochs": training_args.num_train_epochs,
                "max_steps": training_args.max_steps,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "seed": seed,
            },
        }
        save_json(results_dir / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"[sft] adapter -> {output_dir.resolve()}")
        print(f"[sft] metrics -> {(results_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
