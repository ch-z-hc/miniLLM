"""SFT 数据管线：GSM8K parquet -> ChatML text 监督。

职责（承上启下）：
  configs/sft.yaml  --读-->  本模块  --产出-->  HF Dataset(text)  --喂给-->  scripts/train_sft.py (SFTTrainer/Trainer)

核心转换：
  question: "Natalia sold ..." 
  answer  : "Natalia sold 48/2 = <<48/2=24>>24 ... \\n#### 72"
            ↓
  messages: [
    {"role":"user", "content": "Please solve ...\\n\\nQuestion: Natalia sold ..."},
    {"role":"assistant", "content": "Natalia sold 48/2 = 24 ... \\n\\boxed{72}"}
  ]
            ↓ apply_chat_template(tokenizer) ↓
  text    : "<|im_start|>user\\nPlease solve ...<|im_end|>\\n<|im_start|>assistant\\nNatalia ... \\boxed{72}<|im_end|>"

设计要点：
  - 复用 src.prompts / src.verifier / src.utils，不重复造轮子
  - 处理 #### -> \\boxed{}，清理 <<...>> 计算标记
  - 训练用 right padding（与 eval left padding 区分），max_seq_length 1024 截断由 Trainer 负责
  - 同时导出 pandas 轻量路径与 HF Datasets 路径，兼容有/无 datasets 的环境
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.prompts import GSM8K_INSTRUCTION, apply_chat_template, build_gsm8k_prompt
from src.verifier import extract_ground_truth, normalize_answer

# ---- 清理 <<...>> ----
# GSM8K 推理中形如 "48/2 = <<48/2=24>>24"，其中 <<>> 是计算器标注，训练时应去掉标记保留可读性
_CALC_RE = re.compile(r"<<.*?>>")


def _clean_calc_markers(text: str) -> str:
    """去掉 <<...>> 标记，保留其后的数值，避免 '48/2=24 24' 重复。

    策略：直接删掉 <<...>>，原文本已在标记后重复写了结果值（如 >>24），
    删后得到 '48/2 = 24'，最简洁可读。
    同时做空白归一化。
    """
    if not text:
        return text
    # 删标记
    text = _CALC_RE.sub("", text)
    # 归一化空白：多个空格 -> 单空格，多个连续换行保留最多 2 个
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉行首行尾空格
    lines = [ln.strip() for ln in text.splitlines()]
    # 去掉空行堆积
    cleaned_lines: list[str] = []
    prev_empty = False
    for ln in lines:
        is_empty = (ln == "")
        if is_empty and prev_empty:
            continue
        cleaned_lines.append(ln)
        prev_empty = is_empty
    text = "\n".join(cleaned_lines).strip()
    # 再次合并残留的双空格（跨行处理后可能产生）
    text = re.sub(r"  +", " ", text)
    return text


def format_assistant_answer(answer_raw: str, add_boxed: bool = True) -> str:
    """将 GSM8K 原始 answer 转为 SFT 监督用的 assistant 内容。

    输入:
      answer_raw: parquet 的 answer 列，如
        "Natalia sold 48/2 = <<48/2=24>>24 ...\\n#### 72"
    输出:
      监督文本，如
        "Natalia sold 48/2 = 24 clips in May.\\n...\\n\\boxed{72}"
      若 add_boxed=False 则不追加 boxed（仅清理推理部分，用于分析）

    逻辑:
      1. 用 extract_ground_truth 取 #### 后真值 GT
      2. 取 #### 前的推理部分，清理 <<...>>
      3. 若推理部分为空/过短则仅返回 boxed
      4. 追加 \\boxed{GT}（若原文推理末尾已含 boxed 则不重复）
    """
    if answer_raw is None:
        answer_raw = ""
    answer_raw = str(answer_raw)

    gt = extract_ground_truth(answer_raw)
    gt_norm = normalize_answer(gt) or gt.strip()

    # 推理部分：#### 之前的文本
    if "####" in answer_raw:
        reasoning = answer_raw.rsplit("####", 1)[0].strip()
    else:
        # 无 #### 时，尝试去掉末尾的 GT 行，避免重复
        reasoning = answer_raw.strip()
        # 若推理末尾就是 GT，去掉最后一行中的 GT
        if gt and reasoning.endswith(gt):
            reasoning = reasoning[: -len(gt)].strip().rstrip(".,:")

    reasoning = _clean_calc_markers(reasoning)

    # 若推理为空（如数据异常），则只给 boxed
    if not reasoning:
        return f"\\boxed{{{gt_norm}}}" if add_boxed and gt_norm else ""

    if not add_boxed or not gt_norm:
        return reasoning

    # 避免重复 boxed：若推理已以 \boxed{GT} 结尾则直接返回
    if f"\\boxed{{{gt_norm}}}" in reasoning:
        return reasoning
    # 也处理带空格的 boxed 变体
    if re.search(r"\\boxed\s*\{\s*" + re.escape(gt_norm) + r"\s*\}", reasoning):
        return reasoning

    # 统一追加格式：换行后给 boxed
    # 保持简洁，不加多余前缀，模型在 eval 阶段被要求输出 \boxed{}
    return f"{reasoning}\n\\boxed{{{gt_norm}}}"


def build_sft_text(
    question: str,
    answer_raw: str,
    tokenizer=None,
    instruction: str | None = None,
    use_chat_template: bool = True,
) -> str:
    """单条样本转为 SFT 训练用的 text。

    参数:
      question: 原始问题
      answer_raw: 原始 answer（含 ####）
      tokenizer: 若提供且 use_chat_template=True 则走 Qwen ChatML
      instruction: 覆盖默认 GSM8K_INSTRUCTION，None 则用 prompts 中的默认
      use_chat_template: 是否尝试 chat_template，False 时用纯文本 "User: ...\\nAssistant: ..."

    返回:
      可直接作为 datasets text 列的字符串（含 <|im_start|> 等）
    """
    # 允许调用方覆盖 instruction（来自 sft.yaml 的 data.instruction）
    # build_gsm8k_prompt 内部硬编码了 GSM8K_INSTRUCTION，若需覆盖则手动拼接
    if instruction is not None and instruction != GSM8K_INSTRUCTION:
        q = question.strip() if isinstance(question, str) else str(question).strip() if question is not None else ""
        if not q:
            user_content = instruction
        else:
            user_content = f"{instruction}\n\nQuestion: {q}"
    else:
        user_content = build_gsm8k_prompt(question)

    assistant_content = format_assistant_answer(answer_raw, add_boxed=True)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    if use_chat_template and tokenizer is not None:
        return apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    # fallback 纯文本（tokenizer 为 None 或禁用模板时）
    return f"User: {user_content}\n\nAssistant: {assistant_content}"


def load_parquet_df(path: str | Path, max_samples: int | None = None, shuffle: bool = False, seed: int = 42) -> pd.DataFrame:
    """读 parquet 为 DataFrame，支持采样与 shuffle。"""
    df = pd.read_parquet(str(path))
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if max_samples is not None:
        df = df.head(int(max_samples)).reset_index(drop=True)
    return df


def build_sft_records(
    df: pd.DataFrame,
    tokenizer=None,
    instruction: str | None = None,
    use_chat_template: bool = True,
    text_column: str = "text",
) -> list[dict[str, Any]]:
    """将 DataFrame 转为 SFT records 列表，每条含 text 列。"""
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        q = row.get("question", "")
        a = row.get("answer", "")
        text = build_sft_text(q, a, tokenizer=tokenizer, instruction=instruction, use_chat_template=use_chat_template)
        rec: dict[str, Any] = {text_column: text, "question": str(q), "answer": str(a)}
        # 也保留 gt 便于调试
        rec["ground_truth"] = extract_ground_truth(str(a))
        records.append(rec)
    return records


def load_sft_datasets(
    config_path: str | Path = "configs/sft.yaml",
    tokenizer=None,
):
    """按 sft.yaml 产出 HF Datasets 的 train/eval Dataset。

    参数:
      config_path: sft.yaml 路径
      tokenizer: 已加载的 tokenizer；None 时用 fallback 纯文本且不做 chat_template

    返回:
      (train_dataset, eval_dataset) 均为 datasets.Dataset
      若未安装 datasets，则返回 (train_records, eval_records) 的 list[dict] 兼容形态

    用法:
      from transformers import AutoTokenizer
      from src.data import load_sft_datasets
      tok = AutoTokenizer.from_pretrained("tmp_models/Qwen/Qwen2___5-1___5B-Instruct", trust_remote_code=True)
      train_ds, eval_ds = load_sft_datasets("configs/sft.yaml", tok)
    """
    from src.utils import load_yaml

    cfg = load_yaml(str(config_path))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    train_path = data_cfg.get("train_path", "data/gsm8k/train.parquet")
    test_path = data_cfg.get("test_path", "data/gsm8k/test.parquet")
    max_train = data_cfg.get("max_train_samples", None)
    max_eval = data_cfg.get("max_eval_samples", 200)
    shuffle = bool(data_cfg.get("shuffle", True))
    seed = int(data_cfg.get("seed", 42))
    instruction = data_cfg.get("instruction", None)
    text_column = data_cfg.get("text_column", "text")
    use_chat_template = bool(model_cfg.get("use_chat_template", True))
    # 若 tokenizer 为 None，强制 fallback
    if tokenizer is None:
        use_chat_template = False

    train_df = load_parquet_df(train_path, max_samples=max_train, shuffle=shuffle, seed=seed)
    eval_df = load_parquet_df(test_path, max_samples=max_eval, shuffle=False, seed=seed)

    train_records = build_sft_records(train_df, tokenizer=tokenizer, instruction=instruction, use_chat_template=use_chat_template, text_column=text_column)
    eval_records = build_sft_records(eval_df, tokenizer=tokenizer, instruction=instruction, use_chat_template=use_chat_template, text_column=text_column)

    # 尝试转为 HF Dataset
    try:
        from datasets import Dataset

        train_ds = Dataset.from_list(train_records)
        eval_ds = Dataset.from_list(eval_records)
        return train_ds, eval_ds
    except ImportError:
        # datasets 未安装时返回 list
        return train_records, eval_records


__all__ = [
    "format_assistant_answer",
    "build_sft_text",
    "load_parquet_df",
    "build_sft_records",
    "load_sft_datasets",
]
