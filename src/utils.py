"""
src/utils.py — 通用工具：随机种子 / 结构化落盘

职责（被 scripts/eval.py 等调用）：
  - set_seed:        一键固定 Python / NumPy / Torch / CUDA 随机性，保证可复现
  - save_jsonl:      按行写 JSONL（generations.jsonl）
  - save_json:       写 JSON（metrics.json / config_snapshot.json）
  - ensure_dir:      确保输出目录存在
  - load_yaml:       轻量读 yaml（不强制依赖 pyyaml 时可 fallback）

设计要点：
  - 遵循 AGENTS §0-4：所有实验可复现（seed/config 全记录）
  - 遵循 AGENTS §0-5：结果存结构化数据，不只打印
  - 无分布式/训练逻辑，仅提供 eval 管线所需的最简工具
  - save_jsonl/save_json 均自动创建父目录，原子写入语义（先写临时后 rename 更稳，
    当前简化为直接写，8G 单机场景足够）
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    固定所有相关随机源。

    覆盖：
      - Python random
      - NumPy (若已安装)
      - PyTorch CPU / CUDA
      - 环境变量 PYTHONHASHSEED

    参数:
      seed: 随机种子，默认 42（与 configs/eval.yaml 对齐）
      deterministic: 若 True 则开启 torch.backends.cudnn.deterministic
                     并关闭 benchmark，以最严格可复现换取少量性能损失；
                     默认 False（eval 阶段无需严格确定性，速度优先）

    注意:
      - 即使 do_sample=False 的贪心解码，固定 seed 仍有助于 shuffle / 数据采样可复现
      - 调用应在程序入口尽早执行（eval.py 读取 config 后第一件事）
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy 可选依赖
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch 可选依赖
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # 保持默认：benchmark=True 有助卷积加速，对 LLM 影响小
            # 显式设为 False 可避免非确定性，但 Phase0 无需
            pass

        # 新版 PyTorch 的确定性开关（可选）
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """
    确保目录存在，返回 Path 对象。

    例:
      ensure_dir("results/eval_baseline") -> Path("results/eval_baseline")
    """
    p = Path(path)
    # 若传入的是文件路径，取父目录
    if p.suffix:
        p = p.parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | os.PathLike, data: Any, indent: int = 2, ensure_ascii: bool = False) -> Path:
    """
    写 JSON 文件，自动创建父目录。

    参数:
      path: 输出路径，如 "results/eval_baseline/metrics.json"
      data: 可序列化对象（dict / list）
      indent: 缩进，默认 2
      ensure_ascii: 是否转义非 ASCII，默认 False（保留中文）

    返回:
      写入的 Path
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 原子性：先写临时文件再 rename（防中断产生半写文件）
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        f.write("\n")
    tmp.replace(p)
    return p


def save_jsonl(path: str | os.PathLike, rows: list[dict[str, Any]] | Any, ensure_ascii: bool = False) -> Path:
    """
    写 JSONL 文件（每行一个 JSON），自动创建父目录。

    参数:
      path: 输出路径，如 "results/eval_baseline/generations.jsonl"
      rows: 可迭代对象，每项为 dict；也接受单个 dict
      ensure_ascii: 默认 False

    返回:
      写入的 Path

    兼容:
      - rows 为空列表 -> 产生空文件（不报错）
      - rows 中含非 dict（如 str）-> 按 json.dump 原样写
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        rows = [rows]
    # 统一为 list 以便计数；若是 generator 则先物化
    if not isinstance(rows, list):
        rows = list(rows)

    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=ensure_ascii)
            f.write("\n")
    tmp.replace(p)
    return p


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """
    读 YAML 配置，优先用 yaml.safe_load，无 pyyaml 时 fallback 到简易解析。

    仅用于 scripts/eval.py 读 configs/eval.yaml；
    若 pyyaml 未安装且 yaml 含复杂结构会抛异常提示安装。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        # 极简 fallback：仅支持本项目的扁平 yaml（不推荐，提示安装）
        raise ImportError(
            f"pyyaml not installed, cannot parse {p}. "
            "Install with: pip install pyyaml"
        )


__all__ = ["set_seed", "ensure_dir", "save_json", "save_jsonl", "load_yaml"]
