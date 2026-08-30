"""
src/verifier.py — GSM8K 答案抽取与判分

职责（被 scripts/eval.py 调用）：
  - extract_ground_truth: 从 parquet 的 answer 字段 ".... #### 18" 抽 GT
  - extract_boxed:        从 generation 抽 \\boxed{...}，取最后一次出现
  - extract_final_answer: 优先 boxed，否则回退到文末数字/表达式抽取
  - normalize_answer:     去逗号/美元/空格/百分号，统一大小写
  - is_correct:           归一化后字符串相等，或数值容差相等

设计要点：
  - GSM8K 的 GT 形如 "#### 70,000" / "#### 3/4"，需 normalize 再比
  - generation 可能是 Qwen Instruct 的自由格式，不强制 \\boxed，但有则优先
  - 数值比较用容差 1e-2 级别，避免 0.30000001 这类浮点误差
  - 无外部依赖，纯正则+字符串，便于 TRL GRPO 的 reward 复用

参考：旧版 20 条 smoke accuracy 0.7 / boxed 0.85，说明该策略有效
"""

from __future__ import annotations

import re
import math

# ---- 正则 ----
# \boxed{...} — 支持中间含空格/逗号/负号/小数/分数，最外层一对括号
# 用非贪婪 + 取最后一次出现，避免贪婪跨多个 boxed
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^{}]*?)\s*\}")

# 备选：$\\boxed{}$ 外套 $ 符号已在上式兼容（只抽括号内）

# 数字抽取：整数/小数/分数/百分数/带逗号/负数
# 例：-1,234.56  70,000  3/4  12.5%  $18
_NUMBER_RE = re.compile(
    r"-?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?)?\s*%?"
    r"|-?\s*\$?\s*\d+(?:\.\d+)?\s*(?:/\s*\d+(?:\.\d+)?)?\s*%?",
    re.VERBOSE,
)

# 更严格的“末尾答案”候选：行末或句末附近的数字
# 用于 extract_final_answer 的回退
_TRAILING_NUMBER_RE = re.compile(
    r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?"
    r"|-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)"
    r"\s*[\.\!\?\)\]\"]*\s*$",
    re.MULTILINE,
)


def extract_ground_truth(answer: str) -> str:
    """
    从 GSM8K 的 answer 字段抽 GT。

    GSM8K 格式约定：推理过程后用 "#### <answer>" 给出 GT。
    例：
      "Janet sells ... #### 18" -> "18"
      ".... #### 70,000"        -> "70,000"（带逗号，交给 normalize 再处理）

    若无 "####" 则返回 strip 后的全文（兼容异常数据）。
    """
    if not isinstance(answer, str):
        return str(answer).strip() if answer is not None else ""
    if "####" in answer:
        # 取最后一次 #### 之后的内容（防正文中出现 ####）
        gt = answer.rsplit("####", 1)[-1].strip()
        # 去掉首行换行/空格，取第一行/第一个 token 段
        # GSM8K GT 通常就是一行一个数，但也可能带句号
        gt = gt.splitlines()[0].strip().strip(".").strip()
        return gt
    return answer.strip()


def extract_boxed(text: str) -> str | None:
    """
    抽取 \\boxed{...} 中的内容，取最后一次出现。

    例：
      "So answer is \\boxed{18}" -> "18"
      "\\boxed{ 70,000 }"         -> "70,000"
      无 boxed                    -> None

    注意：
      - 只剥最外层一对括号，内容原样返回（后续 normalize）
      - 若内容含嵌套括号（如分数 \\frac），当前正则不处理嵌套，
        但 GSM8K 无此情况；如需可升级为栈解析
    """
    if not text or not isinstance(text, str):
        return None
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    # 取最后一次（模型可能先写草稿 boxed 再改）
    last = matches[-1].strip()
    # 去掉首尾 $ 符号残留
    last = last.strip().strip("$").strip()
    return last if last else None


def normalize_answer(answer: str | None) -> str | None:
    """
    归一化答案字符串，便于比较。

    步骤：
      1. strip + 去首尾句号/感叹号
      2. 去逗号 "," / 美元 "$" / 空格 " " / 百分号后转小数由 is_correct 处理
      3. 去 \\boxed 残留（若直接传入 boxed 内容已剥离则无）
      4. 小写（处理 "A" / "a" 这类选择题，GSM8K 用不上但保留）
      5. 去前导零（"007" -> "7"，但 "0" 保留）

    返回 None 表示输入为空。
    例：
      "  $70,000. " -> "70000"
      " 18.0 "     -> "18.0"（保留小数，数值比较时再容差）
      None         -> None
    """
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None

    # 若传入的是完整 generation 误传，先尝试剥 boxed
    # （防御性，不影响正常答案）
    # 不做全局替换，只处理首尾
    s = s.strip()

    # 去掉首尾标点（保留内部负号/小数点/斜杠）
    # 先去首尾的 $ 和空格
    s = s.strip().strip("$").strip()
    s = s.rstrip(".!?").strip()

    # 去逗号、空格、美元符号（千分位/货币）
    s = s.replace(",", "").replace("$", "").replace(" ", "")

    # 小写
    s = s.lower()

    # 处理百分号： "75%" -> 归一化时保留 "%" 标记，由 is_correct 转 0.75
    # 这里不转，保留 "%" 让 is_correct 识别

    # 去前导零（纯数字情况）
    # 例 "007" -> "7"，但 "0.5" / "-007" 特殊处理
    # 仅当字符串匹配可选负号+数字(+小数)时才去零，避免把 "0/1" 误改
    if re.fullmatch(r"-?\d+(\.\d+)?%?", s):
        neg = s.startswith("-")
        core = s[1:] if neg else s
        is_percent = core.endswith("%")
        if is_percent:
            core = core[:-1]
        if "." in core:
            int_part, dec_part = core.split(".", 1)
            int_part = int_part.lstrip("0") or "0"
            core = f"{int_part}.{dec_part}"
        else:
            core = core.lstrip("0") or "0"
        s = ("-" if neg else "") + core + ("%" if is_percent else "")

    return s if s else None


def _to_float(s: str | None) -> float | None:
    """尝试把归一化后的字符串转 float，处理分数和百分数。"""
    if s is None or s == "":
        return None
    # 百分数
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    # 分数 "3/4" / "12/5"（不含逗号，已在 normalize 去掉）
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den == 0:
                    return None
                return num / den
            except ValueError:
                return None
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred: str | None, gt: str, tol: float = 1e-2) -> bool:
    """
    判断预测是否正确。

    逻辑：
      1. normalize 两边
      2. 若任一为 None/空 -> False
      3. 字符串精确相等 -> True
      4. 尝试数值比较：都可转 float 且 |a-b| <= tol 或相对误差 <= tol -> True
      5. 否则 False

    tol 默认 1e-2：GSM8K 多为整数，留 0.01 容差足够；
    对小数题也避免 0.333 vs 0.33 误判为错（相对误差兜底）。

    例：
      is_correct("18.0", "18")    -> True
      is_correct("$70,000", "70000") -> True（normalize 后相等）
      is_correct("0.33", "1/3")   -> True（0.333... 容差内）
    """
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)

    if pred_norm is None or gt_norm is None:
        return False
    if pred_norm == gt_norm:
        return True

    # 数值容差比较
    p_val = _to_float(pred_norm)
    g_val = _to_float(gt_norm)
    if p_val is not None and g_val is not None:
        if math.isclose(p_val, g_val, rel_tol=tol, abs_tol=tol):
            return True
        # 额外：整数容差（GSM8K 常见）
        if abs(p_val - g_val) <= tol:
            return True

    return False


def extract_final_answer(text: str) -> str | None:
    """
    从 generation 抽最终答案。

    优先级：
      1. \\boxed{...} 存在 -> 返回其内容（最后一次）
      2. 否则取文末附近的最后一个数字（兼容无 boxed 的自由格式）
      3. 否则全局最后一个数字
      4. 无数字 -> None

    例：
      "Reasoning ... So answer is \\boxed{18}" -> "18"
      "The total is 9*2=18."                  -> "18"
      "Answer: $70,000"                       -> "70,000"
    """
    if not text or not isinstance(text, str):
        return None

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    # 策略2：文末数字（最后 500 字符内找 trailing number）
    tail = text[-800:] if len(text) > 800 else text
    # 找所有 trailing number 行，取最后一个
    trailing_candidates = []
    for m in _TRAILING_NUMBER_RE.finditer(tail):
        trailing_candidates.append(m.group(1).strip())
    if trailing_candidates:
        # 取最后一个非空
        for cand in reversed(trailing_candidates):
            if cand:
                return cand.strip().rstrip(".").strip()

    # 策略3：全局最后一个数字
    all_nums = _NUMBER_RE.findall(text)
    # _NUMBER_RE 含捕获组，findall 可能返回 tuple，统一处理
    nums_flat: list[str] = []
    for n in all_nums:
        if isinstance(n, tuple):
            # 取第一个非空分组
            for g in n:
                if g and g.strip():
                    nums_flat.append(g.strip())
                    break
        elif isinstance(n, str) and n.strip():
            nums_flat.append(n.strip())

    if nums_flat:
        last = nums_flat[-1].strip().rstrip(".").strip()
        # 清理首尾 $/空格
        last = last.strip().strip("$").strip()
        return last if last else None

    return None


# ---- 可选：供 eval.py 统计 boxed 率 ----
def has_boxed(text: str) -> bool:
    """是否包含 \\boxed{}，用于统计 format_success_rate / boxed_rate。"""
    return extract_boxed(text) is not None


__all__ = [
    "extract_ground_truth",
    "extract_boxed",
    "extract_final_answer",
    "normalize_answer",
    "is_correct",
    "has_boxed",
]
