"""
src/prompts.py — GSM8K Prompt 构建与 Chat Template 封装

职责（被 scripts/eval.py 调用）：
  - build_gsm8k_prompt:  把原始 question 包装为带 \\boxed{} 指令的 user content
  - apply_chat_template: 优先用 tokenizer.chat_template (Qwen Instruct) 渲染为
                         <|im_start|>... 格式，缺失时 fallback 到纯文本

设计要点：
  - 指令强制 \\boxed{}：便于 verifier.extract_boxed 优先抽取，也便于后续
    RL 的 format reward；即使模型不遵守，verifier 仍有文末数字回退
  - 与 configs/eval.yaml 的 generation.add_generation_prompt 对齐
  - 无外部依赖，仅依赖 tokenizer.apply_chat_template（若可用）
  - 兼容三类输入：str question / str user_content / List[Dict] messages

参考：
  - Qwen2.5-Instruct 的 chat_template 会自动注入 system: "You are Qwen..."
    并以 <|im_start|>user / <|im_start|>assistant 包裹
  - 旧版 20 条 smoke boxed_rate 0.85 说明该指令有效
"""

from __future__ import annotations

from typing import Union

# ---- 指令模板 ----
# 说明：刻意簡潔，避免过长挤占 max_input_length=1024
# 保留 "step by step" 触发 CoT，末句强制 boxed 便于可验证 reward
GSM8K_INSTRUCTION = (
    "Please solve the math problem step by step and "
    "put your final answer within \\boxed{}."
)

# 可选：更严格版本（备用，当前不启用）
# GSM8K_INSTRUCTION_STRICT = (
#     "Solve the problem step by step. "
#     "After reasoning, give the final answer in \\boxed{answer} format "
#     "with only the number inside the box."
# )


def build_gsm8k_prompt(question: str) -> str:
    """
    将 GSM8K 原始 question 包装为 user content。

    输入:
      question: parquet 中的 question 字段，如
        "Janet's ducks lay 16 eggs per day..."

    输出:
      供 apply_chat_template 使用的 user 消息字符串，形如:
        "Please solve the math problem step by step ...\n\nQuestion: Janet's ducks..."

    边界:
      - question 为空/None -> 返回仅指令（避免 crash，eval 可照常跑）
      - 自动 strip，去掉首尾空白与多余换行
    """
    q = question.strip() if isinstance(question, str) else str(question).strip() if question is not None else ""
    if not q:
        return GSM8K_INSTRUCTION
    # 统一格式：指令 + 空行 + Question: <q>
    # 好处：a) 模型易识别题干边界 b) 人工抽查 generations 时易读
    return f"{GSM8K_INSTRUCTION}\n\nQuestion: {q}"


def apply_chat_template(
    tokenizer,
    prompt: Union[str, list],
    add_generation_prompt: bool = True,
    system_message: str | None = None,
) -> str:
    """
    将 prompt 渲染为模型输入文本。

    优先级:
      1. 若 tokenizer 有 chat_template 且 apply_chat_template 可用 -> 用它
         （Qwen2.5-Instruct 会生成 <|im_start|>system/user/assistant 结构）
      2. 否则 fallback 到纯文本: "User: ...\\nAssistant:"

    参数:
      tokenizer:  HuggingFace tokenizer (AutoTokenizer)
      prompt:     str (user content) 或 List[Dict] (OpenAI 格式 messages)
                  - str: 自动包为 [{"role": "user", "content": prompt}]
                  - list: 直接作为 messages 透传（需含 role/content）
      add_generation_prompt: 是否在末尾追加 <|im_start|>assistant\\n
                             对应 eval.yaml generation.add_generation_prompt
      system_message: 可选覆盖 system prompt；None 则用模板默认
                      （Qwen 默认 "You are Qwen, created by Alibaba Cloud..."）

    返回:
      str: 可直接送 tokenizer(..., return_tensors="pt") 的输入文本

    兼容:
      - tokenizer 为 None -> 直接 fallback
      - tokenizer 无 chat_template 属性 -> fallback
      - apply_chat_template 抛异常 -> fallback
    """
    # ---- 1. 归一化为 messages ----
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        # 防御：非预期类型转字符串
        messages = [{"role": "user", "content": str(prompt)}]

    # 可选 system 覆盖：若调用方显式传入 system_message，则插到首位
    # 注意：Qwen 模板对 system 有特殊分支（见 tokenizer_config.json），
    # 若 messages[0] 已是 system 则模板会复用，否则自动注入默认 system
    if system_message is not None:
        # 若首条已是 system 则覆盖，否则插入
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_message
        else:
            messages = [{"role": "system", "content": system_message}] + messages

    # ---- 2. 优先走 chat_template ----
    if tokenizer is not None:
        # 显式检查 chat_template 是否存在且非空
        chat_template = getattr(tokenizer, "chat_template", None)
        apply_fn = getattr(tokenizer, "apply_chat_template", None)
        if chat_template and callable(apply_fn):
            try:
                return apply_fn(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # 模板渲染失败则降级到 fallback（不抛异常，保证 eval 不中断）
                pass

    # ---- 3. Fallback 纯文本 ----
    # 格式：逐条 "Role: content" 拼接，末尾按 add_generation_prompt 追加
    # 保持与 chat_template 语义一致：加 "Assistant:" 提示模型续写
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # 首字母大写：system/user/assistant -> System/User/Assistant
        role_cap = role.capitalize() if role in ("system", "user", "assistant") else role
        parts.append(f"{role_cap}: {content}")
    text = "\n\n".join(parts)
    if add_generation_prompt:
        text += "\n\nAssistant:"
    return text


__all__ = ["GSM8K_INSTRUCTION", "build_gsm8k_prompt", "apply_chat_template"]
