# TinyReasoner

基于 Qwen2.5-1.5B 的全流程后训练复现项目，涵盖 Base 评测、SFT、GRPO、Test-Time Scaling 与自适应采样，面向 8GB 显存环境的可复现实现。

GitHub: https://github.com/ch-z-hc/miniLLM · Docs: https://ch-z-hc.github.io/miniLLM/

## 概述

本项目旨在完整复现大模型后训练的核心链路，而非单一技术点的验证。流程为：

预训练（已完成，Qwen2.5-1.5B 为起点）→ SFT → GRPO → Test-Time Scaling → Adaptive Scaling

其中 SFT 负责格式与推理模式的模仿，GRPO 基于可验证奖励进行组内对比优化，Test-Time Scaling 与自适应采样在固定模型下通过增加或动态调整采样量提升效果。

硬件环境为 RTX 4060 Laptop 8GB / 32GB RAM，模型以 bfloat16 加载，显存占用约 3GB，支持 batch size 4 的稳定评测与训练。

## 硬件与数据

*   模型：`Qwen2.5-1.5B-Instruct`（ModelScope 本地部署，`tmp_models/Qwen/Qwen2___5-1___5B-Instruct`）
*   数据：GSM8K，`test.parquet` 1319 条 / `train.parquet` 7473 条，已转为本地 parquet，无需在线加载
*   依赖：PyTorch 2.6+cu124，Transformers 4.57.6，PEFT / TRL / Accelerate

## 评测

Phase 0 已完成，提供了统一的评测管线，用于评估 Base、SFT、GRPO 等各阶段模型。评测配置集中于 `configs/eval.yaml`，输出包含配置快照、原始生成与量化指标。

```bash
pip install torch transformers datasets peft trl accelerate pyyaml pandas pyarrow

# 20 条快速验证
python scripts/eval.py --config configs/eval.yaml

# 100 条或全量
python scripts/eval.py --max_samples 100
python scripts/eval.py --max_samples 1319
```

输出位于 `results/eval_baseline/`：

*   `config_snapshot.json` — 本次运行的完整配置
*   `generations.jsonl` / `generations.csv` — 每条样本的生成文本与抽取结果
*   `metrics.json` — accuracy、boxed 率、平均生成长度、时延等指标

当前 20 条贪心解码基线：accuracy 0.5，boxed_rate 0.7，平均生成长度 348 tokens。

## 仓库结构

```
configs/eval.yaml        评测配置
src/verifier.py          答案抽取与正确性判定
src/prompts.py           提示词构建与 Chat Template 封装
src/utils.py             随机种子、目录与序列化工具
scripts/eval.py          评测主流程
data/gsm8k/              GSM8K 数据
tmp_models/              本地模型（不纳入版本控制）
results/                 实验输出
AGENTS.md                阶段进度与实现记录
```

`AGENTS.md` 记录各阶段的文件实现状态与验证标准。

## 可复现性

*   配置、随机种子、数据切分、生成参数均通过 `config_snapshot.json` 存档
*   评测结果以 JSONL/CSV 结构化保存，支持人工抽查与后续分析
*   训练与评测阶段共用同一套 `verifier` 与 `prompts`，保证评估标准一致

## 进度

*   Phase 0（Baseline 评测管线）：已完成，含批量生成左填充切片问题的修复
*   Phase 1（SFT）：待实现，基于 LoRA 与 Transformers Trainer
*   Phase 2（GRPO）：待实现，基于 TRL GRPOTrainer 的小规模验证
*   Phase 3（Test-Time Scaling / Adaptive）：待实现

详细的阶段清单与验证标准见 `AGENTS.md`。

## 许可

MIT
