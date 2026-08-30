# TinyReasoner — Qwen2.5-1.5B 全流程后训练小项目

> **一句话：** 在 8GB 显存的笔记本上，用 Qwen2.5-1.5B 把大模型后训练全链路走通：**Base → SFT → GRPO → Test-Time Scaling → Adaptive Scaling**，可复现、可放简历/GitHub。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.6+cu124](https://img.shields.io/badge/torch-2.6%2Bcu124-red)](https://pytorch.org/)
[![Transformers 4.57](https://img.shields.io/badge/transformers-4.57-yellow)](https://github.com/huggingface/transformers)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Repo:** https://github.com/ch-z-hc/miniLLM

---

## 1. 项目目标

不是刷榜，而是**完整走通工业级后训练流水线**，每个阶段都可运行、可检查、可复现：

```
预训练(别人已做) → SFT(模仿) → GRPO(试错/R可验证) → Test-Time Scaling(多采样) → Adaptive(动态停止)
```

*   **Base:** Qwen2.5-1.5B-Instruct 开箱评测，建立可靠 baseline
*   **SFT:** 用高质量推理示范教 `\boxed{}` 格式，Loss 只算答案部分，LoRA 微调
*   **GRPO:** 同一题采样 N 个答案，可验证 reward (+1/-0) 做组内对比，无需 Reward Model
*   **Test-Time Scaling:** 固定模型，采样 N=1/4/8/16 统计 pass@k / majority vote，画 Accuracy vs Compute 曲线
*   **Adaptive:** 先采 4 个，若答案一致度高则停，否则继续采，省算力

**机器约束：** RTX 4060 Laptop 8GB + 32GB RAM，bfloat16 约 3G 显存，batch=4 安全跑通。

## 2. 快速开始

### 环境

```bash
# torch 2.6+cu124, transformers 4.57.6, 已验证
pip install torch transformers datasets peft trl accelerate pyyaml pandas pyarrow
```

### 数据与模型（已就位，无需在线下载）

```bash
data/gsm8k/test.parquet   # 1319 条
data/gsm8k/train.parquet  # 7473 条
tmp_models/Qwen/Qwen2___5-1___5B-Instruct  # ModelScope, ~2.9G
```

用 `requests+proxy` 直下 parquet，避免 `load_dataset` 网络问题。

### 一键评测 Baseline

```bash
# 20 条 smoke（~70s, 3.5s/条）
python scripts/eval.py --config configs/eval.yaml

# 100 条稳定
python scripts/eval.py --max_samples 100

# 全量 1319 条
python scripts/eval.py --max_samples 1319
```

输出到 `results/eval_baseline/`：
```
config_snapshot.json  # 配置快照，可复现
generations.jsonl     # 每条原始 generation + 抽取结果（人工抽查 30 条）
generations.csv       # 同上，表格
metrics.json          # accuracy / boxed_rate / avg_tokens / latency
```

**20条实测：** `accuracy 0.5 / boxed_rate 0.7 / avg_gen 348 / tokens_per_sec 99`（Qwen2.5-1.5B 贪心解码）

## 3. 目录结构

```
miniLLM/
├── AGENTS.md              # 唯一流程记录入口，逐文件打勾
├── README.md              # 本文件
├── configs/
│   └── eval.yaml          # 评测配置（模型/dtype/数据/采样/batch/seed/输出）
├── src/
│   ├── verifier.py        # GSM8K 判卷：extract_boxed / is_correct(容差) / has_boxed
│   ├── prompts.py         # Prompt 包装：build_gsm8k_prompt + apply_chat_template(Qwen优先)
│   ├── utils.py           # 通用工具：set_seed / save_jsonl / save_json / ensure_dir
│   └── rewards.py         # (Phase2) correctness + format 极简 reward
├── scripts/
│   ├── eval.py            # Phase0 核心管线：yaml→parquet→batch generate→verifier→metrics
│   ├── train_sft.py       # (Phase1) PEFT LoRA + Trainer
│   ├── train_grpo.py      # (Phase2) TRL GRPOTrainer
│   └── test_time_scaling.py # (Phase3) pass@k / majority vote
├── data/gsm8k/            # parquet 已就位
├── tmp_models/            # 本地模型，不上传
├── results/               # 每次实验结构化落盘
└── analysis/              # failure_cases.md
```

## 4. 设计原则（AGENTS.md §0）

1.  **一次只做一个 milestone**，做完能运行、能检查
2.  **优先成熟生态：** PyTorch / Transformers / PEFT / TRL / Accelerate，不手写 Trainer/分布式
3.  **能用 LoRA 就不 full finetune**
4.  **所有实验可复现：** seed / config / dataset split / checkpoint / generation config 全记录
5.  **结果存结构化数据**（JSONL/CSV），不只打印
6.  **不偷改 evaluation**，效果不好也保留并分析

## 5. Phase 进度

| Phase | 文件 | 状态 |
|-------|------|------|
| **Phase0 Baseline** | `configs/eval.yaml` / `src/verifier.py` / `src/prompts.py` / `src/utils.py` / `scripts/eval.py` | ✅ 已闭环（20条通过，修左padding批量切片bug） |
| **Phase1 SFT** | `configs/sft.yaml` / `src/data.py` / `scripts/train_sft.py` | ⏳ 下一步 |
| **Phase2 GRPO** | `src/rewards.py` / `configs/grpo.yaml` / `scripts/train_grpo.py` | ⏳ |
| **Phase3+** | `analysis/failure_cases.md` / `test_time_scaling.py` / 自适应采样 | ⏳ |

详细勾选见 `AGENTS.md`。

## 6. 核心实现要点

**评测一致性：** 同一套 `src/verifier.py` + `src/prompts.py` + `scripts/eval.py` 评所有模型 Base/SFT/GRPO，避免评估偷换。

**判分容错：** 
*   GT 用 `####` 后抽取
*   预测优先 `\boxed{}`，否则回退文末数字
*   归一化去 `$,%` 逗号，`is_correct` 支持 `18.0==18`、`0.33≈1/3` 容差 `1e-2`

**批量生成坑（已修）：** 左 padding 时 `input_ids.shape[1]=113` 才是生成起点，`prompt_lens=[74,113]` 不能直接当切片，需用 `outputs[j, padded_len:]`，否则会把 `within \boxed{}` 当成生成。

**显存友好：** `bfloat16` (~3G) + `batch=4` + `max_new_tokens=512` + `max_input_length=1024` 在 8G 卡稳定；吃紧时切 `4bit` 量化。

## 7. 复现与记录

每次运行自动存 `results/<exp>/{config_snapshot.json,generations.jsonl,metrics.json,generations.csv}`，`AGENTS.md` 逐文件打勾，`git commit + push` 到 `origin/master`。

## 8. Roadmap

*   [x] Phase0 20条 smoke 通过
*   [ ] 100/1319 条全量 baseline
*   [ ] SFT LoRA 训通，`train/eval loss` 曲线
*   [ ] GRPO 小数据小步数验管线
*   [ ] Test-Time Scaling 画 `Accuracy vs N` 曲线，对比 `Fixed N` vs `Adaptive` 的 `compute saving`

---

**适合谁看：** 想在有限资源下，完整理解并复刻 SFT/RLVR 后训练全流程的同学。可直接 clone 按 `AGENTS.md` 顺序逐文件实现。

**联系：** https://github.com/ch-z-hc
