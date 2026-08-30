# TinyReasoner — AGENTS 流程说明

> 目标：用两周做一个可放简历/GitHub 的小模型后训练项目，完整走通
> Base → SFT → GRPO → Test-Time Scaling → Adaptive Scaling
> 本文件是唯一的过程记录入口，后续每实现一个文件都在此打勾。

## 0. 项目原则（必须遵守）

1. 一次只做一个 milestone，做完要能运行、能检查结果
2. 优先用成熟生态：PyTorch / Transformers / Datasets / PEFT / TRL / Accelerate，不手写 Trainer/分布式
3. 能用 LoRA 就不 full finetune
4. 所有实验可复现：seed / config / dataset split / checkpoint / generation config 全记录
5. 结果存为结构化数据（JSONL/CSV），不只打印到终端
6. 不为了漂亮结果偷改 evaluation，效果不好也保留并分析

## 1. 总流程

```
预训练(别人已做) → SFT(模仿) → GRPO(试错/R可验证) → Test-Time Scaling(多采样) → Adaptive(动态停止)
```

- 预训练：next-token 预测，海量数据，本项目不做但需理解。Qwen2.5-1.5B 就是预训练好的起点。
- SFT：用高质量推理示范教格式，Loss 仍是 CrossEntropy，只算答案部分。
- GRPO：同一题采样 N 个答案，可验证 reward (+1/-0) 做组内对比，无需 Reward Model。
- Test-Time Scaling：固定模型，采样 N=1/4/8/16 统计 pass@k / majority vote 画 Accuracy vs Compute 曲线。
- Adaptive：先采 4 个，若答案一致度高则停，否则继续采，省算力。

## 2. 机器与模型

- GPU: RTX 4060 Laptop 8GB，RAM 32GB
- 已下载：`tmp_models/Qwen/Qwen2___5-1___5B-Instruct` (ModelScope, ~2.9G, bfloat16 约 3G 显存)
- 数据：`data/gsm8k/test.parquet(1319)` / `train.parquet(7473)` 已用 requests+proxy 直下，无需在线 load_dataset
- 备选：显存吃紧时切 Qwen3-0.6B 或开 4bit 量化

## 3. 目录演进（按需创建，不一次建全）

```
miniLLM/
├── AGENTS.md              ← 本文件，流程与进度
├── data/gsm8k/            ← 已就位
├── tmp_models/            ← 已就位
├── configs/               ← 每阶段一个 yaml，记录可复现配置
├── src/                   ← prompts / verifier / rewards / utils
├── scripts/               ← eval / train_sft / train_grpo / test_time_scaling
├── results/               ← 每次实验的 generations.jsonl + metrics.json
├── analysis/              ← failure_cases.md
└── plots/                 ← 训练曲线等
```

## 4. 分阶段文件实现清单（一次只实现一个）

### Phase 0: Baseline — 可靠的评测管线（当前阶段）

目标：同一套管线评所有模型 Base/SFT/GRPO

- [x] `configs/eval.yaml` — 模型路径/dtype/数据路径/采样数/batch/温度/seed/输出目录
- [x] `src/verifier.py` — `extract_ground_truth` (####后) / `extract_boxed` / `extract_final_answer` / `normalize_answer` / `is_correct` (容差) — 2026-08-30 已实现并自测通过
- [x] `src/prompts.py` — `build_gsm8k_prompt` + `apply_chat_template` (Qwen chat_template 优先，fallback 纯文本) — 2026-08-30 已实现并自测通过
- [ ] `src/utils.py` — `set_seed` / `save_jsonl` / `save_json`
- [ ] `scripts/eval.py` — 读yaml→读parquet→tokenizer batch→model.generate→decode→verifier→算 accuracy/format/avg_tokens/latency→存 results/

验证标准：20 条能跑通，打印 Sample 0/1 的抽取与正误，再跑 100 条看稳定指标。

### Phase 1: SFT

- [ ] `configs/sft.yaml`
- [ ] `src/data.py` — 推理数据加载与格式化为 `<reasoning>/<answer>` 统一格式
- [ ] `scripts/train_sft.py` — PEFT LoRA + Transformers Trainer/trl SFT，记录 train/eval loss

### Phase 2: Reward / GRPO

- [ ] `src/rewards.py` — 极简：correctness + format
- [ ] `configs/grpo.yaml`
- [ ] `scripts/train_grpo.py` — TRL GRPOTrainer，小数据小步数先验管线

### Phase 3~5: Dynamics / Test-Time Scaling / Adaptive

- [ ] `analysis/failure_cases.md`
- [ ] `scripts/test_time_scaling.py` — N=1/2/4/8/16 的 pass@k/majority vote
- [ ] 自适应采样逻辑 + 对比 Fixed N 的 compute saving

## 5. 评测指标（每阶段必存）

- accuracy / pass@1
- format_success_rate / boxed_rate
- avg/median/p90 generated tokens + 长度分布
- latency / tokens_per_sec
- 每条样本的原始 generation（用于人工抽查 30 条）

## 6. 复现与记录

- 每次运行把 `config_snapshot.json` + `generations.jsonl` + `metrics.json` + `generations.csv` 存 `results/<exp>/`
- Git：每完成一个文件/阶段，做完一步验证（能运行+自测通过）后立即 `git commit -m "Phase0: implement xxx"`，再进入下一文件（不要攒多个文件一起提交）
- 失败也保留，写原因到 `analysis/failure_cases.md`

## 7. 当前进度

- [x] 环境探查完成（torch 2.6+cu124, transformers 4.57.6, 8G 显存）
- [x] 模型与数据已下载（见 §2）
- [x] Phase0 旧版已跑通 20 条（accuracy 0.7, boxed 0.85）— 已清理，待你逐文件重实现
- [x] `configs/eval.yaml` 已实现（2026-08-30）
- [x] `src/verifier.py` 已实现（2026-08-30）
- [x] `src/prompts.py` 已实现（2026-08-30，Qwen chat_template 优先+fallback，指令强制 \\boxed{}，input_ids 113<1024）
- [ ] 下一步：实现 `src/utils.py`

## 8. 下一步指令（给下一个对话框的 Agent）

> 用户要一个一个文件实现。请按 §4 顺序，每次只实现一个文件，讲清作用、关键代码、如何测试，再让用户确认后再进入下一文件。不要一次性生成全部。

