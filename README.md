# TinyReasoner

在 8GB 显存的笔记本上，把大模型后训练的整条链路跑通。

> Base 评测 → SFT 模仿 → GRPO 试错 → Test-Time 多采样 → 自适应采样。不是为了刷分，是想完整走一遍工业里实际怎么训、怎么评、怎么省算力。

用的是 `Qwen2.5-1.5B-Instruct` + `GSM8K`。机器是 RTX 4060 Laptop (8GB) + 32GB 内存，`bfloat16` 下模型占 3G 左右，`batch=4` 刚好能跑，不会爆显存。

Repo: https://github.com/ch-z-hc/miniLLM

## 为什么做这个

很多教程只讲单点：怎么调 `transformers`、怎么跑 `GRPO`。但真实链路是连起来的——评测不准，后面训练的 `reward` 就全错；`prompt` 不统一，`SFT` 和 `GRPO` 就对不上。

所以定了一个规矩：**同一套评测管线评所有模型**。`Base` 是什么分，`SFT` 和 `GRPO` 就用什么尺子量，不偷换标准。

## 现在能跑什么

`Phase0` 已经跑通了。虽然 `GSM8K 20` 条上 `accuracy 0.5` 不高，但链路是干净的：

```bash
# 20条，70秒左右，看看链路通不通
python scripts/eval.py --config configs/eval.yaml

# 100条，跑稳一点
python scripts/eval.py --max_samples 100

# 全量 1319条
python scripts/eval.py --max_samples 1319
```

每次跑完在 `results/eval_baseline/` 会留下四个文件：

*   `config_snapshot.json` 当时用的配置
*   `generations.jsonl` 每道题模型原样输出了什么
*   `generations.csv` 同上，表格版好筛选
*   `metrics.json` 分数、格式合格率、平均 token 数、耗时

之前踩过一个坑：`batch=4` 时左 padding 没切对，把输入的尾巴 `within \boxed{}` 也当成生成算进去了。修了之后 `Sample 1` 才正常，这也验证了批量评测必须小心切片。

## 目录长什么样

```
configs/eval.yaml        # 所有参数放这里，跑实验只改 yaml
src/verifier.py          # 怎么从自由文本里把答案抠出来、怎么判对
src/prompts.py           # 怎么把题包成 Qwen 的 ChatML
src/utils.py             # 固定种子、存 json/jsonl 这些杂活
scripts/eval.py          # 把上面串起来的总管线
data/gsm8k/              # parquet，已经下好了，不用联网
tmp_models/              # 本地模型，不进 git
results/                 # 跑出来的结果
```

`AGENTS.md` 是进度本，每做完一个文件就在里面打勾，不攒着一起提交。

## 怎么复现

```bash
pip install torch transformers datasets peft trl accelerate pyyaml pandas pyarrow
# transformers 4.57.6 / torch 2.6+cu124 已验证过
```

数据和模型都是本地 `tmp_models` 和 `data/gsm8k`，`eval.py` 不联网。`seed`、`batch`、`temperature` 全写在 `yaml` 里，`results` 里会再存一份快照。

## 接下来做什么

*   [x] Phase0：评测管线跑通，能看 `Sample 0/1` 的抽取和正误
*   [ ] Phase1：`SFT`，用 `LoRA` 教模型按 `\boxed{}` 格式输出
*   [ ] Phase2：`GRPO`，同一题采多个答案，用可验证的 `+1/-0` 做组内对比
*   [ ] Phase3：`Test-Time Scaling`，`N=1/4/8/16` 看 `pass@k` 和 `majority vote`
*   [ ] 自适应采样：先采 4 个，一致就停，不一致再采，省算力

每一步做完都能跑、能看结果，失败的也会留在 `analysis/failure_cases.md`，不藏着。

## 适合谁

想在资源有限的情况下，亲手把 `SFT -> RL` 这套东西从头到尾摸一遍的人。直接按 `AGENTS.md` 的顺序一个文件一个文件看就行。

---
Haoze Chen · https://github.com/ch-z-hc
