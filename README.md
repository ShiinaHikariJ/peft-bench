# PEFT 微调方法统一评测

在同一基座、同一数据、同一训练预算下，对比 16 种微调方法 + 不微调基线，共 **34 组实验（17 × 2 任务），已全部完成**。

第二轮在全部 checkpoint 上加测了 MMLU（灾难性遗忘）和 MATH-500（难题泛化），并补跑了 PiSSA（lr=2e-4），见[第二轮](#第二轮灾难性遗忘mmlu与难题泛化math-500)。

## 实验设置

| 项目 | 设置 |
|---|---|
| 基座模型 | Qwen2.5-1.5B |
| 硬件 | 单卡 RTX 4090 24GB（AutoDL），总耗时约 13 小时 |
| 框架 | transformers 5.8 + PEFT 0.18.1 |
| 常识推理 | 训练：Commonsense170K 随机抽 2 万条；测试：BoolQ / PIQA / SIQA / HellaSwag / WinoGrande / ARC-e / ARC-c / OBQA，共 22,419 题（LLM-Adapters / DoRA 协议） |
| 数学推理 | 训练：MetaMathQA 随机抽 2 万条；测试：GSM8K 1,319 题 |
| 训练 | 1 epoch，有效 batch 32，cosine，warmup 3%，只对回答算 loss，Alpaca 模板 |
| 评测 | 贪心解码，正则抽取答案；可合并的 adapter 先 merge 再评测 |

## 测试的方法

| 类别 | 方法 |
|---|---|
| 基线 | base（不微调）、Full FT、Freeze（只训最后 2 层） |
| LoRA 家族 | LoRA、QLoRA（4bit nf4）、DoRA、rsLoRA、PiSSA、LoRA+、AdaLoRA |
| 极少参数 | VeRA、IA³、LoKr、OFT |
| 软提示 | Prefix-Tuning、Prefix-Tuning（MLP 重参数化）、Prompt Tuning |

## 结果（准确率 %）

| 方法 | 常识推理 avg | GSM8K | 可训练参数 | 峰值显存 CS / Math (GB) | 训练时间 CS / Math (min) |
|---|---|---|---|---|---|
| base | 44.54 | 56.48 | 0 | – | – |
| Full FT | 76.06 | 68.76 | 100% | 16.9 / 19.4 | 26.5 / 29.9 |
| Freeze | 62.70 | 53.30 | 6.06% | 13.0 / 18.7 | 10.0 / 15.9 |
| LoRA | **78.45** | 69.90 | 1.18% | 8.65 / 14.1 | 15.4 / 23.2 |
| QLoRA | 77.80 | 69.37 | 1.18% | 7.96 / 13.7 | 20.5 / 31.1 |
| DoRA | **78.45** | 70.13 | 1.22% | 8.66 / 14.1 | 34.4 / 53.9 |
| rsLoRA | 78.02 | 69.52 | 1.18% | 8.65 / 14.1 | 15.6 / 23.3 |
| PiSSA | 76.60 | 69.75 | 1.18% | 8.65 / 14.1 | 14.6 / 21.6 |
| LoRA+ | 76.76 | 69.14 | 1.18% | 8.65 / 14.1 | 15.4 / 23.2 |
| AdaLoRA | 73.18 | 70.28 | 1.76% | 9.20 / 14.7 | 23.2 / 32.2 |
| VeRA | 69.26 | **71.04** | 0.045% | 8.40 / 13.9 | 18.1 / 28.2 |
| IA³ | 72.97 | 69.98 | 0.017% | 8.38 / 13.9 | 10.8 / 17.3 |
| OFT | 78.06 | 70.20 | 0.51% | 8.49 / 14.0 | 21.3 / 24.3 |
| LoKr | 73.27 | 70.36 | 0.043% | 8.38 / 13.9 | 21.5 / 32.7 |
| Prefix | 19.90 | 31.01 | 0.030% | 14.5 / 14.5 | 6.9 / 11.7 |
| Prefix-MLP | 36.86 | 68.31 | 0.48% | 14.6 / 14.7 | 6.9 / 11.7 |
| Prompt Tuning | 68.98 | 62.93 | 0.003% | 15.5 / 15.0 | 7.3 / 11.7 |

常识推理 8 个子集的分项分数、学习率和训练 loss 见 [`results/results.md`](results/results.md)，机器可读版本见 `results/results.csv` 和 `results/all_results.json`。

## 主要发现

1. LoRA 与 DoRA 常识推理并列第一（78.45），DoRA 训练时间约为 LoRA 的 2.2 倍。
2. QLoRA 精度几乎无损，但 1.5B 模型上只省 0.5–0.7GB 显存，且慢约 33%；它的价值要在更大模型上才体现。
3. Full FT 比 LoRA 低 1–2 分（2 万条、1 epoch 的小数据设置下）。
4. VeRA 只训 0.045% 参数，GSM8K 最高（71.04），但常识推理掉约 9 分。
5. Prefix-Tuning 在常识推理上失败：用 generate bs=1 / bs=32 / 无缓存逐步解码三种方式核对结果一致（见 `code/diag_prefix.py`），模型几乎总选第一个选项，属方法本身在该设置下未学会，不是评测 bug。
6. PiSSA 使用论文推荐学习率 2e-5（LoRA 的 1/10），常识推理低约 2 分；用 2e-4 补跑后反而更差（见第二轮）。

## 第二轮：灾难性遗忘（MMLU）与难题泛化（MATH-500）

不重新训练，直接评测上面 34 个 checkpoint：

- **MMLU**：14,042 题，标准 5-shot 格式，取 " A"–" D" 中 logit 最大者（不依赖生成格式）。基座 61.05，与官方报告的 60.9 一致。用来衡量微调对通用知识的损伤。
- **MATH-500**：只测数学任务的模型，Alpaca 模板 + 贪心生成（最多 1024 token），用 math-verify 判定答案等价，按难度 1–5 级分别统计。

另外按 LoRA 的学习率 2e-4 补跑了 PiSSA（结果在 `results_pissa_lr2e-4/`）。

| 方法 | MMLU（常识微调后） | Δ | MMLU（数学微调后） | Δ | MATH-500 | Level 5 |
|---|---|---|---|---|---|---|
| base | 61.05 | – | 61.05 | – | 13.8 | 5.2 |
| Full FT | 52.57 | **−8.48** | 59.29 | −1.76 | 33.0 | 9.7 |
| Freeze | 60.92 | −0.13 | 60.39 | −0.66 | 32.8 | 9.0 |
| LoRA | 60.14 | −0.91 | 60.67 | −0.38 | 36.2 | 15.7 |
| QLoRA | 58.44 | −2.61 | 57.98 | −3.07 | 32.0 | 9.7 |
| DoRA | 60.36 | −0.69 | 60.62 | −0.43 | 37.8 | 16.4 |
| rsLoRA | 57.19 | −3.86 | 59.81 | −1.24 | 36.2 | 11.9 |
| PiSSA (lr 2e-5) | 60.99 | −0.06 | 60.76 | −0.29 | 37.6 | 11.2 |
| PiSSA (lr 2e-4) | 50.99 | **−10.06** | 58.60 | −2.45 | 33.4 | 13.4 |
| LoRA+ | 53.61 | **−7.44** | 59.23 | −1.82 | 35.4 | 14.9 |
| AdaLoRA | 61.28 | +0.23 | 60.68 | −0.37 | **39.0** | **17.2** |
| VeRA | 61.39 | +0.34 | 60.71 | −0.34 | 36.4 | 14.9 |
| IA³ | 60.49 | −0.56 | 60.29 | −0.76 | 35.8 | 13.4 |
| OFT | 59.62 | −1.43 | 60.18 | −0.87 | 34.6 | 13.4 |
| LoKr | 61.37 | +0.32 | 60.83 | −0.22 | 37.6 | 14.2 |
| Prefix | 24.62 | −36.43 | 31.73 | −29.32 | 9.8 | 0.7 |
| Prefix-MLP | 26.40 | −34.65 | 52.80 | −8.25 | 34.2 | 14.9 |
| Prompt Tuning | 59.29 | −1.76 | 61.05 | 0.00 | 29.0 | 13.4 |

完整表（含 MATH-500 各难度、MMLU 四大类和 57 个学科）见 [`results/extra_results.md`](results/extra_results.md) 和各目录下的 `extra_results.json`。

### 第二轮发现

1. **全量微调遗忘最重**：常识微调后 MMLU 掉 8.5 分，而 LoRA 只掉 0.9，PiSSA / AdaLoRA / VeRA / LoKr 几乎不掉。
2. **遗忘程度与更新幅度相关**：LoRA+（B 矩阵 16 倍学习率）掉 7.4，rsLoRA（缩放因子更大）掉 3.9，PiSSA 用 2e-4 掉 10.1，用 2e-5 只掉 0.1。
3. **PiSSA 补跑否定了"学习率太低"的猜想**：2e-4 下常识推理 75.50（原 76.60）、GSM8K 69.07（原 69.75）、MATH-500 33.4（原 37.6），全面变差。论文推荐的 2e-5 是对的。
4. **MATH-500 排名与 GSM8K 不同**：AdaLoRA 39.0 最高，DoRA 37.8、PiSSA / LoKr 37.6 次之；Full FT 33.0、QLoRA 32.0 偏低。QLoRA 与 LoRA 的差距在难题上从 0.5 分扩大到 4.2 分。
5. **Prefix-Tuning 的失败不只是格式问题**：MMLU 用的是 logit 打分，不依赖输出格式，仍掉到接近随机（25%），说明 prefix 破坏了模型本身的知识调用。
6. Freeze 几乎不遗忘，但任务提升也最少；Prompt Tuning 同样遗忘很小，但 MATH-500 只有 29.0。

## 局限

- 每组只跑 1 个随机种子，0.5 分以内的差距不宜下结论。
- 学习率使用各方法推荐默认值，未逐方法调参。
- 只测了 1.5B 一个模型规模。

## 目录结构

```
code/
  run.py           统一训练 + 评测脚本（--method / --task / --out_dir / --eval_only）
  queue.sh         按顺序跑全部 (方法, 任务)，已有 results.json 的跳过
  rerun.sh         主队列结束后补跑失败项
  pilot2.sh        学习率/流程试跑
  prep_data.sh/.py 下载并抽样训练/测试数据
  summarize.py     汇总所有 results.json 生成 results.md / csv / json
  diag_prefix.py   Prefix-Tuning 低分诊断
  prep_extra.py    下载 MMLU / MATH-500
  extra_eval.py    第二轮评测（MMLU logit 打分、MATH-500 生成 + math-verify）
  extra_queue.sh   第二轮队列（含 PiSSA lr=2e-4 补跑）
  summarize_extra.py  汇总第二轮结果
results/
  {commonsense,math}/{method}/  每组的 results.json、extra_results.json、训练日志、预测样例
  extra_results.md / .csv       第二轮汇总
results_pissa_lr2e-4/           PiSSA 用 LoRA 学习率补跑的结果
dashboard_data/    可视化报告使用的数据与页面源码
```

## 复现

```bash
bash code/prep_data.sh                         # 准备数据到 /root/autodl-tmp/data
python code/run.py --method lora --task commonsense --out_dir outputs/commonsense/lora
bash code/queue.sh                             # 或一次跑完全部
python code/summarize.py outputs
```

脚本中的路径按 AutoDL 默认布局（`/root/autodl-tmp/...`）写死，换环境需修改 `run.py` 顶部的 `MODEL` / `DATA`。国内环境需设置 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`。

## 下一步

- 主要方法补跑 3 个种子
- LoRA+、Full FT 学习率扫描（看能否在不加重遗忘的前提下保持任务分数）
- 增加代码任务（HumanEval）
- 48GB 显卡上用 7B 模型复测
