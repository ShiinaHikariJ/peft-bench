# PEFT 微调方法统一评测

在同一基座、同一数据、同一训练预算下，对比 16 种微调方法 + 不微调基线，共 **34 组实验（17 × 2 任务），已全部完成**。

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
6. PiSSA 使用论文推荐学习率 2e-5（LoRA 的 1/10），常识推理低约 2 分，建议用 2e-4 补跑。

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
results/
  {commonsense,math}/{method}/  每组的 results.json、训练日志、预测样例
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
- PiSSA、LoRA+ 学习率扫描
- 增加代码任务（HumanEval）
- 48GB 显卡上用 7B 模型复测
