## 1.5B main table (mean ± std over seeds; n = number of seeds)

| config | n | Commonsense | GSM8K | MATH-500 | MMLU after CS | MMLU after Math | mem CS/Math GB | time CS/Math s |
|---|---|---|---|---|---|---|---|---|
| Full FT (lr 2e-5) | 3 | 76.27 ± 0.18 | 68.33 ± 0.58 | 32.53 ± 1.30 | 52.05 ± 1.49 | 59.87 ± 0.46 | 16.88 ± 0.00 / 19.34 ± 0.00 | 991 ± 11 / 1535 ± 6 |
| Full FT (lr 1e-5, tuned) | 3 | 78.09 ± 0.30 | 70.69 ± 0.09 | 35.00 ± 1.31 | 58.19 ± 0.40 | 60.39 ± 0.28 | 16.88 ± 0.00 / 19.34 ± 0.00 | 1071 ± 2 / 1608 ± 4 |
| LoRA (lr 2e-4) | 3 | 78.23 ± 0.22 | 70.25 ± 1.08 | 37.00 ± 1.00 | 60.46 ± 0.29 | 60.57 ± 0.30 | 8.65 ± 0.00 / 14.13 ± 0.00 | 867 ± 10 / 1410 ± 47 |
| LoRA (lr 5e-4) | 1 | 77.57 | 70.23 ± 0.70 | 37.00 ± 1.44 | 56.33 | 59.95 ± 0.17 | 8.65 / 14.13 ± 0.00 | 871 / 1462 ± 4 |
| QLoRA | 3 | 77.21 ± 0.55 | 69.15 ± 0.65 | 31.07 ± 1.01 | 58.49 ± 0.19 | 58.00 ± 0.09 | 7.95 ± 0.01 / 13.69 ± 0.01 | 1186 ± 36 / 1889 ± 46 |
| DoRA | 3 | 78.25 ± 0.30 | 70.66 ± 1.39 | 36.93 ± 1.33 | 60.21 ± 0.36 | 60.54 ± 0.21 | 8.66 ± 0.00 / 14.14 ± 0.00 | 2052 ± 12 / 3286 ± 81 |
| PiSSA | 3 | 76.65 ± 0.09 | 70.48 ± 0.98 | 38.07 ± 1.17 | 60.92 ± 0.22 | 60.59 ± 0.15 | 8.65 ± 0.00 / 14.13 ± 0.01 | 824 ± 44 / 1312 ± 37 |
| OFT | 3 | 77.92 ± 0.18 | 71.04 ± 0.27 | 35.13 ± 0.76 | 58.96 ± 0.58 | 60.12 ± 0.06 | 8.49 ± 0.00 / 13.98 ± 0.00 | 1297 ± 19 / 1491 ± 30 |
| AdaLoRA | 3 | 73.01 ± 0.32 | 70.23 ± 0.71 | 39.47 ± 0.99 | 61.35 ± 0.07 | 60.67 ± 0.10 | 9.20 ± 0.00 / 14.68 ± 0.01 | 1393 ± 21 / 1932 ± 5 |
| LoKr | 3 | 72.99 ± 0.25 | 69.80 ± 0.76 | 37.67 ± 0.70 | 61.36 ± 0.03 | 60.74 ± 0.13 | 8.38 ± 0.00 / 13.87 ± 0.00 | 1331 ± 38 / 2034 ± 61 |
| VeRA | 3 | 69.21 ± 0.02 | 70.86 ± 0.52 | 37.07 ± 0.61 | 61.33 ± 0.07 | 60.79 ± 0.07 | 8.40 ± 0.00 / 13.88 ± 0.01 | 1049 ± 33 / 1719 ± 46 |
| rsLoRA | 1 | 78.02 | 69.52 | 36.20 | 57.19 | 59.81 | 8.65 / 14.14 | 936 / 1400 |
| LoRA+ | 1 | 76.76 | 69.14 | 35.40 | 53.61 | 59.23 | 8.65 / 14.14 | 922 / 1394 |
| IA³ | 1 | 72.97 | 69.98 | 35.80 | 60.49 | 60.29 | 8.38 / 13.86 | 650 / 1036 |
| Freeze | 1 | 62.70 | 53.30 | 32.80 | 60.92 | 60.39 | 12.98 / 18.71 | 598 / 952 |
| Prompt | 1 | 68.98 | 62.93 | 29.00 | 59.29 | 61.05 | 15.45 / 15.04 | 438 / 700 |
| Prefix-MLP | 1 | 36.86 | 68.31 | 34.20 | 26.40 | 52.80 | 14.63 / 14.66 | 417 / 700 |
| Prefix | 1 | 19.90 | 31.01 | 9.80 | 24.62 | 31.73 | 14.51 / 14.54 | 415 / 702 |
| base | 1 | 44.54 | 56.48 | 13.80 | – | 61.05 | – / – | – / – |

## Paired bootstrap (per-question correctness averaged over seeds; 95% CI of A − B, points)

| task | A | B | A − B | 95% CI | p |
|---|---|---|---|---|---|
| commonsense | LoRA (lr 2e-4) | Full FT (lr 2e-5) | +1.96 | [+1.49, +2.42] | 0.0002 |
| commonsense | LoRA (lr 2e-4) | Full FT (lr 1e-5, tuned) | +0.15 | [-0.20, +0.49] | 0.4036 |
| commonsense | LoRA (lr 5e-4) | Full FT (lr 1e-5, tuned) | -0.51 | [-1.00, -0.02] | 0.0424 |
| commonsense | Full FT (lr 1e-5, tuned) | Full FT (lr 2e-5) | +1.81 | [+1.44, +2.19] | 0.0002 |
| commonsense | QLoRA | LoRA (lr 2e-4) | -1.02 | [-1.38, -0.67] | 0.0002 |
| commonsense | DoRA | LoRA (lr 2e-4) | +0.01 | [-0.20, +0.22] | 0.9004 |
| commonsense | PiSSA | LoRA (lr 2e-4) | -1.58 | [-1.95, -1.21] | 0.0002 |
| commonsense | OFT | LoRA (lr 2e-4) | -0.31 | [-0.61, -0.01] | 0.0432 |
| commonsense | AdaLoRA | LoRA (lr 2e-4) | -5.23 | [-5.83, -4.61] | 0.0002 |
| commonsense | LoKr | LoRA (lr 2e-4) | -5.25 | [-5.87, -4.62] | 0.0002 |
| commonsense | VeRA | LoRA (lr 2e-4) | -9.02 | [-9.77, -8.25] | 0.0002 |
| math | LoRA (lr 2e-4) | Full FT (lr 2e-5) | +1.92 | [+0.33, +3.54] | 0.014 |
| math | LoRA (lr 2e-4) | Full FT (lr 1e-5, tuned) | -0.43 | [-1.90, +1.01] | 0.61 |
| math | LoRA (lr 5e-4) | Full FT (lr 1e-5, tuned) | -0.45 | [-1.64, +0.73] | 0.4688 |
| math | Full FT (lr 1e-5, tuned) | Full FT (lr 2e-5) | +2.35 | [+1.11, +3.59] | 0.0002 |
| math | QLoRA | LoRA (lr 2e-4) | -1.11 | [-2.78, +0.48] | 0.1812 |
| math | DoRA | LoRA (lr 2e-4) | +0.40 | [-0.33, +1.14] | 0.2884 |
| math | PiSSA | LoRA (lr 2e-4) | +0.23 | [-0.86, +1.26] | 0.6564 |
| math | OFT | LoRA (lr 2e-4) | +0.78 | [-0.51, +2.05] | 0.2604 |
| math | AdaLoRA | LoRA (lr 2e-4) | -0.03 | [-1.36, +1.26] | 0.9684 |
| math | LoKr | LoRA (lr 2e-4) | -0.45 | [-1.72, +0.78] | 0.4744 |
| math | VeRA | LoRA (lr 2e-4) | +0.61 | [-0.78, +1.90] | 0.3916 |

## Learning curves (mean over seeds) and time to 95% of tuned Full FT

- commonsense|full: 0.25→69.3 (250s), 0.50→73.56 (495s), 0.75→75.84 (742s), 1.00→76.27 (991s)
- commonsense|full_lr1e-5: 0.25→72.87 (270s), 0.50→76.11 (537s), 0.75→78.06 (804s), 1.00→78.09 (1071s)
- commonsense|lora: 0.25→73.76 (217s), 0.50→76.7 (433s), 0.75→77.9 (649s), 1.00→78.23 (867s)
- math|full: 0.25→59.31 (382s), 0.50→62.75 (763s), 0.75→67.12 (1149s), 1.00→68.33 (1535s)
- math|full_lr1e-5: 0.25→66.56 (401s), 0.50→68.01 (802s), 0.75→70.2 (1206s), 1.00→70.69 (1608s)
- math|lora: 0.25→69.77 (348s), 0.50→70.08 (701s), 0.75→70.13 (1056s), 1.00→70.25 (1410s)
- math|lora_lr5e-4: 0.25→67.65 (362s), 0.50→69.32 (727s), 0.75→70.06 (1096s), 1.00→70.23 (1462s)

- commonsense|full: target 74.19, time 562.6 s, full tuned run 1070.87 s, speed-up 1.9x
- commonsense|full_lr1e-5: target 74.19, time 378.4 s, full tuned run 1070.87 s, speed-up 2.83x
- commonsense|lora: target 74.19, time 248.1 s, full tuned run 1070.87 s, speed-up 4.32x
- math|full: target 67.16, time 1160.4 s, full tuned run 1607.77 s, speed-up 1.39x
- math|full_lr1e-5: target 67.16, time 565.6 s, full tuned run 1607.77 s, speed-up 2.84x
- math|lora: target 67.16, time 348.5 s, full tuned run 1607.77 s, speed-up 4.61x
- math|lora_lr5e-4: target 67.16, time 361.9 s, full tuned run 1607.77 s, speed-up 4.44x

## 3 epochs (seed 42, eval each epoch)

- commonsense|full: [71.08, 73.95, 74.04]  MMLU 44.21  MATH-500 None  time 3241.7 s
- commonsense|lora: [77.99, 78.6, 78.06]  MMLU 58.63  MATH-500 None  time 2816.2 s
- commonsense|qlora: [77.11, 77.71, 77.36]  MMLU 57.29  MATH-500 None  time 3758.5 s
- commonsense|pissa: [77.59, 77.98, 77.78]  MMLU 60.43  MATH-500 None  time 2606.7 s
- math|full: [63.91, 67.1, 68.08]  MMLU 59.51  MATH-500 32.6  time 4884.4 s
- math|lora: [69.75, 69.83, 69.45]  MMLU 60.54  MATH-500 36.0  time 4392.2 s
- math|qlora: [69.37, 68.16, 67.48]  MMLU 58.13  MATH-500 35.2  time 5856.5 s
- math|pissa: [70.28, 68.76, 70.58]  MMLU 60.4  MATH-500 34.6  time 4091.9 s

## Learning rate / optimizer (seed 42)

- commonsense|full_lr1e-5: 78.38  MMLU 58.47  MATH-500 None
- commonsense|full_lr5e-5: 67.26  MMLU 33.7  MATH-500 None
- commonsense|lora_lr1e-4: 77.73  MMLU 60.65  MATH-500 None
- commonsense|lora_lr5e-4: 77.57  MMLU 56.33  MATH-500 None
- commonsense|full_adamw32: 76.57  MMLU 52.74  MATH-500 None
- math|full_lr1e-5: 70.43  MMLU 59.91  MATH-500 35.0
- math|full_lr5e-5: 61.33  MMLU 56.63  MATH-500 23.6
- math|lora_lr1e-4: 68.69  MMLU 60.74  MATH-500 38.4
- math|lora_lr5e-4: 70.96  MMLU 60.11  MATH-500 38.4
- math|full_adamw32: 68.01  MMLU 59.24  MATH-500 33.0

## Qwen2.5-0.5B

| method | CS | GSM8K | MATH-500 | MMLU CS / Math | mem CS/Math | time CS/Math |
|---|---|---|---|---|---|---|
| base | 30.99 | 26.31 | 9.20 | 47.66 / 47.66 | – / – | – / – |
| full | 52.06 ± 2.51 | 45.51 ± 1.23 | 17.27 ± 0.23 | 38.48 ± 5.86 / 46.94 ± 0.39 | 6.94 ± 0.00 / 9.45 ± 0.00 | 818 ± 17 / 846 ± 8 |
| lora | 60.86 ± 0.98 | 46.37 ± 0.29 | 19.13 ± 1.40 | 44.95 ± 0.71 / 47.24 ± 0.24 | 6.42 ± 0.00 / 11.77 ± 0.00 | 691 ± 2 / 777 ± 13 |
| qlora | 59.65 | 43.14 | 18.00 | 43.09 / 45.52 | 6.55 / 12.03 | 932 / 959 |
| dora | 60.48 | 46.02 | 21.20 | 44.48 / 47.19 | 6.42 / 11.78 | 1141 / 1448 |
| pissa | 59.77 | 46.47 | 18.60 | 47.13 / 47.25 | 6.42 / 11.77 | 650 / 742 |
| vera | 51.72 | 45.56 | 21.00 | 47.43 / 47.55 | 6.30 / 11.65 | 700 / 872 |
| ia3 | 54.60 | 45.41 | 21.00 | 47.54 / 47.22 | 6.29 / 11.64 | 385 / 510 |

## Same micro-batch (4) memory / time

- 1.5B|commonsense: LoRA 5.91 GB vs Full 16.88 GB (−65.0%), time LoRA 1648.6 s vs Full 1070.9 s
- 1.5B|math: LoRA 8.65 GB vs Full 19.34 GB (−55.3%), time LoRA 1721.6 s vs Full 1607.8 s
- 0.5B|commonsense: LoRA 3.74 GB vs Full 6.94 GB (−46.1%), time LoRA 1425.5 s vs Full 830.7 s
- 0.5B|math: LoRA 6.42 GB vs Full 9.45 GB (−32.1%), time LoRA 1417.8 s vs Full 851.0 s
