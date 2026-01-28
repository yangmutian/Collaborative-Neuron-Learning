# 协同神经元学习 (Collaborative Neuron Learning, CNL)

基于梯度方向选择性更新神经元，缓解大语言模型微调中的灾难性遗忘问题。

## 概述

在对大语言模型注入新知识时，模型往往会遗忘已掌握的知识，即**灾难性遗忘**。本项目实现了**协同神经元学习 (CNL)** 方法，通过分析神经元级别的梯度方向，选择性地只更新对"学习新知识"和"保留旧知识"都有正向贡献的神经元。

### 核心思想

![框架图](frame.jpg)

**核心概念：**

| 术语 | 定义 |
|------|------|
| **掌握集 (Mastered Set, M)** | 模型已经能正确回答的样本 |
| **注入集 (Injection Set, I)** | 模型当前回答错误的样本（待学习的新知识） |
| **梯度相似度** | $S(\mathcal{M}, \mathcal{I}) = \nabla_{\theta} \mathcal{L}^{\mathcal{M}}(\theta)^{\top} \nabla_{\theta} \mathcal{L}^{\mathcal{I}}(\theta)$ |
| **神经元贡献** | $s_{\theta_j}(\mathcal{M}, \mathcal{I}) = \nabla_{\theta_j} \mathcal{L}^{\mathcal{M}}(\theta) \cdot \nabla_{\theta_j} \mathcal{L}^{\mathcal{I}}(\theta)$ |

**神经元分类：**

- **协同神经元** ($s_{\theta_j} \geq 0$)：梯度方向一致 → 允许更新
- **矛盾神经元** ($s_{\theta_j} < 0$)：梯度方向冲突 → 冻结不更新

**CNL 更新规则：**

$$\theta \gets \theta - \eta \cdot \mathbb{I}[s_{\theta}(\mathcal{M}, \mathcal{I}) \geq 0] \odot \nabla_{\theta} L^{\mathcal{I}}(\theta)$$

## 项目结构

```
├── infer/                  # 推理与数据划分
│   ├── infer.py
│   └── infer_*.sh
│
├── sft/                    # CNL 微调（主方法）
│   ├── sft.py
│   └── sft_*.sh
│
├── neuron/                 # 神经元级梯度分析
│   ├── neuron.py
│   └── neuron_*.sh
│
├── distance/               # 梯度点积分析
│   ├── distance.py
│   ├── neg_distance.py
│   └── distance_*.sh
│
├── sft_lora/               # LoRA 基线方法
├── sft_moment/             # CNL + 动量
├── sft_adam/               # CNL + Adam 优化器
├── sft_adamw/              # CNL + AdamW 优化器
├── sft_replay/             # Replay 基线方法（混合训练）
├── sft_ood_correct/        # OOD 实验（正确样本）
└── sft_ood_wrong/          # OOD 实验（错误样本）
```

## 快速开始

### 0. 数据格式

输入数据应为 JSONL 文件，每一行遵循以下格式：

```json
{
  "label": "A",
  "question": "Question: [具体问题]\nOptions:\nA: [选项 A]\nB: [选项 B]\nC: [选项 C]\nD: [选项 D]\n\nPlease reply with only A, B, C, or D. Do not provide any explanation."
}
```

### 1. 推理与划分数据

将数据集划分为正确样本（掌握集）和错误样本（注入集）：

```bash
bash infer/infer_qwen_1.5b.sh csqa
```

接着，将数据集进一步划分为两部分（用于 Replay 方法或 OOD 实验）：

```bash
# 划分掌握集 (用于 Replay / OOD-Correct)
python infer/split_data.py \
  --input_jsonl ./data/csqa_correct_Qwen2.5-1.5B-Instruct.jsonl \
  --output1 ./data/csqa_correct1_Qwen2.5-1.5B-Instruct.jsonl \
  --output2 ./data/csqa_correct2_Qwen2.5-1.5B-Instruct.jsonl

# 划分注入集 (用于 OOD-Wrong)
python infer/split_data.py \
  --input_jsonl ./data/csqa_wrong_Qwen2.5-1.5B-Instruct.jsonl \
  --output1 ./data/csqa_wrong1_Qwen2.5-1.5B-Instruct.jsonl \
  --output2 ./data/csqa_wrong2_Qwen2.5-1.5B-Instruct.jsonl
```

生成文件：
- `data/{dataset}_correct_{model}.jsonl` — 掌握集
- `data/{dataset}_wrong_{model}.jsonl` — 注入集
- `data/{dataset}_correct1/2_{model}.jsonl` — 掌握集划分
- `data/{dataset}_wrong1/2_{model}.jsonl` — 注入集划分

### 2. CNL 训练

```bash
# 使用 CNL（use_freeze=1）
bash sft/sft_qwen_1.5b.sh csqa

# 不使用 CNL — 传统微调（use_freeze=0）
# 修改脚本中 USE_FREEZE=0
```

### 3. 查看结果

查看输出目录下的 `summary.csv`：

| 列名 | 含义 |
|------|------|
| `epoch` | 训练轮次 |
| `train_avg_loss` | 平均训练损失 |
| `wrong_to_correct` | 注入集：错→对 = **学习** |
| `correct_to_wrong` | 掌握集：对→错 = **遗忘** |

### 4. 梯度分析

分析神经元级别的梯度统计：

```bash
bash neuron/neuron_qwen_1.5b.sh csqa
```

分析梯度点积分布：

```bash
bash distance/distance_qwen_1.5b.sh csqa
python distance/neg_distance.py
```

## 支持的模型

- Qwen2.5-1.5B/3B/7B-Instruct
- Llama-3.2-1B/3B-Instruct

## 命令行参数

所有脚本的第一个参数为数据集名称（默认 `csqa`）：

```bash
bash sft/sft_qwen_1.5b.sh arc_c      # ARC-Challenge
bash sft/sft_qwen_1.5b.sh csqa       # CommonsenseQA
bash sft/sft_qwen_1.5b.sh medqa      # MedQA
bash sft/sft_qwen_1.5b.sh mmlu       # MMLU
```

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use_freeze` | 1: 启用 CNL，0: 禁用 CNL | 1 |
| `--epochs` | 训练轮数 | 25 |

## 实验环境

- **Python**: 3.12.2
- **推理精度**: FP32
- **复现性**: Temperature 设置为 0

## 依赖

- `torch==2.2.0`
- `transformers==4.48.0`
- `tqdm==4.67.1`
- `numpy==2.2.6`
- `peft==0.12.0`

## License

 Apache-2.0 license
