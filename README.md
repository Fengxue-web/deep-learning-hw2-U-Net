# HW2 U-Net

本仓库对应深度学习课程作业 HW2 任务 3，内容为从零实现 U-Net 语义分割模型，并在 Stanford Background Dataset 上完成像素级训练与评估。

本项目使用 PyTorch 基础 API 搭建模型，不使用任何预训练权重。实验比较以下三种 loss 配置：

- `ce`: Cross-Entropy Loss
- `dice`: 手动实现的 Dice Loss
- `ce_dice`: Cross-Entropy Loss + Dice Loss

## 项目结构

```text
hw2_U-Net/
├── checkpoints/              # 保存训练得到的 best checkpoint
├── data/
│   └── stanford_background/   # 放置 Stanford Background Dataset
├── outputs/
│   ├── figures/              # 保存报告用图片或截图
│   └── logs/                 # 保存 CSV 训练日志
├── src/
│   ├── dataset.py            # Stanford Background Dataset 读取
│   ├── evaluate.py           # 加载 checkpoint 并评估
│   ├── losses.py             # CE、Dice、CE+Dice loss
│   ├── metrics.py            # pixel accuracy 和 mIoU
│   ├── model_unet.py         # 从零实现的 U-Net
│   └── train.py              # 训练入口
├── requirements.txt
└── README.md
```

## 环境

在 Python 环境中安装依赖：

```powershell
pip install -r requirements.txt
```

本项目主要依赖 PyTorch。若使用 GPU 训练，需要确保 PyTorch 与本机 CUDA 版本匹配。环境安装后可通过下文的 sanity check 命令验证各模块是否可运行。

## 数据集

Stanford Background Dataset 需解压至以下项目内路径：

```text
data/stanford_background/iccv09Data/
```

代码读取的数据结构为：

```text
data/stanford_background/iccv09Data/
├── images/   # .jpg 图像文件
└── labels/   # *.regions.txt 语义分割标签文件
```

数据划分与标签设置：

- 总样本数：715
- 训练集：572
- 验证集：71
- 测试集：72
- 有效类别数：8
- `ignore_index`: `-1`

数据集 sanity check：

```powershell
python src/dataset.py
```

该命令用于验证数据读取、图像 tensor shape、mask shape 和标签范围，不会启动训练。

## 模型

U-Net 模型定义在：

```text
src/model_unet.py
```

模型实现：

- 使用 PyTorch 基础 API 从零实现
- 不使用预训练权重
- 包含编码器、解码器和 skip connection
- 输入 shape: `[B, 3, H, W]`
- 输出 logits shape: `[B, num_classes, H, W]`

模型 sanity check：

```powershell
python src/model_unet.py
```

## 损失函数

Loss 定义在：

```text
src/losses.py
```

实现以下配置：

```text
ce       Cross-Entropy Loss
dice     manually implemented Dice Loss
ce_dice  Cross-Entropy Loss + Dice Loss
```

损失函数 sanity check：

```powershell
python src/losses.py
```

该命令验证三种 loss 是否能输出标量，并能否正常反向传播。

## 评估指标

评估指标定义在：

```text
src/metrics.py
```

实现指标：

- pixel accuracy
- mean IoU, 即 mIoU

评估指标 sanity check：

```powershell
python src/metrics.py
```

## 训练

Smoke test 命令:

```powershell
python src/train.py --loss ce --epochs 1 --batch-size 2
python src/train.py --loss dice --epochs 1 --batch-size 2
python src/train.py --loss ce_dice --epochs 1 --batch-size 2
```

三种损失配置的完整训练命令：

```powershell
python src/train.py --loss ce --epochs 50 --batch-size 4 --logger swanlab --max-train-batches 0 --max-val-batches 0
python src/train.py --loss dice --epochs 50 --batch-size 4 --logger swanlab --max-train-batches 0 --max-val-batches 0
python src/train.py --loss ce_dice --epochs 50 --batch-size 4 --logger swanlab --max-train-batches 0 --max-val-batches 0
```

主要参数：

- `--loss`: 选择 loss 配置，可选 `ce`、`dice`、`ce_dice`
- `--epochs`: 训练 epoch 数
- `--batch-size`: batch size
- `--logger`: 日志工具，可选 `none`、`wandb`、`swanlab`
- `--max-train-batches 0`: 训练时完整遍历训练集
- `--max-val-batches 0`: 验证时完整遍历验证集

训练脚本会保存对应的 best checkpoint：

```text
checkpoints/unet_ce_best.pth
checkpoints/unet_dice_best.pth
checkpoints/unet_ce_dice_best.pth
```

CSV 训练日志会保存到：

```text
outputs/logs/train_ce.csv
outputs/logs/train_dice.csv
outputs/logs/train_ce_dice.csv
```

## Evaluation

验证集评估命令：

```powershell
python src/evaluate.py --checkpoint checkpoints/unet_ce_best.pth --split val --batch-size 4 --max-batches 0
python src/evaluate.py --checkpoint checkpoints/unet_dice_best.pth --split val --batch-size 4 --max-batches 0
python src/evaluate.py --checkpoint checkpoints/unet_ce_dice_best.pth --split val --batch-size 4 --max-batches 0
```

测试集评估命令：

```powershell
python src/evaluate.py --checkpoint checkpoints/unet_ce_best.pth --split test --batch-size 4 --max-batches 0
python src/evaluate.py --checkpoint checkpoints/unet_dice_best.pth --split test --batch-size 4 --max-batches 0
python src/evaluate.py --checkpoint checkpoints/unet_ce_dice_best.pth --split test --batch-size 4 --max-batches 0
```
