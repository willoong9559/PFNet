# PFNet

PFNet 是用于 SemanticKITTI 点云全景分割的 PyTorch 实现。它保留 GASNv2 语义骨干，并采用 Panoptic-PolarNet 风格的极坐标实例头。

## 运行环境

- Python 3.7
- PyTorch 1.6
- torchvision 0.7.0
- CUDA（需要 GPU；代码会直接调用 CUDA）
- spconv 1.1
- torch-cluster 1.6.0
- torch-scatter 2.0.8
- numpy、scipy、scikit-learn、numba、PyYAML、easydict、tensorboard 2.7.0、tqdm

建议使用与 PyTorch 1.6 和 CUDA 版本匹配的 `spconv`、`torch-cluster`、`torch-scatter` 构建包。

## 数据准备

下载 SemanticKITTI 数据集，并按以下目录放置：

```text
data/
└── sequences/
    ├── 00/
    │   ├── velodyne/
    │   │   └── 000000.bin
    │   └── labels/
    │       └── 000000.label
    ├── 08/                  # 验证集
    └── 11/ ... 21/          # 测试集
```

默认数据路径为 `./data/sequences`。如使用其他位置，请修改 [cfgs/pfnet.yaml](cfgs/pfnet.yaml) 中的 `DATASET_PATH`。

## 权重准备

创建 `weights/` 目录，并放入：

- 语义骨干权重：配置中 `MODEL.SEM_PRETRAIN` 指向的文件（默认 `weights/kitti_backbone_v2.pth`）。
- PFNet 完整模型权重：用于验证、测试或继续训练时，通过 `--pretrained_ckpt` 指定。该权重包含极坐标实例头。

## 两阶段流程与 GASNv2

PFNet 的语义骨干是 `GASNv2`，实现位于 [`models/pcd_seg_v2.py`](models/pcd_seg_v2.py)。PFNet 会在构建模型时自动创建该骨干，并从 `MODEL.SEM_PRETRAIN` 加载其权重；实例头训练时，语义骨干参数会被冻结，不参与反向传播。GASN 的原始训练代码见 [ItIsFriday/PcdSeg](https://github.com/ItIsFriday/PcdSeg)。

### GASNv2 特征提取与融合

GASNv2 输入每个 LiDAR 点的 `(x, y, z, intensity)`，并在多个体素尺度上构造局部几何特征：原始点特征、点相对同体素点均值的偏移，以及点相对体素中心的偏移。随后通过 VFE 编码、跨尺度特征聚合和四级稀疏 3D CNN 提取三维上下文。

骨干还会将稀疏 3D 体素沿高度维聚合为稠密 BEV 特征图，经轻量 2D U-Net 增强后，再按每个点的 XY 坐标回采样为 BEV 点特征。最后拼接四级 3D CNN 的点级投影特征、初始多尺度几何融合特征与 BEV 点特征：

```text
点云 (x, y, z, intensity)
  → 多尺度 VFE / 几何特征
  → 稀疏 3D CNN × 4 ────────────────┐
  → 高度聚合 → BEV 2D U-Net → 点采样 ├→ 6 组 64 维特征拼接（384 维）
  → 多尺度几何融合 ─────────────────┘
                                      → 语义分类头 → 19 类 logits
```

这 384 维点特征一方面送入语义分类头，另一方面在 PFNet 中投影到极坐标 BEV 网格，作为 Panoptic-PolarNet 风格实例头的输入。实例头预测类别无关的中心热图与每个网格单元到中心的二维偏移，再以最近中心投票生成实例 ID。

完整流程分为两个彼此独立的阶段：

1. **语义骨干阶段**：训练 GASNv2 的 19 类点语义分割能力，产出语义骨干权重。GASNv2 使用 point-level CE、Lovász loss 和多尺度体素辅助监督。
2. **实例阶段**：加载并冻结 GASNv2；将其点特征投影到极坐标 BEV 网格，训练中心热图与偏移回归。推理时，对检测到的中心做 NMS，并把每个 thing 点按预测偏移投票给最近中心。该部分采用 [Panoptic-PolarNet](https://github.com/edwardzhou130/panoptic-polarnet) 的中心与偏移实例表示。

当前仓库包含 GASNv2 的网络定义和权重加载逻辑，但**未提供独立训练 GASNv2 的配置、脚本或命令入口**。请使用 [ItIsFriday/PcdSeg](https://github.com/ItIsFriday/PcdSeg) 完成第一阶段训练，或准备其语义骨干权重；随后在 `MODEL.SEM_PRETRAIN` 中指定权重路径。`cfgs/pfnet.yaml` 的训练命令只训练第二阶段的极坐标实例头，不会重新训练语义骨干。

## 运行

所有命令使用 `--config` 指定配置文件。

验证完整 PFNet：

```bash
python cfg_train.py --config cfgs/pfnet.yaml --onlyval \
  --pretrained_ckpt /path/to/kitti_pfnet.pth
```

在 SemanticKITTI 测试集生成预测：

```bash
python cfg_train.py --config cfgs/pfnet.yaml --onlytest \
  --pretrained_ckpt /path/to/kitti_pfnet.pth \
  --log_dir ./output --tag pfnet_test
```

训练极坐标实例头：

```bash
python cfg_train.py --config cfgs/pfnet.yaml \
  --log_dir ./output --tag pfnet_train
```

单机多卡训练可使用：

```bash
bash scripts/pytorch_train.sh <gpu_num> <batch_size> cfgs/pfnet.yaml ./output pfnet_train
```

输出内容会写入 `<log_dir>/<tag>/`，包括日志、TensorBoard 摘要、检查点和测试预测结果。

## 配置说明

- `cfgs/pfnet.yaml`：GASNv2 语义骨干与极坐标实例头的配置。
- `MODEL.POLAR_INSTANCE`：极坐标网格分辨率、最大半径、中心 NMS 阈值、中心数上限和两项实例损失的权重。
