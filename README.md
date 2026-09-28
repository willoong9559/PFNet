# PFNet

PFNet 是一个 SemanticKITTI 点云全景分割实现：使用冻结的 GASNv2 语义骨干，实例分支在极坐标 BEV 上回归点到实例中心的偏移，并以伪热力图完成实例聚类。

## 运行环境

- Python 3.7
- PyTorch 1.6
- torchvision 0.7.0
- CUDA（需要 GPU；代码直接调用 CUDA）
- spconv 1.1
- torch-cluster 1.6.0
- torch-scatter 2.0.8
- numpy、scipy、scikit-learn、numba、PyYAML、easydict、tensorboard 2.7.0、tqdm

建议安装与 PyTorch 1.6、CUDA 版本匹配的 `spconv`、`torch-cluster` 和 `torch-scatter` 构建包。

## 数据准备

下载 SemanticKITTI，并按以下目录放置：

```text
data/
└── sequences/
    ├── 00/
    │   ├── velodyne/000000.bin
    │   └── labels/000000.label
    ├── 08/                  # 验证集
    └── 11/ ... 21/          # 测试集
```

默认路径是 `./data/sequences`；其他位置请修改 [`cfgs/pfnet.yaml`](cfgs/pfnet.yaml) 的 `DATASET_PATH`。

## 语义骨干权重

PFNet 的语义骨干为 GASNv2，网络定义在 [`models/pcd_seg_v2.py`](models/pcd_seg_v2.py)。在实例阶段，它会从 `MODEL.SEM_PRETRAIN`（默认 `weights/kitti_backbone_v2.pth`）加载并冻结。第一阶段语义训练请使用 [GASN/PcdSeg](https://github.com/ItIsFriday/PcdSeg) 的代码和权重；本仓库只训练实例分支。

## 实例伪热力图后处理

实例头仅学习二维偏移，不再预测或训练中心热图。推理时会：

1. 将每个 thing 点按预测偏移平移到实例中心附近。
2. 将偏移后的 thing 体素按语义类别计数投影到笛卡尔 BEV 网格，得到 19 通道类别计数图；沿语义通道相加即为伪热力图。
3. 在伪热力图上以局部最大值 NMS 提取候选中心；以候选点邻域的类别计数确定中心类别。
4. 对同类别、距离小于类别合并半径的中心合并，再将每个有效 thing 点分配给最近中心。

伪热力图网格大小、NMS 核、类别投票核和类别合并半径由 `MODEL.POLAR_INSTANCE` 配置。`CENTER_GROUP_RADII` 是当前针对 SemanticKITTI thing 类设置的初始值，需要按目标传感器和类别尺度调参。

这改变了实例头的推理结构；已有的 `weights/kitti_pfnet.pth` 可通过 `--pretrained_ckpt` 按匹配参数作为初始化加载，但不应作为新伪热力图后处理的最终权重。需要重新训练实例阶段；GASNv2 语义骨干权重仍然可复用。

## 运行

训练新的实例头：

```bash
python cfg_train.py --config cfgs/pfnet.yaml \
  --log_dir ./output --tag pfnet_ph_train
```

验证或测试时使用新训练生成的实例检查点：

```bash
python cfg_train.py --config cfgs/pfnet.yaml --onlyval \
  --pretrained_ckpt /path/to/pfnet_ph.pth

python cfg_train.py --config cfgs/pfnet.yaml --onlytest \
  --pretrained_ckpt /path/to/pfnet_ph.pth \
  --log_dir ./output --tag pfnet_ph_test
```

单机多卡训练：

```bash
bash scripts/pytorch_train.sh <gpu_num> <batch_size> cfgs/pfnet.yaml ./output pfnet_ph_train
```

输出会写入 `<log_dir>/<tag>/`，包含日志、TensorBoard 摘要、检查点和测试预测结果。
