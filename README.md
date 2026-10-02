# PFNet

PFNet 是一个基于 SemanticKITTI 的点云全景分割实现。GASNv2 提供冻结的语义预测和点特征；实例分支沿用 Panoptic-PolarNet 的极坐标 BEV 表达，回归偏移并通过伪热力图生成实例。

## 网络结构

```text
点云
  └─ GASNv2（冻结）
       ├─ 19 类语义 logits
       └─ 每点 384 维特征
            └─ 极坐标体素化（480 × 360，体素内均值池化）
                 └─ BEV 偏移头（下采样 / 上采样 / 融合）
                      └─ 每体素 2D 偏移（半径、方位角）
```

训练时，实例头以同一实例的极坐标体素中心生成偏移监督，并使用 Smooth L1 损失；GASNv2 不参与反向传播。

推理时，thing 点按预测偏移平移到中心附近，按语义类别计数投影到笛卡尔 BEV，得到伪热力图。局部极大值给出候选中心，邻域投票确定中心类别；同类近邻中心合并后，点被分配给最近中心。

## 环境

已验证组合：Linux、Python 3.6、PyTorch 1.6.0（CUDA 10.2）、torchvision 0.7.0、spconv 2.1.9、torch-scatter 2.0.6。

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch==1.6.0 torchvision==0.7.0 \
  --extra-index-url https://download.pytorch.org/whl/cu102
pip install spconv-cu102==2.1.9
pip install torch-scatter==2.0.6 \
  -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
pip install numpy==1.19.5 scipy==1.5.4 scikit-learn==0.24.2 \
  numba==0.53.1 PyYAML==5.4.1 easydict==1.10 \
  tensorboard==2.7.0 tqdm open3d==0.13.0
```

需要可用的 NVIDIA GPU。运行前可检查：

```bash
python -c "import torch, spconv.pytorch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## 数据与语义权重

下载 SemanticKITTI 后，将 `DATASET_PATH` 指向其 `sequences` 目录：

```text
<dataset-root>/sequences/
├── 00/velodyne/
├── 00/labels/
├── 08/                 # 验证集
└── 11/ ... 21/         # 测试集
```

编辑 [`cfgs/pfnet.yaml`](cfgs/pfnet.yaml)：

```yaml
DATA_CONFIG:
  DATASET_PATH: /path/to/SemanticKITTI/sequences
```

语义训练：`python cfg_train.py --config cfgs/gasn_semantic.yaml --batch_size 6 --log_dir ./output --tag gasn_semantic`。语义 checkpoint 输出到 `output/gasn_semantic/ckpt/checkpoint_epoch_*.pth`；训练实例头前，将 `cfgs/pfnet.yaml` 的 `MODEL.SEM_PRETRAIN` 改为该 checkpoint 的绝对路径。

## 使用

训练实例头：

```bash
python cfg_train.py --config cfgs/pfnet.yaml \
  --batch_size 4 --log_dir ./output --tag pfnet_train
```

从已有检查点恢复训练：

```bash
python cfg_train.py --config cfgs/pfnet.yaml \
  --batch_size 4 --log_dir ./output --tag pfnet_train \
  --ckpt_name checkpoint_epoch_<N>_<...>.pth
```

验证：

```bash
python cfg_train.py --config cfgs/pfnet.yaml --onlyval \
  --batch_size 4 --pretrained_ckpt /path/to/checkpoint.pth \
  --log_dir ./output --tag pfnet_val
```

测试并导出 SemanticKITTI 格式预测：

```bash
python cfg_train.py --config cfgs/pfnet.yaml --onlytest \
  --batch_size 4 --pretrained_ckpt /path/to/checkpoint.pth \
  --log_dir ./output --tag pfnet_test
```

训练日志、TensorBoard 摘要、检查点和测试预测分别保存在 `output/<tag>/` 下。
