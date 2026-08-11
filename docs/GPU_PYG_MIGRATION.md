# UV-Net 现代 GPU / PyG 迁移、复现与验证说明

本文记录本仓库从原始 DGL 0.6 / PyTorch 1.8 训练栈迁移到现代
PyTorch Geometric（PyG）GPU 训练栈的设计、实现与验证过程。目标是让后续维护者
能够理解每项改动为何存在，并能在不修改系统 Python、全局包或系统 CUDA 配置的
前提下重新搭建环境、转换数据、训练和测试。

## 1. 迁移目标和边界

原始 UV-Net 代码以 2021 年的软件生态为基准：Python 3.9、PyTorch 1.8、
DGL 0.6.1、PyTorch Lightning 1.3 和 TorchMetrics 0.3。这个环境仍然可以在
CPU 上读取官方 DGL `.bin` 数据，但不适合作为 RTX 50 系列（Blackwell）等新 GPU
的长期训练环境。

本次采用并行保留的方式：

- 原始代码和 DGL 环境不删除，作为数据语义和模型实现的参考。
- `.venv` 使用 Python 3.9 和 DGL 0.6.1，只负责读取历史 `.bin` 文件。
- `.venv-gpu` 使用现代 PyTorch、CUDA wheel 和 PyG，负责训练与推理。
- DGL 数据先无损转换为框架中立的 NPZ；现代训练过程不依赖 DGL。
- 数据、环境、缓存和训练结果都放在仓库目录内，并通过 `.gitignore` 排除。

当前现代化入口覆盖 **MFCAD face segmentation**。原始 SolidLetters 分类和
Fusion 360 Gallery segmentation 尚未迁移到 PyG。

## 2. 已验证环境

本次实际验证的平台：

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti，compute capability 12.0 |
| Python | 3.12.13 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| PyTorch Geometric | 2.7.0 |
| torch-scatter | 2.1.2，PyTorch 2.8 / CUDA 12.8 wheel |
| Lightning | 2.6.5 |
| TorchMetrics | 1.9.0 |
| NumPy | 2.5.2 |

PyTorch 的 CUDA wheel 自带所需的 CUDA runtime。日常训练不需要修改全局 Python，
也不需要为了该项目修改系统 CUDA toolkit；宿主机只需提供足够新的 NVIDIA 驱动并
能通过 `nvidia-smi` 正常识别显卡。

## 3. 使用 uv 创建隔离的 GPU 环境

所有命令均在仓库根目录执行：

```bash
cd ~/workspace/Work/UV-Net
uv venv --python 3.12 .venv-gpu
```

先从 PyTorch 官方 CUDA 12.8 wheel 索引安装 PyTorch：

```bash
UV_CACHE_DIR=.cache/uv uv pip install \
  --python .venv-gpu/bin/python \
  torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

再安装仓库记录的现代训练依赖：

```bash
UV_CACHE_DIR=.cache/uv uv pip install \
  --python .venv-gpu/bin/python \
  -r requirements-gpu.txt
```

最后安装与 PyTorch/CUDA 组合严格匹配的 `torch-scatter` wheel：

```bash
UV_CACHE_DIR=.cache/uv uv pip install \
  --python .venv-gpu/bin/python \
  torch-scatter==2.1.2 \
  --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

这里显式指定 `.venv-gpu/bin/python`，不会写入系统 Python。`UV_CACHE_DIR` 也被限制
在仓库的 `.cache/uv` 中。

检查依赖和 CUDA：

```bash
UV_CACHE_DIR=.cache/uv uv pip check --python .venv-gpu/bin/python
.venv-gpu/bin/python gpu_smoke_test.py
```

`gpu_smoke_test.py` 不只是调用 `torch.cuda.is_available()`，还会在真实 GPU 上执行：

1. 2D CNN、BatchNorm、激活和池化；
2. PyG `NNConv` edge-conditioned message passing；
3. 图级 max pooling；
4. loss 和反向传播；
5. CUDA 同步。

因此 `CUDA dense + PyG forward/backward: PASS` 能同时验证 CUDA kernel、PyG 图算子
和梯度链路。

## 4. 为什么先把 DGL `.bin` 转成中立格式

官方 MFCAD 下载包已经包含 UV-Net 所需的 B-rep 派生特征。每个 DGL 图中：

- node 对应一个 B-rep face；
- `ndata["x"]` 是 `float32 [num_faces, 10, 10, 7]` 的 face UV-grid；
- edge 对应 face-adjacency graph 中的一条有向邻接边；
- `edata["x"]` 是 `float32 [num_edges, 10, 6]` 的 edge UV-grid；
- JSON 中每个 face 的 `segment.index` 是类别标签，范围为 0–15。

如果直接从 STEP 重新生成数据，OpenCascade、采样代码、拓扑遍历顺序或几何容差的
变化都可能改变输入，难以区分“框架迁移差异”和“数据重新生成差异”。因此本次先
精确保留官方输入，只替换存储容器和图框架。

转换命令必须使用能够读取 DGL 0.6 文件的旧环境：

```bash
DGLBACKEND=pytorch .venv/bin/python \
  -m process.convert_mfcad_dgl_to_npz \
  ./data/mfcad --workers 8
```

转换器执行以下操作：

1. 从 `split.json` 读取官方 train/validation/test 划分并检查重复 ID；
2. 用 `dgl.data.utils.load_graphs` 读取每个 `.bin`；
3. 按 DGL edge ID（EID）顺序读取 `src`、`dst` 和 edge feature；
4. 从 JSON 按 face 顺序读取 label；
5. 检查数组维度、face/label 数量、edge/feature 数量、索引范围和标签范围；
6. 原子写入压缩 NPZ，避免中断后留下半个文件；
7. 写入带 schema、特征规格和 split 数量的 `manifest.json`。

每个 `data/mfcad/neutral/<sample_id>.npz` 包含：

| 字段 | dtype 和形状 | 含义 |
| --- | --- | --- |
| `face_uv` | `float32 [N, 10, 10, 7]` | face 坐标、法向和 trimming mask |
| `edge_uv` | `float32 [E, 10, 6]` | edge 坐标和切向量 |
| `edge_index` | `int64 [2, E]` | PyG/DGL 通用的 COO 邻接关系 |
| `node_y` | `int64 [N]` | 每个 face 的类别 |
| `sample_id` | string | 原始样本 ID |
| `schema_version` | int16 scalar | 中立格式版本 |

转换器默认可恢复：已有输出会跳过；需要重新生成时使用 `--overwrite`。

## 5. 数据转换验证

官方 split 一共转换 15,461 个样本：

| split | 样本数 |
| --- | ---: |
| train | 9,277 |
| validation | 3,090 |
| test | 3,094 |

下载包中另有 27 个 `.bin` 不属于官方 `split.json`，因此没有纳入训练、验证或测试，
也没有转换。这是有意遵循官方划分，不是转换遗漏。

本次对 15,461 个官方样本做了逐文件精确比较：

- NPZ `face_uv` 与 DGL `ndata["x"]` 完全相等；
- NPZ `edge_uv` 与 DGL `edata["x"]` 完全相等；
- NPZ `edge_index` 与 DGL `edges(order="eid")` 完全相等；
- NPZ `node_y` 与原始 JSON label 完全相等；
- 没有缺失文件或失败样本。

转换阶段保存原始未归一化数组。center/scale 和可选的 90 度轴旋转仍在 Dataset
读取阶段完成，这与原始训练管线的职责划分一致。

## 6. PyG Dataset

`datasets/pyg_mfcad.py` 提供 `MFCADPyGDataset`：

- 继续使用官方 `split.json`，不重新随机划分数据；
- 加载 NPZ 并创建标准 `torch_geometric.data.Data`；
- `face_uv` 作为 node-aligned tensor，`edge_uv` 作为 edge-aligned tensor；
- `edge_index` 直接采用 COO `[source, destination]`；
- `y` 保存逐 face 标签；
- `sample_id` 用于将批处理结果还原到原始 CAD；
- center/scale 只使用 trimming mask 可见点，与原实现一致；
- `--random_rotate` 可启用沿规范轴的随机 90 度旋转。

PyG `DataLoader` 会把多个 CAD 合并成一张不相连的大图，并额外创建 `batch` 向量，
标记每个 face 属于哪个 CAD。训练只对 train loader 使用 `drop_last=True`；validation
和 test 使用 `drop_last=False`，确保最后一个不足 batch size 的批次也被评估。

## 7. DGL 模型到 PyG 模型的语义映射

迁移不是重新设计网络，而是保留原 UV-Net 的参数规模、层次和消息传递数学含义：

| 原始组件 | PyG 实现 | 保留内容 |
| --- | --- | --- |
| Curve encoder | `UVNetCurveEncoder` | 3 层 Conv1d、BN、LeakyReLU、全局平均池化、FC |
| Surface encoder | `UVNetSurfaceEncoder` | 3 层 Conv2d、BN、LeakyReLU、全局平均池化、FC |
| DGL NNConv | PyG `NNConv` | edge network 生成权重、sum aggregation |
| DGL max node pooling | `global_max_pool` | 每个 CAD 的 graph embedding |
| Edge update MLP | `_EdgeConv` | source/destination 投影、edge residual、BN/激活 |
| Segment classifier | `_NonLinearClassifier` | 512/256 hidden layers、BN、dropout、16 类输出 |

PyG `NNConv` 明确配置：

```python
NNConv(
    in_channels=node_feats,
    out_channels=out_feats,
    nn=edge_network,
    aggr="add",
    root_weight=False,
    bias=False,
)
```

`root_weight=False` 和 `bias=False` 很重要，否则 PyG 会额外加入 root transform 或 bias，
不再等价于原始 DGL 层。

模型参数总数为 1,366,612，与原始 DGL 模型一致。对于旧 DGL state dict，
`convert_dgl_state_dict()` 将 `.gconv.edge_func.` 映射为 PyG 的 `.gconv.nn.`；其余参数名
保持一致。

## 8. 模型等价性验证

使用相同随机种子初始化原始 DGL 模型和 PyG 模型，把 DGL 权重严格映射到 PyG 后，
在两个真实 MFCAD 图上比较各层输出。最大绝对误差为：

| 输出 | 最大绝对误差 |
| --- | ---: |
| curve embedding | `1.19e-6` |
| surface embedding | `7.15e-7` |
| node embedding | `4.29e-6` |
| graph embedding | `2.15e-6` |
| local/global concat | `4.29e-6` |
| logits | `5.25e-6` |

在 `rtol=1e-4, atol=1e-5` 下全部通过。该量级来自 PyTorch 1.8 与 2.8、DGL 与
PyG kernel 的浮点累加顺序差异，没有发现模型语义偏移。

## 9. Lightning 2.x 训练与测试入口

`segmentation_pyg.py` 是现代训练入口，主要改动包括：

- 使用 `lightning` 2.x 的 `Trainer`、`ModelCheckpoint` 和 `TensorBoardLogger`；
- 使用 `accelerator`、`devices` 和 `precision` 代替旧版 `--gpus` 参数；
- 使用现代 `MulticlassJaccardIndex` 和 `MulticlassAccuracy`；
- 指标按 epoch 聚合，并支持 distributed sync；
- 固定 seed 时同时设置 DataLoader worker seed；
- 按最低 `val_loss` 保存 `best.ckpt`，同时保存 `last.ckpt`；
- 提供 train/val/test batch 限制，便于低成本 smoke test；
- FP32 基线确认后可使用 `--precision bf16-mixed`；
- 测试可选输出逐 face 的真实类别、预测类别、置信度和状态。

快速训练检查：

```bash
.venv-gpu/bin/python segmentation_pyg.py train \
  --dataset_path ./data/mfcad \
  --max_epochs 1 --batch_size 16 --num_workers 0 \
  --limit_train_batches 10 --limit_val_batches 2 \
  --accelerator gpu --devices 1 --precision 32-true \
  --seed 42 --experiment_name mfcad-pyg-smoke
```

完整 100 epoch FP32 基线：

```bash
.venv-gpu/bin/python segmentation_pyg.py train \
  --dataset_path ./data/mfcad \
  --max_epochs 100 --batch_size 64 --num_workers 4 \
  --accelerator gpu --devices 1 --precision 32-true \
  --seed 42 --experiment_name mfcad-pyg-fp32-100ep
```

测试最佳检查点：

```bash
.venv-gpu/bin/python segmentation_pyg.py test \
  --dataset_path ./data/mfcad \
  --batch_size 64 --num_workers 4 \
  --accelerator gpu --devices 1 --precision 32-true --seed 42 \
  --checkpoint ./results/mfcad-pyg-fp32-100ep/<date>/<time>/best.ckpt
```

## 10. 逐 face 推理报告

默认测试只输出整体 mIoU。加以下参数可以检查单个 CAD 内每个 face：

```bash
--verbose_predictions --prediction_samples 3
```

报告包含 sample ID、face 编号、真实 class ID、预测 class ID、softmax 置信度、是否
匹配以及每个 CAD 的 face accuracy。详细模式复用同一次 forward 的 logits，不会进行
第二次推理，也不会改变模型权重或指标。

排错时使用：

```bash
--verbose_predictions --prediction_samples 10 --prediction_errors_only
```

该模式会扫描测试集、跳过完全正确的 CAD，并只显示前 10 个含错误 face 的样本。

MFCAD 数据包没有随当前代码提供可靠的 class ID 到语义名称映射，因此报告只显示
0–15 的 class ID，避免把类别错误命名。

## 11. 已完成的训练和最终结果

已完成的验证包括：

- GPU CNN + PyG NNConv + pooling + backward smoke test：通过；
- 真实 MFCAD batch=16 的 GPU forward/backward：通过；
- 1 epoch 全 train/validation：通过；
- BF16 mixed precision smoke test：通过；
- 100 epoch、batch=64、FP32、seed=42：通过；
- 最佳 checkpoint 的完整 3,094 样本测试：通过。

100 epoch 最佳 checkpoint 来自 zero-based `epoch=86`（第 87 个 epoch），最佳
`val_loss` 约为 0.0020。完整测试集结果：

| 指标 | 结果 |
| --- | ---: |
| 测试 CAD 数 | 3,094 |
| 测试 face 数 | 69,411 |
| face correct / error | 69,388 / 23 |
| face accuracy | 99.9669% |
| mean confidence | 99.9836% |
| macro mIoU | **99.9256%** |

一个 seed 的结果可以验证实现和复现链路，但严谨实验应使用多个 seed，报告均值和标准
差；若要对齐论文训练方案，还应明确记录 epoch 数、数据增强和指标定义。

## 12. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `requirements-gpu.txt` | 现代 GPU 环境顶层依赖和特殊 wheel 安装提示 |
| `gpu_smoke_test.py` | 真实 CUDA/PyG forward/backward 检查 |
| `process/convert_mfcad_dgl_to_npz.py` | DGL `.bin` 到中立 NPZ 转换器 |
| `datasets/pyg_mfcad.py` | MFCAD PyG Dataset、归一化和增强 |
| `uvnet/pyg_encoders.py` | PyG 版本 curve/surface/graph encoder |
| `uvnet/pyg_models.py` | segmentation model、Lightning module、旧权重名映射 |
| `segmentation_pyg.py` | 训练、测试、checkpoint、日志和详细推理 CLI |

## 13. 推荐复现顺序

1. 检查 `nvidia-smi`，但不要先修改系统 CUDA。
2. 用 uv 在仓库内创建 `.venv-gpu`。
3. 安装 PyTorch CUDA wheel、requirements 和匹配的 scatter wheel。
4. 运行 `gpu_smoke_test.py`。
5. 用旧 `.venv` 将 DGL 数据转换一次。
6. 核对 `neutral/manifest.json` 的 split 数量。
7. 运行 10/2 batch smoke training。
8. 运行完整 1 epoch 并测试 checkpoint。
9. 运行目标 epoch 数的正式 FP32 基线。
10. 测试最佳 checkpoint，再尝试 BF16、数据增强或多 seed 实验。

这样可以把环境、数据、模型、训练和精度问题逐层隔离，出现差异时容易定位。
