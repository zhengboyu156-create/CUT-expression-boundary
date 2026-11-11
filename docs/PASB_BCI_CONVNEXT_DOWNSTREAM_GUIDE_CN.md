# BCI HER2 下游分类：参考 PASB 的实验说明

本文说明如何在本项目中设计参考 PASB 的下游分类实验。**这是一份操作方案，不是 PASB 作者公开的分类脚本，也不是已完成的实验。** PASB 正文没有明确说明表 2 的分类类别数、每组具体训练流程和全部超参数，因此不能称作对论文表 2 的严格复现。

## 1. 实验要回答什么？

用同一个分类模型分别处理 H&E、不同方法的虚拟 HER2 IHC、真实 IHC，比较它们能否预测已有的 HER2 等级。每组单独训练一个分类器，在同一测试样本上评价；这是为本项目提出的可复现方案，论文没有完整明确所有组的分类器训练细节。

PASB 论文第 4.2.2 节明确说：在 BCI 上用 ImageNet-1k 预训练的 **ConvNeXt-T**，通过不同生成方法的 IHC 图像进行微调，评估 HER2 grading accuracy。论文表 2 给出 H&E、不同方法的虚拟 IHC 和真实 IHC 的结果。论文没有在表 2 报告 H&E＋虚拟 IHC 融合结果。

| 论文表 2 的输入 | 准确率 |
| --- | ---: |
| H&E | 0.640 |
| CUT 虚拟 IHC | 0.747 |
| PSPStain 虚拟 IHC | 0.762 |
| PASB 虚拟 IHC | 0.770 |
| 真实 IHC | 0.804 |

来源：[PASB 论文](https://doi.org/10.1016/j.media.2025.103869)，第 4.2.2 节和表 2；[作者代码仓库](https://github.com/SRMLabHainu/PASB)。截至本文编写时，作者仓库默认分支中未找到该项 ConvNeXt-T 分类微调和评估脚本。类别数、标签处理、划分与具体权重版本未获确认，不能直接将上述数字与本项目的四分类结果比较。

## 2. 本项目的分组

使用与现有 ABMIL 实验相同的固定训练、验证、测试清单；若生成结果不完整，先报告各组实际样本及差异，不得只挑选效果好的样本。

| 组别 | 单独输入 ConvNeXt-T 的图像 | 回答的问题 |
| --- | --- | --- |
| H | H&E | 原图上能达到多少分类性能？ |
| V-CUT | CUT 虚拟 HER2 IHC | CUT 结果保留了多少可用于分类的信息？ |
| V-其他 | 同一 H&E 对应的其他模型虚拟 HER2 IHC | 不同生成方法的下游表现如何？ |
| R | 真实 HER2 IHC | 真实 IHC 输入的参考表现是多少？ |

每组分类任务使用相同标签集合；本项目建议沿用已有的 **0、1+、2+、3+ 四分类**，这是本项目设定，并非已核实的 PASB 表 2 类别设置。H&E＋虚拟 IHC 融合可另行做扩展实验，不计作表 2 原始对照。

标签来源和层级需要记录。尤其 BCI 的切片级 HER2 标签若被赋给图像块，不能视为每个图像块都经病理医师独立分级。真实 IHC 与 HER2 标签来源有关，R 组仅作参考，不能当作与只使用 H&E 具有相同可用信息的公平预测基线。

## 3. 数据准备和划分

1. 保留图像 ID、标签、数据划分以及 H&E、真实 IHC 和各模型生成图像的路径。生成图像必须能追溯到输入 H&E。
2. 四个类别均应在训练、验证、测试中有记录；报告各类数量。检查每个 ID 唯一、文件可读取、标签匹配。训练、验证、测试互不重叠。
3. 如能取得病例或切片映射，应先按**病例**划分并检查各分区无病例重叠，再在分区内关联图像；同一病例不同图像不要分到不同分区。若映射缺失，保留现有图像级划分并明确无法证明病例独立性，也不要称为病例级外部验证。
4. 生成器权重的训练来源、输入及生成过程应记录，确保测试标签或测试真实 IHC 未被用于训练、调参或选择分类模型。若生成方法推理时使用参考真实 IHC，应如实记录，不视作纯 H&E 条件生成。
5. 为训练/验证/测试分别准备对应输入及标签。验证集用于选择模型；测试集只用于最终评估。原始图像、权重和结果应保留在本机，不上传到本公开仓库。

## 4. 预训练权重

用户已下载到本机：

`D:\\xuniranse\\weights\\convnext_tiny-983f1562.pth`

文件大小约 109 MiB，SHA-256：

`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`

来源：[PyTorch 官方权重](https://download.pytorch.org/models/convnext_tiny-983f1562.pth)、[TorchVision 模型说明](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html)。

在 Windows PowerShell 核对文件：

```powershell
Get-FileHash 'D:\xuniranse\weights\convnext_tiny-983f1562.pth' -Algorithm SHA256
```

这是 TorchVision 的 ImageNet-1K 权重，不是 PASB 作者的 HER2 微调权重；论文未明确具体权重发行版本，不能断言二者完全相同。若需重新下载，可从仓库根目录运行：

```powershell
python tools/download_convnext_tiny.py 'D:\xuniranse\weights\convnext_tiny-983f1562.pth'
```

下载脚本不会覆盖已有文件。权重被本仓库的 `.gitignore` 排除，不会随代码上传。

### 本地加载示例

以下仅展示如何装入权重并将输出改为四类；**不是训练或评估脚本**。请在已安装兼容版本 PyTorch、TorchVision 的环境运行：

```python
from pathlib import Path
import torch
from torch import nn
from torchvision.models import convnext_tiny

weights_path = Path(r"D:\xuniranse\weights\convnext_tiny-983f1562.pth")

# 先建立原始 1000 类结构，再载入 ImageNet 权重。
model = convnext_tiny(weights=None)
state = torch.load(weights_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)

# 将输出层替换为本项目的四类；新层需要训练。
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
print("四分类模型已初始化；尚未进行 HER2 训练")
```

TorchVision 默认预处理可作为本项目起点：缩放后中心裁剪至 224×224，RGB 归一化 mean=(0.485, 0.456, 0.406)、std=(0.229, 0.224, 0.225)。应保存实际预处理参数；这不是已知的 PASB 分类预处理设置。病理图像缩放可能改变可见细胞细节，所有组应使用预先确定的一致规则。

## 5. 本项目拟采用的分类训练流程

以下参数和做法是为实现可比较实验提出的方案；论文未披露时不冒充原作者设置。

1. 固定同一图像级或病例级划分；训练、验证、测试用相同 ID 和对应标签，分别为各图像输入组建立数据清单。
2. 各组从相同 ImageNet 预训练权重初始化，各自训练独立的 ConvNeXt-T 四分类器。所有组保持相同预处理、优化器、超参数、验证规则和随机种子；如需例外，提前记录原因。是否冻结骨干、损失函数、训练轮数和学习率由本实验配置明确记录，并非论文已确认参数。
3. 使用训练集训练、验证集选最优检查点；不得根据测试集结果选择模型。先生成训练集及验证集对应虚拟图像，再按冻结的方案处理测试集。
4. 用冻结的各组最佳检查点，逐一预测同一测试集，保存 ID、真实标签、预测类别、四类概率和模型/生成器版本。
5. 至少报告 Accuracy、Macro-F1、QWK，以及四类的 precision、recall、F1 和混淆矩阵；AUC 如计算，注明四分类 one-vs-rest / one-vs-one、宏平均方式和概率来源。可沿用 3 个随机种子，报告均值 ± 种子标准差，不将其称为病例置信区间。

实际 ConvNeXt 训练脚本和本地运行快照见第 7 节。第 7 节的指标是运行中间结果，不是最终实验汇总。

## 6. 结果解读与记录

- 对照各组训练及测试 ID、标签、生成图映射、随机种子，确保除输入图像外条件可比。
- 记录每个生成方法的训练数据、版本和推理时是否使用真实 IHC。虚拟 IHC 结果较低或融合不提升时如实报告，不应预设必须高于 H&E。
- 同一批样本比较不同输入时可使用配对分析；置信区间的重采样单位应与研究独立单位一致。没有可靠病例映射时，只能报告图像级统计，不得称为病例级。
- 已有 ABMIL 四分类结果可作为本项目内另一个模型的对照；ConvNeXt-T 已按第 7 节配置启动，最终结果尚未完成。不同结构和标签来源会影响解释。

论文声称的范围以原文为准，本方案涉及的病例独立性、组别设计和四分类为项目研究设计，不应写成作者已报告的事实。

## 7. 本地 ConvNeXt 实际运行记录（2026-11-11，进行中）

本节记录本地实际启动的实验配置和当时进度。它补充第 5 节的方案状态；实验完成前，下表只能作为中间结果，不能当作最终方法排名。

### 分组与数据

- 分类器：TorchVision `convnext_tiny`，加载 ImageNet-1K 官方权重 `convnext_tiny-983f1562.pth`，严格加载后将 `classifier[2]` 替换为 4 类线性层；**全网络微调**，不是固定特征提取。
- 分组：真实 H&E、MAPS-GAN V3 虚拟 IHC、CycleGAN 虚拟 IHC、CUT 虚拟 IHC、M2PL-GAN 虚拟 IHC、真实 HER2 IHC。每组单独训练，不做 H&E/IHC 特征融合。
- 标签：0、1、2、3 分别对应项目清单的 HER2 0、1+、2+、3+ 标签。
- 固定图像级划分：fit 3,311 张、validation 585 张、test 977 张；六组使用相同 ID 和标签。MAPS 图像由现有 V3 检查点生成；其他虚拟 IHC 由 `baseline_feature_manifest.csv` 中对应方法的图像路径提供。
- 当前运行包含的是**既有 MAPS-GAN V3**输出；并行训练中的局部 DAB 权重消融尚未纳入本轮分类。

### 输入预处理与训练

| 项目 | 实际配置 |
| --- | --- |
| 输入尺寸 | RGB 方形图先 Resize 到 256，再取 224×224 |
| 训练增强 | RandomCrop(224)、随机水平翻转、随机垂直翻转；不加颜色扰动 |
| 验证/测试 | Resize(256) 后 CenterCrop(224) |
| 归一化 | ImageNet mean `(0.485, 0.456, 0.406)`、std `(0.229, 0.224, 0.225)` |
| 初始化 | 官方 ImageNet-1K ConvNeXt-Tiny；SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d` |
| 优化器 | AdamW，learning rate `1e-4`，weight decay `1e-4` |
| 损失 | 训练集类别频数的平方根倒数权重，归一化到均值 1 后用于加权交叉熵 |
| Batch / 轮数 | batch size 32；最多 50 epoch；validation Macro-F1 连续 10 轮不提升则早停 |
| 检查点选择 | 只按 validation Macro-F1 选择，平分时保留较早轮次 |
| 随机种子 | 1234、2345、3456；每个输入组独立训练 3 次 |
| 数值精度 | CUDA 前向使用 FP16 autocast；概率 softmax 和 AUC 计算用 FP32 |

测试指标包括 Accuracy、四分类 macro one-vs-rest AUC、Macro-F1、QWK、逐类 Precision/Recall/F1 及混淆矩阵。表中种子汇总使用均值 ± 样本标准差；这不是病例级置信区间。

### 启动命令

```powershell
python tools/train_bci_convnext.py `
  --split-csv 'D:\xuniranse\BCI\BCI_HER2_ABMIL_20261110\split.csv' `
  --baseline-manifest 'D:\xuniranse\BCI\BCI_HER2_ABMIL_20261110\comparison_baselines_20261111\baseline_feature_manifest.csv' `
  --weights 'D:\xuniranse\weights\convnext_tiny-983f1562.pth' `
  --output-dir 'D:\xuniranse\BCI\BCI_HER2_CONVNEXT_20261111_retry3\' `
  --methods he maps_v3 cyclegan cut m2plgan real_ihc `
  --seeds 1234 2345 3456
```

本次运行输出目录已存在，直接复跑会被覆盖保护拒绝；复现时应将 `--output-dir` 改成一个新的空目录。脚本默认不覆盖已有实验输出。图像、数据清单、权重和训练结果保存在本地；不应将患者图像、权重或完整预测文件提交到公开仓库。

### 运行快照

截至 **2026-11-11 21:12（UTC+8）**，总计 18 个“输入组×随机种子”运行中有 8 个已完成，CycleGAN seed 3456 正在训练。H&E 和 MAPS-GAN V3 三个种子均已完成：

| 输入组 | Accuracy | Macro OvR AUC | Macro-F1 | QWK |
| --- | ---: | ---: | ---: | ---: |
| 真实 H&E | 0.9335 ± 0.0031 | 0.9934 ± 0.0005 | 0.9296 ± 0.0076 | 0.8969 ± 0.0086 |
| MAPS-GAN V3 虚拟 IHC | 0.7834 ± 0.0101 | 0.9401 ± 0.0031 | 0.7336 ± 0.0178 | 0.6250 ± 0.0186 |

本地完整输出目录为 `D:\xuniranse\BCI\BCI_HER2_CONVNEXT_20261111_retry3`。该划分是图像级，现有文件没有可核实的病例映射，不能据此声称病例级独立验证。MAPS V3 生成器使用过整个 3,896 张训练分区，包含下游分类验证子集；测试 977 张未用于生成器训练。此点可能影响 MAPS 组的验证选模，解释中需保留。

---
