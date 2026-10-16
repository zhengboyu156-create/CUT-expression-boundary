# 在上传的 CUT 代码库上实现区域表达边界、分布生成与自动拒绝

本扩展沿用原始 CUT 的骨干、训练目标和六残差块生成器，新增 `--model expression_cut`。原始 `cut_model.py` 只增加显式命令行参数入口，`train.py` 接入可选进度钩子。原始 CUT 的随机 B 图仍用于对抗目标；`paired_B` 是单独读取的同坐标真实 IHC，供新增监督和离线误差计算。程序不做全局位移、局部对应搜索或人工选区。

**当前支持的表达目标是区域 DAB 染色统计代理。** 原始图像本身提供图像级监督；分箱标签和像素 DAB 伪标签由固定测量器自动派生。像素概率辅助头不是细胞蛋白阳性概率，也不输出 HER2 临床分级。没有细胞/临床真值时，不作相应准确性声明。

## 最新：全流程逐条证据审计

完整报告：`artifacts/complete_evidence/complete_evidence_report.html`。三项创新共 18 步逐项对应真实图片、原始记录和测试；不能据现有数据证明三项研究创新全部有效。软件测试现为 32 项全部通过；本次没有重新训练 G，而是重新加载既有两组 160 步权重，独立推理和 NumPy float64 计量，指标与旧记录最大差小于 1e-6。

新增第 10 项修复：形态检索的 top-16 截断会漏掉合格冲突候选。改为分批固定半径查询，临时邻居数组不覆盖所有窗口，并用 j>i 去重。全量含训练图的候选从 88 增至 106，独立穷举恰好 106；开发/留出 127 窗仍为 0 个冲突候选。该操作只比较不同 H&E 窗的形态；每个窗的 IHC 坐标保持原位，绝不进行区域外对应搜索。

新负对照：默认分布头留出 log-loss 0.9762，劣于训练频率先验 0.1657；像素 DAB 辅助头 BCE 0.7734，劣于仅用训练阳性率的常数基线 0.1350。细胞检测、细胞蛋白阳性真值与膜定位尚未实现/提供，不能把像素辅助头称作细胞阳性模型。默认的面积率错误事件在留出全为 0，但额外的 IoU<0.1 空间诊断全部失败；该空间诊断没有被事后替换成 h 的训练目标。

真正的 HE-only 入口已监测文件读取，只读取输入 H&E；输出 PNG 的解码核验另行记录；门控 G/h 摘要各计算 1 次；缺认证时 0/64 窗放行。数学软件校验：固定 7 阈值、alpha=0.05、预算 0.10，在零错误的有利条件下，仍至少需要 47 个被接受的独立 B 组；不是把 47 个相关图像窗当独立患者。当前没有核实患者组或独立 A/B 证书。身份、标记物和配准精度也没有因程序保持坐标一致而自动得到验证。

```bash
python -m unittest tests.test_expression tests.test_multipair_debug tests.test_proof_audit -v
python -m tools.audit_complete_evidence
python -m tools.report_complete_evidence
```

逐窗独立指标为 `artifacts/complete_evidence/independent_holdout_metrics.csv`；所有证据含 SHA256 索引和原始测试输出。合成单元夹具只验证软件，不作为生物学证据。

## 上轮：新增六对图片的调试

最新报告为 `artifacts/multipair_debug/multipair_debug_report.html`。使用新增 6 对图片和旧 1 对锚点，按图像对固定分成 5 对训练、1 对开发、1 对留出，来源身份均未核实，不能解释为患者独立验证。两臂生成器各训练 160 步、ngf=8、6 残差块。24 项回归测试覆盖已确认的软件问题。

已修复：淡染 IHC 被整图排除、文件名改变采样结果、宿主 argv 污染、统计窗配置失配、评估积累全部图像张量、跨实例全局缓存、非法风险分数和协议失配未拒绝、重复读取权重计算哈希。新六对的有效窗从 8/384 增加为 333/384；同坐标映射始终不变。

实际负结果：区域 MAE 很低仍可能定位错误，留出图 v1 的 MAE=0.003761、DAB 像素 IoU=0.000738。箱内代表值与预测条件混合的组合实验 MAE=0.303735，高于固定零基线 0.005177。组合策略已默认关闭，参数 `--learn_bin_means true --lambda_moment 2 --teacher_probability 0.5` 仅保留为明确的实验开关；单项效果尚未消融。默认参数为 false / 0 / 1，回到兼容训练路径。

兼容分支原样复用 v1 本轮 G 权重，逐窗生成统计差为 0；这不是第三次生成器训练。它重新训练了匹配新协议的 h。开发与留出图的区域错误事件只有一个类别，AUROC 不能计算，不能将低 Brier 或第 0 箱准确率 100% 当作模块有效。没有独立来源认证，可信放行保持 0。

复现多图实验：

```bash
python tools/run_multipair_debug.py --steps 160
python -m tools.report_multipair_debug --base artifacts/multipair_debug_rerun
python -m unittest tests.test_expression tests.test_multipair_debug -v
```

`examples/multipair/pairs.csv` 记录配对与调试角色，不需要 ROI 标注。`v1_source` 是冻结的旧源码，用于复现旧版错误；它不作为默认实现。原始逐窗结果与两次训练权重完整保存。部分训练图的全分辨率预览不重复打包，可用保存权重和 worker 重建；全部统计来自原始浮点输出。

## 历史：最初单图像对实跑

打开 `artifacts/attachment/verification_report.html`。`verification.json` 是原始数值，`step_evidence.json` 对照三个创新点共 18 步，`tests.txt` 是自动测试结果，两个模型目录含实际训练权重与每步损失。

两张 1024×1024 附件自动得到 64 对 128×128 窗。所有窗固定 `he_bbox == ihc_bbox`。该实例只是一组来源，训练/评估同源，不能拆为 64 个独立患者。默认 `ngf=64` 的六残差块模型做真实前后向测试；CPU 微型训练使用 `ngf=8`、相同六残差块、相同训练步数、batch=4。原始骨干初始种子、批次顺序和每步 PatchNCE 随机种子固定。两种模型的额外模块和目标不同，因此总 G 损失不可横向比较。

示例推理采用真实神经网络输出，不用手工着色代替 G。FiLM 响应不达标、样本差异过小等结果照实保留。错误预测器在本例也只有样本内训练；没有独立校准器，所以实际 H&E 推理自动拒绝全部可信表达展示。

## 安装和复现

已测试 Python 3.12、PyTorch 2.7.1 CPU、torchvision 0.22.1。建议独立虚拟环境。先按计算设备安装匹配的 torch/torchvision，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-expression.txt
python -m expression.cli demo --he examples/attachment_pair/he.jpg --ihc examples/attachment_pair/ihc.jpg --out artifacts/attachment --steps 120
python -m unittest tests.test_expression -v
```

上述命令跑的是明确标记的附件过拟合演示。正式 `train` 不接受 demo split；演示代码通过显式 `demo_overfit` 开关执行。微型权重不能创建正式风险认证。

## 完整 MIST 的输入与全自动流程

完整数据未包含在附件中。本扩展不根据 patch 文件名猜测患者身份，不声称已核实官方 4,642/1,000 划分或患者关系。图像对数量、官方来源划分和组别信息以真实清单为准。

`pairs.csv` 列为 `pair_id,group_id,group_kind,split,he,ihc`，见模板。`group_kind` 必须如实填 `patient`、`slide` 或 `unknown`。正式认证拒绝 unknown。路径可相对 CSV 所在目录。整图坐标不需要填，由程序自动分窗。允许 split：`train,val,risk_train,calibrate,certify,test`；test 表示锁定的官方测试来源。

同名配对可以由下列程序自动生成。文件后缀不一致时，用身份正则提取共同的 pair key；这只是按文件身份配对，不搜索图像区域。来源组元数据必须来自数据提供者，不能由本程序创造。

```bash
python tools/build_expression_pairs.py --he-dir /dataset/HE --ihc-dir /dataset/IHC --metadata /dataset/source_metadata.csv --out pairs.csv
python tools/run_expression_experiment.py --pairs pairs.csv --out runs/mist --g-steps 10000 --risk-steps 5000 --ngf 64 --batch-size 1
```

自动流程保留 test 全部来源组；其余组用固定种子划为 train、val、risk_train、calibrate A、certify B。图像内容哈希、pair ID、group ID 都检查跨 split 重叠；发现冲突立即报错。未知来源身份或来源组不足不会伪造分组继续认证。每一步的退出码写 `pipeline_status.json`。

依次执行：固定网格/固定质量规则 → CUT 与扩展 CUT 训练 → 训练组拟合 H&E 形态表示并在 test 自动检索跨组染色冲突候选 → 冻结 G 后在 risk_train 训练 h → 独立 A 保序校准 → 独立 B 风险阈值认证 → 锁定 test 评价。训练步数是预先指定的固定预算，不按测试结果选择 checkpoint；本脚本暂未实现开发组早停，val 可用于事先设计实验，不能用 test 调参。

固定训练预算不是正式有效性的充分条件。发表级实验还应增加仅分布头等消融、多随机种子和来源组重采样区间。本代码不把这些尚未运行的实验写成已完成。

### 原版训练入口 / 现有 CUT 权重

新数据集/模型接入原版工厂，可用原来的 `train.py`，保留原脚本的 GPU 与日志功能。本次另外实跑了原生入口的一个 CPU 优化步，启用 `nce_idt=true`，完成 G/F/D 保存；日志见 `artifacts/native_entry/native_entry.log`。关闭 Visdom 时上述扩展依赖即可；启用 GUI 时再装原 `requirements.txt` 的对应依赖。

```bash
python train.py --model expression_cut --dataset_mode expression_pair --pair_manifest runs/mist/manifest.jsonl --pair_split train --netG resnet_6blocks --ngf 64 --stats_window 128 --preprocess none --gpu_ids 0 --display_id -1 --name mist_expression --nce_idt true
```

`--backbone_checkpoint /path/latest_net_G.pth` 可严格加载原 CUT 生成器权重，再初始化新头与恒等 FiLM；通道数、抗混叠配置等必须与该 checkpoint 一致。六块限制会显式检查。PatchNCE 的特征索引沿用上传库的 `0,4,8,12,16`，均在该六块架构中有效。分布头放在条件化之前；PatchNCE 的特征抽取走未条件化骨干，不读取真实 IHC。

`expression.cli` 无 GUI 训练/风险/认证流程当前使用 CPU；大型正式训练推荐原生 GPU 入口。若要由原生入口的权重进入后续流程，需使用下面的转换程序导出带来源元数据的封装文件，不能直接把不明来源的 checkpoint 当独立校准模型。

```bash
python tools/export_expression_checkpoint.py --checkpoint checkpoints/mist_expression/latest_net_G.pth --manifest runs/mist/manifest.jsonl --ngf 64 --steps 10000 --out runs/mist/expression_cut/generator.pt
```

转换器严格核对网络参数，并将训练来源组与原始权重哈希写入封装。既往训练来源是操作者提供的信息，不能仅凭权重自动证实；来源不明时不能声称独立校准。

## 训练目标和自动参考

固定白点 255；H 向量 `[.65,.70,.29]`，DAB 向量 `[.27,.57,.78]`，归一化并作光密度投影。H&E 组织前景阈值为平均 OD > 0.20；IHC 改用更宽的平均 OD > 0.02 近白可观测性掩膜，避免把淡染当作无组织。连通域清理规则固定；窗级覆盖至少 50%，H 标准差至少 0.08。IHC 可观测性只用于离线缺失参考检查，不移动坐标，且仍是需验证的质量代理。参考和生成统计都使用固定 H&E 掩膜作分母；空掩膜不产生阳/阴标签。

每窗区域目标 `S` 是粗组织内 `DAB > 0.20` 的面积率，分成 5 箱。新增损失：分箱交叉熵、生成染色统计一致性、可配置的像素 DAB 伪标签 BCE；原 GAN 和 PatchNCE 保留。可配置配对 L1 默认权重 0。FiLM 位于瓶颈以及解码器最后一次归一化之后，两处最后一层都零初始化，因此未训练时与原骨干完全一致。第一轮只有瓶颈 FiLM 时调制响应很弱，失败结果与当时代码保存在 `artifacts/round1_bottleneck_film`。

默认条件采样是预测分箱采样后取箱中心。实验开关可学习箱内代表值、增加条件量损失并渐变混入预测条件；组合实测退步，默认禁用，不能把该策略称为已验证修复。它表达模型学到的区域染色量分布；单参考训练、分箱离散化和固定染色分解均有限制，不能因多次采样看起来不同就宣称得到了真实生物学后验。生成位置和膜定位有效性须另有真值及独立实验。

## 错误校准与拒绝

`h` 输入 H&E、生成 RGB、真实 G 采样的方差图。采样种子来自 H&E 像素内容，改文件名不会改变采样；风险 checkpoint 和认证文件同时绑定采样/质量协议哈希。旧 h 需要重训，旧认证不会静默套用新协议。IHC 只离线定义 `abs(S_generated-S_reference) > delta` 事件，部署不接受 IHC 输入。原始 h 分数、采样差异和像素概率都不直接叫作校准置信度。

独立校准 A 拟合保序映射；独立认证 B 在事先固定的候选阈值上计算单侧 Clopper–Pearson 二项风险上界，显著性预算 `alpha / 候选数`，自动选预算内覆盖最多者。模型权重哈希绑定风险模型和认证文件。认证缺失、权重不一致、无阈值通过、质量无效或 demo 权重均不放行。

**二项认证的统计对象明确限定为：每独立来源组固定种子均匀抽 1 个有效窗，组等权。** 切片内窗相关，不能把所有窗当 IID 用普通二项公式。此实现不认证 pooled 全窗/全像素风险；全窗目标需要专门的分组风险方法。若 group_kind 为 slide，不能把结果称作患者级保证。软件单元测试中的合成 100 组夹具只验证边界计算和拒绝逻辑，不是附件/MIST 校准证据。

```bash
python -m expression.cli infer --he /new/HE.png --generator runs/mist/expression_cut/generator.pt --risk runs/mist/risk/risk.pt --certificate runs/mist/calibration/certificate.json --out runs/new_case
```

`generated_RESEARCH_ONLY.png` 始终是研究输出；`selective_display.png` 才应用质量/风险遮罩。拒绝窗保留状态，绝不自动当阴性。输出记录概率、每窗坐标、门控原因与覆盖率。

## 文件导航

- `data/expression_pair_dataset.py`：独立配对参考与同步几何处理。
- `models/expression_cut_model.py`：原 CUT 训练循环上的新损失和优化器。
- `expression/statistics.py`：固定可微/硬测量器与组织代理。
- `expression/networks.py`：分布头、像素伪标签头、FiLM 和 h。
- `expression/challenge.py`：只比较 H&E 的形态近邻，IHC 仍取各自原位。
- `expression/calibration.py`：独立分组、概率映射、有限候选阈值认证和拒绝。
- `expression/runner.py`、`expression/cli.py`：批处理训练、测试、H&E-only 推理。
- `tests/test_expression.py`：真实附件模块测试和明确区分的统计软件夹具。

原代码 LICENSE 保留。训练输出不能用于临床判断。

打包时仅把附件 manifest 的图像路由改为相对路径；原 manifest 与原 SHA256 保存在同目录。训练权重保留原始训练 manifest 哈希，`manifest_path_translation.json` 记录路径转换。图像内容哈希、坐标、分组和标签均未更改。
