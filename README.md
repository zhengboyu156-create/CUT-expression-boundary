# CUT 表达可预测边界：实现与证据审计

本仓库保存 CUT 表达可预测边界的实现和证据审计。入口、Python 包和依赖均以仓库根目录为根。

新增模块包括同坐标全图自动分窗、H&E 形态冲突候选检索、区域 DAB 分布头、FiLM 条件生成、像素 DAB 辅助头、错误预测器以及独立来源风险认证和拒绝。生成器使用 `resnet_6blocks`。

**目前三项研究创新都没有完整有效性证明。** 现有数据是 7 对图片、两个未核实来源家族，只有同来源图像对调试；两组微型训练各 160 步、ngf=8。本目录明确保留失败实验，不代表完整 MIST 训练或临床验证。像素辅助头不是细胞蛋白阳性概率模型。

## 安装

克隆后进入独立仓库：

```bash
cd CUT-expression-boundary
python -m venv .venv
```

激活环境后，先安装与机器匹配的 PyTorch，再安装本工程依赖。已验证的 CPU 环境为 Python 3.12、torch 2.7.1、torchvision 0.22.1：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-expression.txt
```

## 不依赖附件的检查

```bash
python -m compileall -q data models options util expression tests tools train.py test.py
python -m unittest tests.test_expression.StatisticalUnitFixtures tests.test_proof_audit.CompleteEvidenceTests.test_radius_search_does_not_drop_dense_cluster_conflicts tests.test_proof_audit.CompleteEvidenceTests.test_zero_error_binomial_upper_matches_closed_form tests.test_proof_audit.CompleteEvidenceTests.test_no_or_failing_certificate_never_releases tests.test_proof_audit.CompleteEvidenceTests.test_invalid_certificate_release_flag_does_not_override_risk_bound -v
```

这 5 项合成单元检查验证软件行为，不能当作生物学证据。

## 恢复附件与完整复核

为保持源码仓库精简，图片、权重、生成图和训练输出不提交到 Git。完整 32 项测试及原始证据审计需要之前交付的 `contrastive-unpaired-translation-master.zip`。将 ZIP 保存在本机任意位置，然后运行：

```bash
python tools/import_repro_assets.py --archive /absolute/path/contrastive-unpaired-translation-master.zip
python -m unittest tests.test_expression tests.test_multipair_debug tests.test_proof_audit -v
python -m tools.audit_complete_evidence
python -m tools.report_complete_evidence
```

Windows 同样可使用本机 ZIP 路径。导入器核对 `repro_assets.json` 中的 SHA256，只恢复 `examples/` 和 `artifacts/` 下的复现资料，不替换 Git 源码。新输出仍被 `.gitignore` 排除。

完整可视化报告在恢复/生成后的 `artifacts/complete_evidence/complete_evidence_report.html`。原始两组训练复现入口为 `python tools/run_multipair_debug.py --steps 160`。

## 已确认的结果

- 10 项软件问题已修复，完整 32 项测试通过；逐项记录见证据文档。
- 固定 top-16 检索会漏掉 18 对候选；分批完整半径检索得到 106 对，与穷举一致。开发/留出图片仍为 0 个满足门槛的候选。
- 默认分布头留出 log-loss 为 0.9762，劣于训练频率先验 0.1657；像素 DAB BCE 为 0.7734，劣于训练阳性率常数基线 0.1350。
- 默认区域 DAB MAE 为 0.003761，但像素 IoU 仅 0.000738。条件训练组合退步，默认关闭。
- h 的开发/留出事件只有一个类别；无独立 A/B 认证；真实 H&E-only 入口可信放行 0/64 窗。

相同坐标只证明程序映射一致，不证明实际切片配准精度。所有 DAB 数值都是图像统计代理，不能解释为 HER2 临床分级或经核实的蛋白表达真值。

## 文件导航

- [完整方法与运行参数](README_EXPRESSION_CN.md)
- [18 个步骤的证据账本](docs/evidence/complete_evidence_report.md)
- [逐步骤结构化证据](docs/evidence/step_evidence_complete.json)
- [32 项测试的历史原始输出](docs/evidence/tests.txt)
- [Git 使用约定](docs/GIT_WORKFLOW_CN.md)
- [本次独立仓库上传核验](docs/evidence/standalone_upload_status.md)
- [原 CUT 来源说明](README_UPSTREAM.md) 与 [原许可证](LICENSE)

本目录的算法、数据读取和测试来自已交付代码快照；新增的 Git 文档与资料导入工具不改变训练算法。
