# CUT 全流程证据审计

三项创新均未得到完整有效性证明。软件 32 项回归测试通过；发现并修复第 10 项问题：top-16 漏掉 18 个符合固定规则的候选。

| 步骤 | 实现 | 结论 | 证据 | 文件 |
|---|---|---|---|---|
| 1.1 | 自动全图分窗 | 软件实测通过 | 7 对、448 窗；每张图覆盖一次；全部 H&E/IHC 框相同。这只验证坐标实现，不证明实际配准精度。 | input_identity.json |
| 1.2 | 自动同坐标质控 | 规则执行通过，组织真值未验证 | 新 6 对 8→333/384 窗；仍是近白可观测性规则，不证明病理组织判断正确。 | input_identity.json；旧/新 manifest |
| 1.3 | 实染参考量测 | 数值复算通过 | 独立 NumPy 与 manifest 的 q 最大差 0；q 只是区域 DAB 代理。 | independent_reference_measurements.json |
| 1.4 | 形态表示训练 | 软件实测通过 | 仅用训练角色 270 窗拟合 9 维 H&E 特征及聚类；固定距离 1.5、差异 0.2。 | morphology_train_only.json |
| 1.5 | 形态近似、表达相反挑战 | 研究证据不足 | 全量含训练图得到 106 对代理候选；开发/留出 127 窗得到 0 对；无经核实的蛋白相反标签。 | challenge_audit.json；完整候选清单 |
| 1.6 | 标记物特异可预测边界 | 未证明 | 标记物身份、病人关系和真实表达标签未核实，当前不能宣称得到 HER2 等标记物的可信边界。 | complete_evidence.json |
| 2.1 | 配对监督与目标自动派生 | 软件实测通过 | 随机 B 与 paired_B 分离；配对同步；五箱标签为固定 DAB 统计。 | tests.txt；逐窗独立记录 |
| 2.2 | 区域表达分布头 | 当前实测不支持有效性 | 留出 log-loss=0.9762，劣于训练直方图先验 0.1657；全部 63 窗属于第 0 箱。 | v2_compat_independent_holdout_records.json |
| 2.3 | 细胞阳性概率／像素头 | 细胞层面未实现；像素代理表现弱 | 实现的是像素 DAB 辅助头，无细胞检测与细胞真值。像素 BCE=0.7734；训练阳性率常数基线 BCE=0.1350。 | v2_compat_pixel_evidence.npz；complete_evidence.json |
| 2.4 | 六残差块与 FiLM 条件控制 | 结构通过，控制量不足 | 高低条件跨度 0.8，生成 DAB 面积率均差仅 0.001732；不能把单调响应当作定量控制成功。 | v2_compat_condition_sweep.json |
| 2.5 | 多样本后验生成 | 采样可运行，生物学后验未证明 | 真实 G 生成样本与采样方差已保存；单参考、分箱代表值和未校准概率均不足以识别真实生物学后验。 | selected_sampling_evidence.json；sample PNG |
| 2.6 | 条件／空间一致性 | 当前实测失败 | 默认 MAE=0.003761 但 IoU=0.000738；实验组合 MAE=0.303735，保持默认关闭。 | 独立逐窗 CSV；condition_sweep.json |
| 3.1 | 来源分组与防泄漏 | 拒绝逻辑通过，数据不独立 | 仅 2 个未核实来源家族；2M 家族跨调试角色；开发与留出源重叠，不能改名伪造独立组。 | certification_audit.json；tests.txt |
| 3.2 | 病理错误事件定义 | 面积率事件可算，病理错误目标不足 | 默认 63 窗面积率事件全 0，但 IoU<0.1 的定位诊断全失败。诊断不是临床金标准，也没有事后替换训练目标。 | 独立逐窗 CSV |
| 3.3 | h 预测真实错误 | 未证明 | 默认 h 开发事件 64 个 0、留出 63 个 0；实验 h 全 1；AUROC 无定义，低 Brier 没有区分价值。 | complete_evidence.json |
| 3.4 | 独立患者概率校准 A | 数据缺失 | 没有可用的独立、身份核实的 A 组。未拟合或发布真实概率校准器。 | certification_audit.json |
| 3.5 | 预设风险认证 B | 数学公式可核验，实证保证未成立 | 固定 7 阈值、alpha=0.05、预算=0.10；即使零错误也至少需要 47 个被接受的独立 B 组；当前 0 个核实患者组。 | certification_audit.json；tests.txt |
| 3.6 | H&E-only 自动拒绝 | 实际入口实测通过 | 监测了文件图像读取；只读输入 H&E。64 窗可信放行 0；拒绝不当作阴性。门控权重摘要每文件仅计算一次。 | he_only_access_audit.json；he_only_inference/prediction.json |

本次 G 未重新训练；重新加载实际权重独立推理、独立 NumPy 计量。全部同坐标；无人工 ROI；来源身份未核实，不能声称患者级泛化。详见自包含 HTML 和 complete_evidence.json。
