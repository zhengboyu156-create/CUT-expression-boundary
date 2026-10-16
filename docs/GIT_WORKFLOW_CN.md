# Git 使用约定

本工程使用独立仓库，所有 Python 命令从仓库根目录运行。

在本机克隆本仓库：

```bash
git clone https://github.com/zhengboyu156-create/CUT-expression-boundary.git
cd CUT-expression-boundary
git switch main
```

继续开发时创建自己的功能分支，按独立改动提交：

```bash
git switch -c feature/your-change
git diff
git add .
git diff --cached --stat
git commit -m "Describe the concrete change"
git push -u origin feature/your-change
```

提交源码、测试、配置模板及短小的证据说明。图片、数据集、权重、训练输出和机器环境由 `.gitignore` 排除。完整证据需用本目录的导入脚本从已交付复现 ZIP 恢复。不要强制添加这些大文件来绕过忽略规则。

修改测量、采样或质量协议后，要同步更新协议定义并重新训练对应 h／认证数据；旧风险证书不能静默套用。实验负结果保留原始记录，方法改动与研究结论分开描述。
