# 第一阶段考核验收对应表

核对日期：2026-11-13。依据随包考核 PDF；下列路径相对于项目源码目录。

| 要求 | 交付证据 | 说明 |
|---|---|---|
| Python、环境与包管理 | src/、objects/、workbench/；requirements*.txt | 独立 venv，记录依赖；不打包机器专属虚拟环境 |
| NumPy、Pandas、Matplotlib | src/data.py、src/metrics.py、src/train.py、scripts/review_experiment.py | 实际数组处理、数据表、统计及图表，产物在 artifacts/ |
| 手动搭建 ResNet | src/model.py | 手写残差块、投影 shortcut、网络组装，不以预训练模型代替 |
| 训练、保存和新图预测 | src/train.py、artifacts/run-001/、workbench/inference.py | 原权重、40 轮记录、验证选择和逐样本预测保留；真实照片能力有限 |
| 迁移学习与模型复用 | objects/general.py、模型来源清单 | 复用官方预训练模型，未宣称自行微调这些大模型 |
| 仓库与持续历史 | https://github.com/qweereeeeeeeeee/digit-review-workbench；交付包 05_开发历史/ | 实际本地和远程阶段提交；首版只有本地历史的事实不改写 |
| 文档与个人感悟 | docs/DELIVERY_REPORT.md；交付包 01_项目说明.html | 感悟基于实际测试反馈整理，不虚构独立编码经历 |
| 运行视频 | 交付包 02_运行视频/用户现场测试.mov | 采用用户提供的原始视频；早期自动化视频在源码中作为历史保留 |
| 参考与创新 | docs/research/、docs/DELIVERY_REPORT.md | 成熟模型与复核、来源校验、版本冲突等工程组合，不冒充原创算法 |
| AI 辅助与代码理解 | 明确披露 AI 辅助；docs/DEFENSE.md；验证脚本 | 本人理解仍须通过讲解和修改展示，不能由打包保证 |
| 结果、仓库链接、说明一并打包 | 交付包首页、源码、视频、模型、历史 | 本次整理交付包，由提交者交给考核指定联系人 |

当前识别结果以 docs/GENERAL_VISION_CHECK.md 为准。旧版文档和视频保留历史语境，不将早期结果当成新版成绩；自动化测试数量不是识别准确率。
