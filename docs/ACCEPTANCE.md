# 验收对应表

| 要求 | 对应实现和证据 |
|---|---|
| Python、NumPy、Pandas、Matplotlib | `src/` 的数据、训练、指标模块；`scripts/review_experiment.py` 的队列分析；CSV 与图表产物 |
| 手写 ResNet、训练、保存、推理 | `src/model.py`、`src/train.py`；`artifacts/run-001/model.pt` 与每轮记录；`workbench/inference.py` |
| 新图片预测 | 图片导入接口、可选预处理、原图与输入对照；仅支持单数字 |
| 参考与拓展 | `docs/research/GITHUB_NOTES.md`；复核状态、重试、冲突、导出与固定预算队列实验 |
| 仓库及依赖 | https://github.com/qweereeeeeeeeee/digit-review-workbench；`requirements.txt`、`requirements-lock.txt`、`requirements-demo.txt` |
| 开发历史 | 本地历史保留；远程从调研起分阶段提交。第一版仅有本地历史，不冒充此前已远程持续提交 |
| 项目说明 | `README.md`、`docs/PROJECT_REPORT.md`、原训练报告 `docs/REPORT.md` |
| 个人感悟 | `docs/DEFENSE.md` 提供理解和动手检查问题；本人仍需基于实际经历补写 |
| 运行视频 | `artifacts/review/demo/运行演示.mp4`，34.8 秒实际 Chrome 画面，自动化操作、无音轨 |
| AI 辅助且理解代码 | AI 辅助明确披露；提交者仍需自行读懂、修改和排查，不能由自动测试替代 |

测试日志和浏览器检查位于 `artifacts/review/`。截图来自项目运行，数字图片来自公开数据或用户上传；扰动场景明确标注为程序变换。
