"""Generate documentation from actual experiment artifacts, never preset scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'artifacts/run-001'
r=json.loads((run/'metrics.json').read_text())
q=json.loads((run/'data_quality.json').read_text())
pct=lambda n:f'{100*n:.2f}%'
t=r['test_calibrated']; raw=r['test_raw']; b=r['baseline_logistic']
lines=['# 墨鉴：让数字识别的结果可以被检验','',
'## 项目问题与定位','',
'这个项目解决单张小尺寸手写数字图像的十分类问题。在完成手写 ResNet 的基础上，进一步回答：概率可信吗、什么情况下交给人复核、图像受损后表现如何、模型依赖哪些像素。项目定位为可复现的学习实验与本地演示服务，不是生产 OCR。','',
'## 数据与方法','',
f"真实数据共 {q['samples']} 张；缺失单元格 {q['missing']}，完全重复图像 {q['duplicate_images']}。图像由 0–16 归一化到 0–1。数据出处和授权见 REFERENCES.md。",'',
f"固定种子 42；实际划分：训练 {r['splits']['train']}，验证 {r['splits']['validation']}，校准 {r['splits']['calibration']}，测试 {r['splits']['test']}。逐行像素、标签与划分记录在 dataset.csv；全部索引和数据指纹保存在 manifest.json。",'',
'手写网络由 3×3 stem、三个阶段的六个残差块、全局平均池化和线性分类头组成。阶段宽度 16/32/64；后两阶段的首块通过 stride=2 降采样，并以 1×1 投影匹配 shortcut。没有使用预训练模型。', '',
f"共有 {r['parameter_count']:,} 个参数。训练 40 轮，AdamW 初始学习率 0.003，权重衰减 0.0001，余弦调度，batch size 64。按验证交叉熵选择第 {r['best_epoch']} 轮。CPU 运行用时 {r['duration_seconds']:.1f} 秒（单次实测，不是性能基准）。",'',
f"温度在独立校准集上从预定义的 [0.25, 4] 对数网格选择，得到 T={r['temperature']:.6f}。同一校准集最大概率的 10% 分位数形成阈值 {r['threshold']:.6f}，目标是约 90% 校准集覆盖率。它不保证测试覆盖率或错误率。",'',
'## 实际实验结果','',
'| 方法 / 场景 | 准确率 | Macro F1 | 覆盖率 | 接受样本准确率 |','|---|---:|---:|---:|---:|']
for name,m in [('逻辑回归基线（不拒识）',b),('ResNet + 温度校准',t)]+list(r['robustness'].items()):
    accepted='无接受样本' if m['accepted_accuracy'] is None else pct(m['accepted_accuracy'])
    lines.append(f"| {name} | {pct(m['accuracy'])} | {m['macro_f1']:.4f} | {pct(m['coverage'])} | {accepted} |")
lines += ['',f"全体测试准确率的 95% Wilson 区间为 [{pct(t['accuracy_95pct_wilson'][0])}, {pct(t['accuracy_95pct_wilson'][1])}]。这是二项比例近似区间；单个随机种子的结果不能说明跨划分稳定性。",'',
f"测试 NLL：校准前 {raw['nll']:.5f}，校准后 {t['nll']:.5f}；10 箱 ECE：校准前 {raw['ece']:.5f}，校准后 {t['ece']:.5f}。温度不改变分类 argmax，因此不会改变总体准确率。校准效果必须按实测比较，不能假定一定变好。",'',
'扰动在同一保留测试集上生成：固定随机种子 43 的 σ=0.25 高斯噪声并裁剪到 [0,1]；亮度乘 0.5；中心 2×2 像素置零。这是指定扰动测试，不是自然分布外样本评估。', '',
'![训练过程](../artifacts/run-001/training.png)','',
'![可靠性与压力测试](../artifacts/run-001/reliability.png)','',
'![混淆矩阵](../artifacts/run-001/confusion.png)','',
'![数据分布](../artifacts/run-001/distribution.png)','',
'所有测试预测见 predictions.csv；错误预测单独保存在 errors.csv。不会删除失败结果，也不会只以接受样本的准确率代替总体准确率。','',
'## 项目级创新','',
'1. 可靠性闭环：独立校准、概率评价、拒识与覆盖率一起呈现。','2. 可见的模型敏感度：批量执行 64 次单像素置零，显示原预测类别概率的变化。它不是因果解释，也可能受扰动产生的分布变化影响。','3. 可追溯推理：每次返回请求 ID、真实服务端耗时与模型 SHA-256；启动时检查模型和评估报告是否匹配。','4. 可检查的交付：真实训练日志、逐样本预测、坏例、数据指纹、依赖锁与阶段性提交均保留。','',
'以上是已有方法的工程组合创新，不宣称原创算法或超过未实际对比的项目。','',
'## 后端设计与验证','',
'FastAPI 在启动阶段只加载一次 CPU 模型。预测输入必须是 8×8、有限、介于 0 和 1 的数值矩阵，拒绝字符串、布尔值、错误形状、空白图像和超过 16 KiB 的请求体。训练产物通过允许列表展示。服务默认绑定 127.0.0.1，仅用于本地考核演示。','',
'测试覆盖 shortcut 形状和梯度、完整模型反向传播、数据划分与重复检查、温度缩放性质、零接受样本指标、输入校验、真实 API 推理、体积限制及图表访问控制。具体通过情况以 artifacts/tests.txt 和 artifacts/demo/browser_checks.json 为准。','',
'## 局限与诚实说明','',
'- 仅 1797 张小图，单次样本级拆分，不代表跨书写者或真实文档上的效果。','- 浏览器新画或上传的图像有分布差异；没有真实标签就不计算其准确率。','- 概率阈值不是分布外检测器，错误预测仍可能有很高置信度。','- 未训练迁移学习模型、未做多种子或消融实验；不能声称这些实验已完成。','- CPU 耗时包含当前环境因素；不是并发吞吐量或生产性能承诺。','- 本项目由 AI 辅助完成，个人感悟和代码理解须由提交者本人补充与验证。','',
'## 真实开发过程中的问题','',
'本机系统 Git 因开发工具缺失无法运行，改用 Codex 自带 Git。官方 Python 包源发生连接中断，改用 HTTPS 镜像安装固定版本；NumPy 固定为 1.26.4，以兼容此 Intel Mac 可安装的 PyTorch 2.2.2。这些是本次实际遇到的工程问题，不是提交者个人学习经历。','',
f"本次训练开始 UTC：{r['started_utc']}；结束 UTC：{r['finished_utc']}。模型 SHA-256：`{r['model_sha256']}`。",'',
'## 个人感悟（需本人完成）','',
'请按 DEFENSE.md 提示，写下自己实际理解、修改、排错和验证的经历。此处保留待完成状态，不代写虚假个人成长记录。']
(ROOT/'docs/REPORT.md').write_text('\n'.join(lines)+'\n')
print(ROOT/'docs/REPORT.md')
