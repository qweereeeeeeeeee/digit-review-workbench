# 第二轮选型：数字、火箭和多主体照片

日期：2026-11-12。状态：源码调研和本地失败基线已完成；新模型尚未安装、尚未推理，不能声称准确度提升。

## 必须满足的实际任务

用户明确要求同时覆盖手写数字、火箭以及一图多物体。不得用 COCO 类别表不含火箭来缩小验收范围，也不得将纸上数字当成生活物体分类任务。数字需要文本定位与识别；物体需要定位与类别/区域描述。全图只输出一个词不是合格方案。

本机最新上传记录印证了这一点：同一张手写照片的两次检测，一次空结果、一次误报 umbrella。火箭照片检测为空。这些不是通过调整界面能解决的缺陷。当前 YOLO11s 的 COCO 权重不能承担开放类别识别或 OCR。

## 本轮实际阅读的项目与代码

| 来源 | 已核对内容 | 对本项目的意义 | 尚未验证 |
|---|---|---|---|
| [Microsoft Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) | 官方模型说明、任务提示、输出格式、官方文件清单及权重 SHA | 同一模型提供区域描述、描述词定位、OCR 区域等任务；选为本地对比候选 | 实际照片效果、Intel CPU 时延、依赖兼容性 |
| [Autodistill Florence-2](https://github.com/autodistill/autodistill-florence-2/blob/main/autodistill_florence_2/model.py) | run_example、Florence2.predict、模型加载 | 借鉴 processor → generate → post_process_generation 的真实路径 | 不能原样照抄：此封装硬编码 CUDA，且给每个检测赋 confidence=1.0；该值不是模型可信概率 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/v2.9.1/tools/infer/predict_system.py) | TextSystem：文字检测、排序、透视裁剪、方向处理、逐区域识别、分数过滤 | 用成熟 OCR 处理数字照片，而非压成 8×8 的全图数字分类 | 当前机器安装和手写照片性能；不能将上游印刷体指标当成手写成绩 |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/easyocr.py) | detect、recognize、readtext；区域合并、对比度处理、返回坐标和文字 | 作为现有 PyTorch 环境的 OCR 对比候选 | 手写数字精度、模型下载和本机运行 |
| [Multi-Vision Toolkit](https://github.com/Limbicnation/multi-vision-toolkit) | README 的模型切换、结果存储和回退机制 | 多任务分开呈现、保留模型来源 | README 声明部分加载失败会退回 Dummy model；本项目不能采用这种回退 |
| [Hugging Face skills](https://github.com/huggingface/skills/blob/main/skills/huggingface-best/SKILL.md) | 模型能力、硬件、公开评测来源检查流程 | 用于选型的工作规范，不是识别模型 | 安装被自动审批额度错误拦截；未安装成功，未完成其评测目录查询流程 |

以上仅意味着读过相关说明或代码，不意味着这些库都已经融合、安装、实测或全部审计。源码文件的 blob SHA 另列于 GENERAL_VISION_SOURCES.json。

## 待验证的组合流程

1. 保存原图并纠正 EXIF 方向；在同一图上分别运行区域识别和 OCR，不靠用户先挑出一个物体。
2. Florence-2 分别执行 DENSE_REGION_CAPTION、OCR_WITH_REGION。必要时比较 CAPTION 与 CAPTION_TO_PHRASE_GROUNDING；不把“rocket”硬编码成每张图的答案或以文件名决定输出。
3. OCR 用 PaddleOCR 或 EasyOCR 的实际结果比较；数字专用字符表只能作为用户明确选择的模式，不能默认为所有文字强制转数字。
4. YOLO 作为原始常见物体基线保留，对比新方案的漏检和误报，不将不同模型分数平均成所谓更高置信度。
5. 保留模型原始文本、区域坐标、模型版本、完整耗时和失败原因；无法提供经过校准的分数时显示“未提供”，不能显示虚构的百分比。
6. 加载失败必须真实报错，不用默认标签、随机网络或假模型补结果。照片不上传外部推理服务。

这只是待实测方案，不能说采用多个模型就一定更准确。

## 先冻结验证标准，再运行候选

- 对比资料：本机已有全部手写上传原图（含多位数字和阴影）、公开火箭原图、现有全部公开示例，成功和失败都保留。不能只挑此前识别正确的图。
- 数字：记录每个文字区域、预测全文、真实人工标签；按整串完全匹配率及字符错误率统计。没有核对的标签不能当真值；单个 7 与多位数字分开看。
- 物体：逐图标注期望主体及框，统计漏检、误报、类别错误和定位。火箭必须作为具体待验收类别，不能用“其他”算正确。
- 多主体：每个物体独立验收，识别出其中一个不等于整图通过。数字和物体同时出现时，两个分支都需要输出。
- 未经现场采集的拼图或变换图必须注明衍生测试，不称为真实新拍照片。
- 这批已查看样本属于开发回归集；阈值/提示选择完成后仍需要额外未调参的验收照片，不能据回归集宣称泛化成绩。
- 官网榜单、pytest 通过数量与实际识别准确率分别记录。本轮不填任何新准确率。

## 已固定的官方模型信息

候选：microsoft/Florence-2-base-ft，MIT，模型卡标注 0.23B 参数。

版本：f6c1a25888ffc1d945ee8a1a77ac833c7303d46e。

model.safetensors：463221266 字节，官方 LFS SHA-256：58757d657ff44051314c8030b68e04cb1bb618ca9a4885418f111f6fb708185a。

这些信息来自官方 Hub API 清单。本地只有清单，没有下载该权重，也没有执行该模型的远程自定义代码。公开评测是上游报告，不是本项目复现结果。

## 当前阻塞

skill-installer 的安装命令未执行。自动审批反馈当前使用额度已耗尽，无法完成审查，且明确不能绕过审批；不是安全性判定。此后没有改用其他通道安装 skill 或下载/执行候选模型。可继续进行的源码阅读和本地资料整理已完成。

现有服务未替换。恢复审批能力后，应先跑上述候选对比，再决定是否接入页面。不能把这份调研当作已完成的模型升级。
