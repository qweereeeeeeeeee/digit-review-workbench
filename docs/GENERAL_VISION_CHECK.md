# 数字、火箭和多物体：实际验证记录

2026-11-12，Intel macOS，16 GiB 内存，Python 3.12，CPU 推理。下面是实际运行输出，不是上游榜单成绩。

## 做了什么

原数字模型回答的是“一个小块更像 0–9 中的哪一个”。手机照片却包含纸张、阴影、背景和多个字符，任务不匹配。后来换用 COCO YOLO 仍不能读文字，80 类中也没有火箭。只提高阈值或修改显示名称解决不了这个问题。

这次直接复用成熟项目的推理链路：Florence-2 的 processor → generate → post_process_generation；PaddleOCR 的文字检测 → 方向处理 → 分块识别，由 RapidOCR/ONNX Runtime 执行。两个分支都接收整张原图，不要求用户先裁出数字，也不根据文件名猜答案。保留 YOLO 和 ResNet 输出作为旧版对照。

实际试过 EasyOCR、PP-OCRv4、Florence 自带 OCR，再换成 PP-OCRv5 server。EasyOCR 和 PPv4 在这批手写照片上漏字、误字较多；Florence 曾把 `114514` 读成 `144514`，也漏掉主要数字 7。生产界面最终采用 PPv5，不把失败候选的结果拼接成一个伪造答案。

## 原有手写照片

从本机历史数据库恢复四张用户上传原图，按字节 SHA-256 核对。没有重新绘制、生成或手工裁剪这些验证输入。下面的“可见主要内容”是开发时对照片的人工阅读，不是用户确认的正式标签。

| 原图摘要前 16 位 | 可见主要内容 | PP-OCRv5 实际全文输出 | 需要说明 |
|---|---|---|---|
| 50ce9ec5e7c54470 | 7 | `7` | 原图有阴影及横线 |
| 3824f83293df422c | 114514，末尾有小点 | `114514.` | 保留末尾点，没有清洗成“更好看”的数字串 |
| b6a2add1fc947bb9 | 8 | `8` | 物体描述分支仍写了 letter B；OCR 与描述的分歧被保留 |
| 98f0f65e3fe7a2af | 6，背景另有笔迹 | `N=`、`6` | 主要数字读出；上方背景仍有额外识别，不能算整图完全正确 |

原图、各候选输出、最终结果、真实网页截图分别在本机 `var/round3/input/`、`var/round3/`、`var/round3/final-results.json`、`var/round3/browser/`。这些内容不上传公开仓库。浏览器测试再次从文件选择控件上传原始字节，调用真实识别接口，没有用预存输出替换响应。

## 全部五张公开照片

来源均为 scikit-image 0.24 的真实照片示例，来源记录继承原项目。原始结果见 `artifacts/vision/public-results.json`。

| 照片 | 当前区域输出 | OCR 输出 | 实际问题 |
|---|---|---|---|
| rocket | rocket | 空 | 能给出火箭位置；全图描述“在建筑顶部”不准确；没有把周围设施全部分开框出 |
| astronaut | woman、flag、helmet、human face | 空 | 能返回多个区域，但人脸属于人物的一部分，不能据此宣称有四个独立物体；背景描述有误 |
| camera | man in suit with camera on tripod | 空 | 人、相机、三脚架被合在一个区域，细分不完整 |
| chelsea | brown and white cat | `op` | 猫被识别；文字分支有误报，未隐藏 |
| coffee | coffee cup、spoon、spoon | 空 | 咖啡杯和勺子能分开；一把勺子返回两个重叠框。当前同名 IoU > 0.5 去重规则没有合并它们 |

这一轮完整推理中，新区域模型加 OCR 的单图耗时约 16.5–37.3 秒。原始 JSON 保留每次具体耗时，网页另计总请求推理时间；机器负载会影响速度。没有针对视频、密集人群、数十个小目标或严重模糊图片给出性能承诺。

## 可复现性和后端检查

- 新模型权重约 628 MB；另有词表、配置及原 YOLO/ResNet 基线权重。下载脚本只访问清单中的官方来源，逐文件核对 SHA-256，失败时明确报错。
- Florence 固定版本 `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`；权重摘要 `58757d657ff44051314c8030b68e04cb1bb618ca9a4885418f111f6fb708185a`。两个清单在 `artifacts/vision/*-source.json`。
- 直接导入经摘要校验的上游本地代码，使用 CPU eager attention。没有为可选 CUDA flash-attention 安装要求伪造模块，也没有修改上游模型来硬编码答案。
- 原图、预测文本、原始模型输出、坐标、模型版本、耗时和人工修改记录分开保存。更正不会覆盖原预测，也不会自动重新训练模型。
- 40 项自动化测试全部通过（15 条上游弃用警告，无失败），验证输入校验、复核版本冲突、请求幂等、坐标还原、模型文件摘要等工程行为；真实图片效果另列于上表。测试通过数量不能代替识别准确率。
- 浏览器检查涵盖原图上传、火箭区域、多区域输出、确认保存后重新打开、CSV 模型来源字段、390 px 布局；结果在 `artifacts/vision/browser-check.json`。

没有计算总准确率：四张私有照片已用于选型，五张公开示例也不是独立测试集；尚无完整逐字符、逐物体标注。下一步评估应使用未参与这轮比较的新照片，分别记录漏字、错字、漏框、错框和重复框。

## 来源与许可

- [Microsoft Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft)：模型仓库 LICENSE 为 MIT；其中建模代码头部保留 Apache-2.0 声明。下载清单固定权重和源码版本。
- [Autodistill Florence-2](https://github.com/autodistill/autodistill-florence-2/blob/main/autodistill_florence_2/model.py)：学习其任务处理方式，没有照搬 CUDA 硬编码或 `confidence=1.0`。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[RapidOCR](https://github.com/RapidAI/RapidOCR)：采用 PP-OCRv5 server ONNX 模型与实际推理库；遵守上游 Apache-2.0 许可。官方模型地址和摘要保存在下载清单。
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)：实际对比过，未用于当前主推理。
- 已通过 skill-installer 安装 [Hugging Face 的 huggingface-best skill](https://github.com/huggingface/skills/tree/main/skills/huggingface-best)，用于选型参考；skill 不是模型，也不会自动提高识别质量。

原考核要求的手写 PyTorch ResNet 和训练记录仍在数字版目录。新的通用识别能力来自上述预训练模型与工程集成，不能在答辩中称为自行训练的大模型。
