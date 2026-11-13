# 图片识别与复核

上传一张照片，同时查看里面的物体区域和文字、数字，再核对、保存或导出。当前版本使用微软 Florence-2 做区域识别，使用 PaddleOCR PP-OCRv5 server 读文字，通过 RapidOCR 在本机运行。采用上游预训练权重，没有把这些模型写成自己的训练成果。

## 启动

本次验证环境：Intel macOS、Python 3.12。无需 API 密钥；首次下载权重需要联网，推理无需上传照片。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_object_model.py
.venv/bin/python scripts/download_vision_models.py
sh scripts/start_objects.sh
```

打开 http://127.0.0.1:8768，选择 JPG / PNG / WebP 后点击“识别图片”。默认处理整张照片；绿色编号对应物体区域，蓝色 T 编号对应文字区域。也可拖动框选局部。人工填写或修改确认结果后保存，原模型输出不会被覆盖。CPU 需要十几到几十秒，不是实时视频识别。

权重不放进 Git 的普通文件历史；下载脚本使用官方地址和完整 SHA-256 校验。数据库 `var/objects.sqlite3` 包含原图和修改记录，不提交公开仓库。服务只绑定本机。

## 能力和边界

- 区域识别不再受原 COCO 80 类限制。本次实际识别出了火箭；同一图可返回多个区域，但也会遗漏、重复或用一个大框描述多个物体。
- OCR 独立处理整张原图，不把背景和多位数字一起压成 8×8。此前上传的四张手写照片分别读到 `7`、`114514.`、`8`、`N=` 和 `6`；最后一张有额外背景误读，结果未被删掉。
- 区域模型没有提供校准置信度，页面不为它伪造百分比。OCR 得分是模型内部输出，不是这张照片的正确概率。英文描述可能说错，例如把数字 8 描述成字母 B。
- 旧 YOLO11s 和 ResNet50 留在折叠对照区。不同模型结果独立保留，不将几个分数平均成所谓更高准确率。
- 确认、无法判断、撤回均留下修改记录。重复请求有回执，过期版本返回 409；CSV 保留原预测、坐标、模型摘要及人工结论。

## 实测与复现

本轮对全部四张已有私有手写照片和五张公开照片进行了实际推理。[逐项验证记录](docs/GENERAL_VISION_CHECK.md)同时列出成功与失败；[公开原始输出](artifacts/vision/public-results.json)包含坐标、耗时、模型版本和文件摘要。公开图片来源见 [sources.json](artifacts/objects/examples/sources.json)。私有照片及完整输出只保存在本机 `var/round3/`，不提交公开仓库。

这些照片已经用于开发比较，不是独立测试集，不能据此计算通用准确率。没有逐物体标注全集，也不能将“识别到几个区域”当成整图识别完整。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_general_vision.py
```

只复现公开图片不需要私有文件。下载模型后可运行测试；浏览器检查另需 `requirements-demo.txt`。本地验证 40 项自动化测试通过，测试数量不代表识别正确率。

上游来源：[Microsoft Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft)、[Autodistill Florence-2](https://github.com/autodistill/autodistill-florence-2)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[RapidOCR](https://github.com/RapidAI/RapidOCR)。[选型记录](docs/research/GENERAL_VISION_ROUND2.md)保留安装前的调查过程；[改造说明](docs/OBJECT_UPGRADE.md)解释旧模型为何不适合手机照片。

## 运行演示

[上一版 YOLO 运行视频](artifacts/objects/demo/物体识别演示.mp4)（36.56 秒）保留当时的误报和火箭未检测到结果；它不是当前 Florence-2 + OCR 的演示。[新版自动化验证视频](artifacts/vision/demo/新版模型实际运行.mp4)保留开发过程；最终提交使用用户提供的原始现场录屏，只随本地交付包提供。

开发时间线见 [开发过程说明](docs/DEVELOPMENT_PROCESS.md)，基础讲解见 [技术与代码说明](docs/CODE_WALKTHROUGH.md)。

最新完整说明与个人感悟见 [交付说明](docs/DELIVERY_REPORT.md)，考核逐项对应见 [验收表](docs/ACCEPTANCE.md)。

## 原考核版本

原手写 PyTorch ResNet、真实训练历史、数字测试结果和视频继续保留，见 [数字版说明](docs/DIGIT_README.md)。旧版入口仍是 `sh scripts/start.sh` / 8766。原来的数字识别视频不能当成本次物体检测演示。

本项目使用 AI 辅助开发；上游模型、自己编写的服务代码、公开数据和用户私有图片分开说明。新增 `objects/` 组件采用 AGPL-3.0，见 [许可](objects/LICENSE)；其他部分的来源见原参考资料。
