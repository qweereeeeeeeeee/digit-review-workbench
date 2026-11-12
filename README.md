# 图片识别与复核

上传生活照片，自动框出其中的常见物体，再人工核对、保存或导出。主检测使用官方 YOLO11s，辅助分类使用官方 ResNet50，均在本机 CPU 运行。不是从零训练的通用大模型。

## 启动

本次验证环境：Intel macOS、Python 3.12。无需 API 密钥；首次下载权重需要联网，推理无需上传照片。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_object_model.py
sh scripts/start_objects.sh
```

打开 http://127.0.0.1:8768，选择 JPG / PNG / WebP 后点击“识别图片”。照片可以包含多个主体，也可以拖动框选局部。绿色编号与右侧物体列表对应。人工填写或修改确认名称后保存；原模型结果不会被覆盖。

权重不放进 Git 的普通文件历史；下载脚本使用官方地址和完整 SHA-256 校验。数据库 `var/objects.sqlite3` 包含原图和修改记录，不提交公开仓库。服务只绑定本机。

## 能力和边界

- 检测支持 COCO 的 80 类常见物体，包括人、猫狗、车辆、杯子、手机等；可一次检测多个。
- 折叠的辅助分类支持 ImageNet 的 1000 个固定类别，只回答画面更像哪个类别。两套输出互不替代。
- 没有框时显示未检测到；这不保证照片没有物体。可能漏检、误检，不支持 OCR 或任意物品名称。
- 确认、无法判断、撤回均留下修改记录。重复请求有回执，过期版本返回 409；CSV 保留原预测、坐标、模型摘要及人工结论。

## 实测与复现

五张公开真实照片完整运行结果在 [detection-smoke.json](artifacts/objects/detection-smoke.json)，来源和文件摘要在 [sources.json](artifacts/objects/examples/sources.json)。它们不是独立测试集，不能计算通用准确率。

咖啡图检测到杯子、勺子和桌子；猫图检测到猫；宇航员图检测到人，但误报了球；摄影者图检测到人，但误报了船；火箭图没有检测结果。失败同样保留，未挑掉。

```sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/evaluate_objects.py
```

[开源代码调研与选型](docs/research/OBJECT_PROJECTS.md)说明实际读了什么、采用什么、没做什么。[改造说明](docs/OBJECT_UPGRADE.md)解释旧版在手机照片上的失败原因。

## 运行演示

[新版连续运行视频](artifacts/objects/demo/物体识别演示.mp4)（36.56 秒）展示咖啡照片检测、宇航员照片误报及人工更正、火箭照片未检测到。使用 Playwright 操作真实浏览器，没有替换模型输出。仅裁去录制底部空白；录制脚本为 `scripts/record_objects.py`，依赖 `requirements-demo.txt`。

## 原考核版本

原手写 PyTorch ResNet、真实训练历史、数字测试结果和视频继续保留，见 [数字版说明](docs/DIGIT_README.md)。旧版入口仍是 `sh scripts/start.sh` / 8766。原来的数字识别视频不能当成本次物体检测演示。

本项目使用 AI 辅助开发；上游模型、自己编写的服务代码、公开数据和用户私有图片分开说明。新增 `objects/` 组件采用 AGPL-3.0，见 [许可](objects/LICENSE)；其他部分的来源见原参考资料。
