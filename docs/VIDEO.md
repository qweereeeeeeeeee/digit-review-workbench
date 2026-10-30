# 运行演示

新版视频：`artifacts/review/demo-v2/运行演示.mp4`。

画面依次展示打开工作区、载入图片、检查一个识别错误、保存并查看记录、刷新恢复、导出表格、导入重复图片和查看实验结果。留出阅读时间，不添加宣传标题、合成配音或背景音乐。

视频由 Playwright 驱动本机 Chrome 连续录制，再转换为 MP4。所有操作经过实际页面和后端，没有替换接口响应。界面署名“演示”是演示记录，不代表某位真实用户进行了标注。纠正依据是公开样本 555 的已有标签；上传图片来自公开样本 0，不是新采集的手写照片。自动化录制说明保留在这里和同目录 `browser_checks.json` 中。

新视频使用独立数据库，避免改变日常工作区的记录。复录方法：

```sh
REVIEW_DB=var/walkthrough.sqlite3 .venv/bin/python -m uvicorn workbench.api:app --host 127.0.0.1 --port 8767
# 另开一个终端，在项目目录执行
.venv/bin/python scripts/record_walkthrough.py
```

依赖见 `requirements-demo.txt`；默认使用 macOS Chrome，可以通过 `CHROME_PATH` 指定路径。重复运行会新增演示批次。旧录制保留在 `artifacts/review/demo/`，原数据校验仍针对旧录制的导出文件。

录屏复用 `imageio-ffmpeg` 提供的本机编码器，在忽略入库的 `var/browser-runtime/` 下建立链接。此路径适配当前 Playwright 1.51.0 和 Intel macOS，其他平台需要调整编码器名称。

最终视频为 40 秒、25 帧/秒。转换时裁掉编码器在底部补出的灰边，保留实际页面内容；无合成配音。
