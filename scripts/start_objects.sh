#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -f artifacts/objects/resnet50-11ad3fa6.pth ] || [ ! -f artifacts/objects/yolo11s.pt ]; then
  echo '缺少官方模型，请先运行 .venv/bin/python scripts/download_object_model.py'
  exit 1
fi
if [ ! -f artifacts/vision/florence/model.safetensors ] || [ ! -f artifacts/vision/ppocr/ch_PP-OCRv5_rec_server_infer.onnx ]; then
  echo '缺少区域识别 / OCR 模型，请运行 .venv/bin/python scripts/download_vision_models.py'
  exit 1
fi
exec .venv/bin/python -m uvicorn objects.api:app --host 127.0.0.1 --port 8768
