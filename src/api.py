"""Local inference service. Run: python -m uvicorn src.api:app --host 127.0.0.1."""
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .model import DigitResNet
from .metrics import softmax

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get('DIGITS_RUN', str(ROOT/'artifacts/run-001')))


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pixels: list[list[float]] = Field(min_length=8, max_length=8)
    explain: bool = False

    @field_validator('pixels', mode='before')
    @classmethod
    def validate_pixels(cls, value):
        if not isinstance(value, list) or len(value) != 8:
            raise ValueError('pixels must be an 8 by 8 array')
        for row in value:
            if not isinstance(row, list) or len(row) != 8:
                raise ValueError('pixels must be an 8 by 8 array')
            if any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 or v > 1 for v in row):
                raise ValueError('pixels must contain finite numbers between 0 and 1')
        return value


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(2)
    report = json.loads((RUN/'metrics.json').read_text())
    actual = hashlib.sha256((RUN/'model.pt').read_bytes()).hexdigest()
    if actual != report['model_sha256']:
        raise RuntimeError('Model checksum does not match evaluation report')
    model = DigitResNet()
    model.load_state_dict(torch.load(RUN/'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    app.state.model = model
    app.state.report = report
    app.state.examples = json.loads((RUN/'examples.json').read_text())
    yield


app = FastAPI(title='墨鉴 · Reliable Digits', version='1.0.0', lifespan=lifespan)


@app.middleware('http')
async def bound_request(request, call_next):
    # Bound actual streamed bytes, including chunked requests without Content-Length.
    if request.method in ('POST', 'PUT', 'PATCH'):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 16384:
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail':'Request exceeds 16 KiB'}, status_code=413)
        request._body = bytes(body)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/health')
def health():
    return {'status':'ok', 'model_sha256':app.state.report['model_sha256'], 'device':'cpu'}


@app.get('/api/report')
def report():
    return app.state.report


@app.get('/api/examples')
def examples():
    return app.state.examples


@app.post('/api/predict')
def predict(request: PredictionRequest):
    begin = time.perf_counter()
    array = np.asarray(request.pixels, dtype=np.float32)
    if float(array.max()-array.min()) < .03 or float(array.sum()) < .1:
        raise HTTPException(422, '图像没有足够的笔画变化，请输入单个数字。')
    tensor = torch.from_numpy(array[None,None])
    model = app.state.model; info = app.state.report
    with torch.inference_mode():
        logits = model(tensor).numpy()
        probs = softmax(logits, info['temperature'])[0]
        heatmap = None
        if request.explain:
            # Occlusion sensitivity is evidence of input dependence, not causal proof.
            perturbed = tensor.repeat(64,1,1,1)
            for i in range(64): perturbed[i,0,i//8,i%8] = 0
            changed = softmax(model(perturbed).numpy(), info['temperature'])
            heatmap = (probs.max()-changed[:,int(probs.argmax())]).reshape(8,8).tolist()
    accepted = bool(probs.max() >= info['threshold'])
    return dict(request_id=str(uuid.uuid4()), prediction=int(probs.argmax()), confidence=float(probs.max()),
                accepted=accepted, decision='可供参考' if accepted else '建议人工复核',
                threshold=info['threshold'], probabilities=probs.tolist(), occlusion=heatmap,
                elapsed_ms=(time.perf_counter()-begin)*1000, model_sha256=info['model_sha256'])


@app.get('/')
def home():
    return FileResponse(ROOT/'web/index.html')


@app.get('/figures/{name}')
def figure(name: str):
    if name not in {'training.png','confusion.png','distribution.png','reliability.png'}:
        raise HTTPException(404)
    return FileResponse(RUN/name)

app.mount('/static', StaticFiles(directory=ROOT/'web'), name='static')
