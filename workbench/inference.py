import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from src.model import DigitResNet
from src.metrics import softmax

ROOT=Path(__file__).resolve().parents[1]


class Engine:
    def __init__(self):
        run=ROOT/'artifacts/run-001'
        self.report=json.loads((run/'metrics.json').read_text())
        self.sha=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest()
        if self.sha != self.report['model_sha256']:raise RuntimeError('模型与实验记录不匹配')
        torch.set_num_threads(2)
        self.model=DigitResNet()
        self.model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True))
        self.model.eval()
        self.lock=threading.Lock()
        self.examples=json.loads((run/'examples.json').read_text())

    def predict(self, entries):
        array=np.asarray([e['pixels'] for e in entries],dtype=np.float32)
        with self.lock,torch.inference_mode():
            probs=softmax(self.model(torch.from_numpy(array[:,None])).numpy(),self.report['temperature'])
        results=[]
        for entry,p in zip(entries,probs):
            pixels=np.asarray(entry['pixels'],dtype='<f4')
            # Uploaded originals define identity: different originals may collapse to the same 8x8 input.
            identity=base64.b64decode(entry['source_image']) if entry.get('source_image') else pixels.tobytes()
            results.append(dict(**entry,digest=hashlib.sha256(identity).hexdigest(),
                prediction=int(p.argmax()),confidence=float(p.max()),probabilities=p.tolist(),model_sha=self.sha))
        return results


def decode_image(encoded, polarity, normalize):
    try:
        data=base64.b64decode(encoded,validate=True)
        if len(data)>2*1024*1024:raise ValueError('单张图片不能超过 2 MB')
        image=Image.open(io.BytesIO(data))
        if image.format not in {'PNG','JPEG','WEBP'}:raise ValueError('只支持 PNG、JPEG、WebP 图片')
        if image.width*image.height>4_000_000:raise ValueError('图片不能超过 400 万像素')
        image=ImageOps.exif_transpose(image).convert('L')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('无法解码图片') from exc
    arr=np.asarray(image,dtype=np.float32)/255
    if polarity=='dark':arr=1-arr
    if arr.max()-arr.min()<.03:raise ValueError('图片没有可辨认的笔画变化')
    if normalize:
        # Explicit user option: improve framing, but never pretend to solve domain shift.
        mask=arr>max(.12,float(arr.max())*.25)
        ys,xs=np.where(mask)
        if not len(xs):raise ValueError('没有找到笔画')
        arr=arr[ys.min():ys.max()+1,xs.min():xs.max()+1]
        h,w=arr.shape
        side=max(h,w)+max(2,round(max(h,w)*.18))
        padded=np.zeros((side,side),dtype=np.float32)
        top=(side-h)//2;left=(side-w)//2;padded[top:top+h,left:left+w]=arr
        arr=padded
    image=Image.fromarray(np.uint8(np.clip(arr,0,1)*255))
    pixels=np.asarray(image.resize((8,8),Image.Resampling.BOX),dtype=np.float32)/255
    if pixels.max()-pixels.min()<.03:raise ValueError('缩小后笔画不清晰，请裁剪为单个数字')
    return pixels.tolist()
