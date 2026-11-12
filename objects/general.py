"""Pinned upstream inference: Microsoft Florence-2 + PaddleOCR via RapidOCR.

No guessed labels, filename rules, random-weight fallback, or fabricated confidence.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'artifacts/vision'


def verify_bundle(folder,manifest):
    meta=json.loads((ASSETS/manifest).read_text())
    for entry in meta['files']:
        path=ASSETS/folder/entry['name']
        if not path.is_file():raise RuntimeError('缺少模型文件，请运行 scripts/download_vision_models.py')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError(f'模型文件校验失败：{entry["name"]}')
    return meta


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


def box_to_original(box,region_size,original_size,origin):
    x1,y1,x2,y2=map(float,box);rw,rh=region_size;ow,oh=original_size;ox,oy=origin
    if not np.isfinite([x1,y1,x2,y2]).all():return None
    x1,x2=np.clip([x1,x2],0,rw);y1,y2=np.clip([y1,y2],0,rh)
    if x2-x1<2 or y2-y1<2:return None
    return [float((x1+ox)/ow),float((y1+oy)/oh),float((x2+ox)/ow),float((y2+oy)/oh)]


class GeneralVision:
    def __init__(self):
        self.florence_meta=verify_bundle('florence','florence-source.json')
        self.ocr_meta=verify_bundle('ppocr','ppocr-source.json')
        os.environ.setdefault('HF_HOME',str(ROOT/'var/huggingface'))
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        folder=ASSETS/'florence'
        # Import hash-verified local upstream source directly. Transformers 4.44's
        # dynamic import scanner incorrectly requires optional CUDA flash_attn on CPU.
        # The upstream model itself already guards that import; no source patch needed.
        package=types.ModuleType('lingjing_pinned_florence')
        package.__path__=[str(folder)]
        sys.modules[package.__name__]=package
        Model=importlib.import_module(package.__name__+'.modeling_florence2').Florence2ForConditionalGeneration
        Processor=importlib.import_module(package.__name__+'.processing_florence2').Florence2Processor
        self.model=Model.from_pretrained(folder,local_files_only=True,use_safetensors=True,
                                        torch_dtype=torch.float32,attn_implementation='eager').eval()
        self.processor=Processor.from_pretrained(folder,local_files_only=True)
        from rapidocr import RapidOCR,OCRVersion,ModelType
        p=ASSETS/'ppocr'
        self.ocr=RapidOCR(params={
            'Det.ocr_version':OCRVersion.PPOCRV5,'Det.model_type':ModelType.SERVER,
            'Rec.ocr_version':OCRVersion.PPOCRV5,'Rec.model_type':ModelType.SERVER,
            'Det.model_path':str(p/'ch_PP-OCRv5_server_det.onnx'),
            'Rec.model_path':str(p/'ch_PP-OCRv5_rec_server_infer.onnx'),
            'Cls.model_path':str(p/'ch_ppocr_mobile_v2.0_cls_infer.onnx'),
            'Global.font_path':str(p/'FZYTK.TTF'),'Global.log_level':'warning',
            'EngineConfig.onnxruntime.intra_op_num_threads':2,
            'EngineConfig.onnxruntime.inter_op_num_threads':2})
        self.lock=threading.Lock()

    def run_task(self,image,task):
        inputs=self.processor(text=task,images=image,return_tensors='pt')
        with torch.inference_mode():
            ids=self.model.generate(input_ids=inputs['input_ids'],pixel_values=inputs['pixel_values'],
                                    max_new_tokens=512,num_beams=3,do_sample=False)
        raw=self.processor.batch_decode(ids,skip_special_tokens=False)[0]
        parsed=self.processor.post_process_generation(raw,task=task,image_size=image.size)
        return dict(raw=raw,parsed=parsed,token_limit_reached=ids.shape[-1]>=512)

    def predict(self,image,crop=None):
        from .engine import selected_region,display_label
        start=time.perf_counter();region=selected_region(image,crop)
        origin=(round(crop[0]*image.width),round(crop[1]*image.height)) if crop else (0,0)
        with self.lock:
            caption=self.run_task(region,'<CAPTION>')
            dense=self.run_task(region,'<DENSE_REGION_CAPTION>')
            ocr_start=time.perf_counter()
            # RapidOCR's ndarray interface expects BGR; PIL images remain RGB upstream.
            ocr=self.ocr(np.asarray(region)[:,:,::-1].copy())
            ocr_seconds=round(time.perf_counter()-ocr_start,3)
        regions=[];data=dense['parsed']['<DENSE_REGION_CAPTION>']
        for label,coords in zip(data['labels'],data['bboxes']):
            label=label.strip();box=box_to_original(coords,region.size,image.size,origin)
            if not label or box is None:continue
            if any(x['label']==label and iou(x['box'],box)>.5 for x in regions):continue
            regions.append(dict(label=label,display=({'notebook':'笔记本','white paper':'白纸','paper':'纸'}.get(label,display_label(label))),box=box,score=None,source='Florence-2'))
        texts=[]
        if ocr.boxes is not None:
            for quad,text,score in zip(ocr.boxes,ocr.txts,ocr.scores):
                points=np.asarray(quad);box=box_to_original([*points.min(0),*points.max(0)],region.size,image.size,origin)
                if box is None:continue
                original_quad=[[float((x+origin[0])/image.width),float((y+origin[1])/image.height)] for x,y in points]
                texts.append(dict(text=text,score=float(score),box=box,quad=original_quad,source='PP-OCRv5 server'))
        return dict(caption=caption['parsed']['<CAPTION>'],regions=regions,texts=texts,
                    raw_caption=caption,raw_regions=dense,
                    model='Microsoft Florence-2-base-ft + PaddleOCR PP-OCRv5 server (RapidOCR)',
                    florence_revision=self.florence_meta['revision'],
                    florence_sha=next(f['sha256'] for f in self.florence_meta['files'] if f['name']=='model.safetensors'),
                    ocr_model_hashes={f['name']:f['sha256'] for f in self.ocr_meta['files'] if f['name'].endswith('.onnx')},
                    seconds=round(time.perf_counter()-start,3),ocr_seconds=ocr_seconds,
                    token_limit_reached=dense['token_limit_reached'],
                    scope='区域识别与文字读取独立执行；生成式描述可能遗漏或说错，不提供虚构准确率。OCR 得分也不保证正确。')
