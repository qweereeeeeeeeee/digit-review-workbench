import base64
from contextlib import asynccontextmanager
import csv
import hashlib
import io
import json
import os
from typing import Literal
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .engine import ROOT, ObjectEngine, decode, display_label
from .store import Store, Conflict


class Photo(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:uuid.UUID
    name:str=Field(min_length=1,max_length=160)
    data:str=Field(min_length=1,max_length=14_000_000)
    crop:tuple[float,float,float,float]|None=None

    @model_validator(mode='after')
    def coordinates(self):
        if self.crop is not None:
            import math
            if not all(math.isfinite(v) for v in self.crop):raise ValueError('框选坐标必须是有限数值')
        return self


class Decision(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:uuid.UUID
    version:int=Field(ge=0,strict=True)
    label:str|None=Field(default=None,max_length=100)
    status:Literal['confirmed','unknown','pending']
    note:str=Field(default='',max_length=500)

    @model_validator(mode='after')
    def label_required(self):
        if self.status=='confirmed':
            if not self.label or not self.label.strip():raise ValueError('请填写确认的物体名称')
            self.label=self.label.strip()
        elif self.label is not None:raise ValueError('无法判断或撤回时不应填写确认标签')
        return self


def create_app(db_path=None,engine=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store=Store(db_path or os.environ.get('OBJECT_DB',str(ROOT/'var/objects.sqlite3')))
        app.state.engine=engine or ObjectEngine()
        yield
    app=FastAPI(title='图片识别与复核',version='3.0.0',lifespan=lifespan)

    @app.exception_handler(KeyError)
    async def missing(request,exc):return JSONResponse({'detail':'记录不存在'},status_code=404)
    @app.exception_handler(Conflict)
    async def conflict(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)

    @app.middleware('http')
    async def limits(request,call_next):
        if request.method=='POST':
            if request.headers.get('origin') and request.headers['origin']!=str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'不接受跨站写入'},status_code=403)
            data=bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data)>15*1024*1024:return JSONResponse({'detail':'请求过大，请缩小图片'},status_code=413)
            request._body=bytes(data)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    @app.get('/')
    def index():return FileResponse(ROOT/'objects/web/index.html')
    @app.get('/api/health')
    def health():return {'status':'ready','model':'ResNet50 ImageNet-1K V2','classes':1000,'detector_classes':80,'sha256':app.state.engine.sha}
    @app.get('/api/categories')
    def categories():return [dict(id=i,label=s,display=display_label(s)) for i,s in enumerate(app.state.engine.categories)]
    @app.get('/api/detection-categories')
    def detection_categories():
        from .detector import NAMES_ZH
        return [dict(id=i,label=s,display=NAMES_ZH[i]) for i,s in app.state.engine.detector.names.items()]
    @app.post('/api/recognize')
    def recognize(body:Photo):
        try:
            raw=base64.b64decode(body.data,validate=True)
            image=decode(raw)
        except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc)) from exc
        fingerprint=hashlib.sha256(raw+json.dumps([body.name,body.crop],ensure_ascii=False).encode()).hexdigest()
        cached=app.state.store.cached(str(body.request_id),fingerprint)
        if cached:return cached
        try:result=app.state.engine.predict(image,body.crop)
        except ValueError as exc:raise HTTPException(422,str(exc)) from exc
        result['original_size']=list(image.size)
        return app.state.store.save(body.name,raw,result,str(body.request_id),fingerprint)
    @app.get('/api/records')
    def records(limit:int=Query(20,ge=1,le=100),offset:int=Query(0,ge=0)):
        return app.state.store.list(limit,offset)
    @app.get('/api/records/{record_id}')
    def detail(record_id:str):return app.state.store.get(record_id)
    @app.get('/api/records/{record_id}/image')
    def original(record_id:str):
        image=decode(app.state.store.original(record_id))
        out=io.BytesIO();image.save(out,format='PNG')
        return Response(out.getvalue(),media_type='image/png',headers={'Cache-Control':'private, max-age=3600'})
    @app.post('/api/records/{record_id}/review')
    def review(record_id:str,body:Decision):
        return app.state.store.review(record_id,body.version,body.label,body.status,body.note,str(body.request_id))
    @app.get('/api/export')
    def export():
        fields=['id','name','model_prediction','model_score','whole_view_prediction','needs_review','final_label','status','version','model_sha','crop','digest','detections','detector_sha']
        out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in app.state.store.export_rows():
            r=row['result'];top=r['candidates'][0]
            flat=dict(id=row['id'],name=row['name'],model_prediction=top['label'],model_score=top['score'],
                      whole_view_prediction=r['whole_candidates'][0]['label'],needs_review=r['needs_review'],
                      final_label=row['final_label'],status=row['status'],version=row['version'],
                      model_sha=r['model_sha'],crop=json.dumps(r['crop']),digest=row['digest'],detections=json.dumps(r.get('detection',{}).get('items',[]),ensure_ascii=False),detector_sha=r.get('detection',{}).get('sha256',''))
            for key,value in flat.items():
                if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):flat[key]="'"+value
            writer.writerow(flat)
        return Response('\ufeff'+out.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="image-review.csv"'})
    app.mount('/static',StaticFiles(directory=ROOT/'objects/web'),name='static')
    return app

app=create_app()
