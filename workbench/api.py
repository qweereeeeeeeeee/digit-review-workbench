from contextlib import asynccontextmanager
import csv
import io
import json
import os
from pathlib import Path
from typing import Literal
import uuid
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .store import Store, Conflict
from .inference import Engine, ROOT, decode_image


class Upload(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=1,max_length=120)
    data:str=Field(min_length=1,max_length=2_800_000)


class ImportRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:uuid.UUID
    name:str=Field(min_length=1,max_length=80)
    images:list[Upload]=Field(min_length=1,max_length=40)
    polarity:Literal['light','dark']='light'
    normalize:bool=False


class DemoRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:uuid.UUID


class Review(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:uuid.UUID
    version:int=Field(ge=0,strict=True)
    label:int|None=Field(default=None,ge=0,le=9,strict=True)
    action:Literal['review','reopen','unreadable']='review'
    note:str=Field(default='',max_length=500)
    actor:str=Field(default='本机用户',min_length=1,max_length=50)

    @model_validator(mode='after')
    def check(self):
        if self.action=='review' and self.label is None:raise ValueError('确认时必须选择一个数字')
        if self.action!='review' and self.label is not None:raise ValueError('此操作不接受标签')
        return self


def create_app(db_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store=Store(db_path or os.environ.get('REVIEW_DB',str(ROOT/'var/review.sqlite3')))
        app.state.engine=Engine()
        yield

    app=FastAPI(title='数字图片复核台',version='2.0.0',lifespan=lifespan)

    @app.exception_handler(KeyError)
    async def missing(request,exc):return JSONResponse({'detail':str(exc)},status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)

    @app.middleware('http')
    async def bounds(request,call_next):
        if request.method=='POST':
            # Reject cross-site browser writes to this local service.
            origin=request.headers.get('origin')
            if origin and origin != str(request.base_url).rstrip('/'):
                return JSONResponse({'detail':'不接受跨站写入'},status_code=403)
            body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>12*1024*1024:return JSONResponse({'detail':'每批请求不能超过 12 MB'},status_code=413)
            request._body=bytes(body)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    @app.get('/')
    def home():return FileResponse(ROOT/'workbench/web/index.html')

    @app.get('/api/health')
    def health():return {'status':'ok','model_sha':app.state.engine.sha}

    @app.get('/api/batches')
    def batches():return app.state.store.batches()

    @app.post('/api/batches/import')
    def import_images(body:ImportRequest):
        entries=[]
        for img in body.images:
            try:pixels=decode_image(img.data,body.polarity,body.normalize)
            except (ValueError,TypeError) as exc:raise HTTPException(422,f'{img.name}: {exc}') from exc
            entries.append(dict(name=img.name,pixels=pixels,source_image=img.data,preprocess=f'polarity={body.polarity};center={body.normalize};box8'))
        predicted=app.state.engine.predict(entries)
        return app.state.store.create_batch(body.name,'用户图片',predicted,str(body.request_id))

    @app.post('/api/batches/demo')
    def demo(body:DemoRequest):
        entries=[dict(name=f"test-{e['sample_id']}.png",pixels=e['pixels'],preprocess='UCI /16; no image conversion') for e in app.state.engine.examples]
        return app.state.store.create_batch('保留测试集 · 270 张','公开测试样本',app.state.engine.predict(entries),str(body.request_id))

    @app.get('/api/batches/{batch_id}/items')
    def items(batch_id:str,status:Literal['all','pending','confirmed','corrected','unreadable']='all',
              order:Literal['uncertain','original']='uncertain',limit:int=Query(default=60,ge=1,le=100),offset:int=Query(default=0,ge=0)):
        return app.state.store.items(batch_id,status,order,limit,offset)

    @app.get('/api/items/{item_id}')
    def detail(item_id:str):return app.state.store.get(item_id)

    @app.post('/api/items/{item_id}/review')
    def review(item_id:str,body:Review):
        return app.state.store.review(item_id,body.version,body.label,body.action,body.note,body.actor,str(body.request_id))

    @app.get('/api/batches/{batch_id}/export')
    def export(batch_id:str):
        rows=app.state.store.export(batch_id)
        out=io.StringIO(newline='')
        fields=['name','prediction','confidence','status','final_label','version','model_sha','preprocess','digest']
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in rows:
            # Spreadsheet formula injection protection for user-provided filenames.
            if row['name'].lstrip().startswith(('=','+','-','@','\t','\r')):row['name']="'"+row['name']
            writer.writerow(row)
        return Response('\ufeff'+out.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="review-results.csv"'})

    @app.get('/api/experiment')
    def experiment():
        path=ROOT/'artifacts/review/queue_experiment.json'
        if not path.exists():raise HTTPException(503,'实验尚未运行')
        return json.loads(path.read_text())

    @app.get('/api/model')
    def model():return app.state.engine.report

    @app.get('/experiment.png')
    def plot():return FileResponse(ROOT/'artifacts/review/queue_experiment.png')

    app.mount('/static',StaticFiles(directory=ROOT/'workbench/web'),name='static')
    return app

app=create_app()
