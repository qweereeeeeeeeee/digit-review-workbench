import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import uuid
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from workbench.api import create_app
from workbench.store import Store,Conflict
from workbench.inference import decode_image


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'db.sqlite')) as c:yield c


def request_id():return str(uuid.uuid4())


def demo(c):
    result=c.post('/api/batches/demo',json={'request_id':request_id()})
    assert result.status_code==200
    batch=result.json()['id'];items=c.get(f'/api/batches/{batch}/items').json()['items']
    return batch,items


def test_demo_retry_and_persistence(client):
    body={'request_id':request_id()}
    a=client.post('/api/batches/demo',json=body);b=client.post('/api/batches/demo',json=body)
    assert a.json()==b.json()
    assert len(client.get('/api/batches').json())==1
    store=Store(client.app.state.store.path)
    assert store.batches()[0]['total']==270


def test_review_preserves_prediction_and_retries(client):
    batch,items=demo(client);item=items[0];original=item['prediction'];new=(original+1)%10
    body={'request_id':request_id(),'version':0,'label':new,'note':'测试纠错，不作为真实数据标签'}
    a=client.post(f"/api/items/{item['id']}/review",json=body)
    assert a.status_code==200
    assert client.post(f"/api/items/{item['id']}/review",json=body).json()==a.json()
    saved=client.get(f"/api/items/{item['id']}").json()
    assert saved['prediction']==original and saved['final_label']==new
    assert saved['status']=='corrected' and len(saved['events'])==1
    stale=dict(body,request_id=request_id(),label=original)
    assert client.post(f"/api/items/{item['id']}/review",json=stale).status_code==409
    reopened=client.post(f"/api/items/{item['id']}/review",json={'request_id':request_id(),'version':1,'action':'reopen'})
    assert reopened.status_code==200 and reopened.json()['final_label'] is None
    assert len(client.get(f"/api/items/{item['id']}").json()['events'])==2


def test_concurrent_review_one_winner(client):
    batch,items=demo(client);item=items[0];store=client.app.state.store
    def edit(label):
        try:return store.review(item['id'],0,label,'review','','test',request_id())
        except Conflict:return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(edit,[1,2]))
    assert results.count('conflict')==1
    assert len(store.get(item['id'])['events'])==1


def test_export_distinguishes_unreviewed_and_formula_escape(client):
    batch,items=demo(client);item=items[0]
    client.post(f"/api/items/{item['id']}/review",json={'request_id':request_id(),'version':0,'label':item['prediction']})
    r=client.get(f'/api/batches/{batch}/export');rows=list(csv.DictReader(io.StringIO(r.text.lstrip('\ufeff'))))
    assert len(rows)==270
    assert sum(row['final_label']!='' for row in rows)==1
    with client.app.state.store.connect() as db:db.execute('UPDATE items SET name=? WHERE id=?',('=1+1',item['id']))
    assert "'=1+1" in client.get(f'/api/batches/{batch}/export').text


def test_upload_duplicates_and_invalid_atomicity(client):
    arr=np.zeros((24,24),dtype=np.uint8);arr[4:20,10:14]=255
    f=io.BytesIO();Image.fromarray(arr).save(f,format='PNG');encoded=base64.b64encode(f.getvalue()).decode()
    body={'request_id':request_id(),'name':'upload','images':[{'name':'a.png','data':encoded},{'name':'b.png','data':encoded}]}
    response=client.post('/api/batches/import',json=body)
    assert response.status_code==200,response.text
    assert response.json()['inserted']==1 and response.json()['duplicates']==1
    changed=dict(body,name='other')
    assert client.post('/api/batches/import',json=changed).status_code==409
    body['request_id']=request_id();body['images'].append({'name':'broken.png','data':'!bad'})
    assert client.post('/api/batches/import',json=body).status_code==422
    assert len(client.get('/api/batches').json())==1


def test_validation_and_cross_site(client):
    assert client.get('/api/items/missing').status_code==404
    assert client.get('/api/batches/missing/items').status_code==404
    assert client.post('/api/batches/demo',headers={'origin':'https://example.com'},json={'request_id':request_id()}).status_code==403
    batch,items=demo(client)
    for label in [None,True,10,-1,'3']:
        assert client.post(f"/api/items/{items[0]['id']}/review",json={'request_id':request_id(),'version':0,'label':label}).status_code==422
    assert client.get(f'/api/batches/{batch}/items?limit=1000').status_code==422


def test_preprocessing_blank_and_polarity():
    def encode(arr):
        f=io.BytesIO();Image.fromarray(arr).save(f,format='PNG');return base64.b64encode(f.getvalue()).decode()
    with pytest.raises(ValueError):decode_image(encode(np.zeros((16,16),dtype=np.uint8)),'light',False)
    x=np.zeros((16,16),dtype=np.uint8);x[3:13,6:10]=255
    a=decode_image(encode(x),'light',True)
    b=decode_image(encode(255-x),'dark',True)
    assert np.allclose(a,b)
