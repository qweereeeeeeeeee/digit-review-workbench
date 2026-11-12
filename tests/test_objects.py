"""Use the real checkpoint and real public photographs, never mocked predictions."""
import base64
import csv
import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from objects.api import create_app
from objects.engine import ObjectEngine, decode, selected_region

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def engine(): return ObjectEngine(general=False)

@pytest.fixture
def client(tmp_path,engine):
    with TestClient(create_app(tmp_path/'objects.sqlite3',engine)) as c: yield c

def photo(name='coffee',**overrides):
    body=dict(request_id=str(uuid.uuid4()),name=f'{name}.png',data=base64.b64encode((ROOT/f'artifacts/objects/examples/{name}.png').read_bytes()).decode())
    return body|overrides

def decision(**values): return dict(request_id=str(uuid.uuid4()),version=0,status='confirmed',label='咖啡')|values

def test_real_prediction_and_idempotency(client):
    body=photo();a=client.post('/api/recognize',json=body);assert a.status_code==200
    record=a.json();result=record['result'];assert result['candidates'][0]['label']=='espresso'
    assert len(result['candidates'])==5 and all(0<=r['score']<=1 for r in result['candidates'])
    assert len(result['model_sha'])==64 and record['final_label'] is None
    assert client.post('/api/recognize',json=body).json()==record
    assert client.get('/api/records').json()['total']==1
    assert client.post('/api/recognize',json=body|{'name':'different.png'}).status_code==409

def test_review_conflicts_audit_and_original_immutable(client):
    record=client.post('/api/recognize',json=photo()).json();rid=record['id'];url=f'/api/records/{rid}/review'
    body=decision();assert client.post(url,json=body).status_code==200
    assert client.post(url,json=body).status_code==200
    assert client.post(url,json=decision(label='茶')).status_code==409
    assert client.post(url,json=decision(version=1,status='unknown',label=None)).status_code==200
    assert client.post(url,json=decision(version=2,status='pending',label=None)).status_code==200
    final=client.get(f'/api/records/{rid}').json()
    assert final['result']==record['result'] and final['version']==3 and final['final_label'] is None
    assert len(final['reviews'])==3
    assert client.get(f'/api/records/{rid}/image').headers['content-type']=='image/png'

def test_real_crop_and_api_contract(client):
    r=client.post('/api/recognize',json=photo('chelsea',crop=[.1,.1,.8,.8]));assert r.status_code==200
    result=r.json()['result'];assert result['region_size']==[361,240]
    assert result['candidates'][0]['label'] in {'Egyptian cat','tabby','tiger cat'}
    assert len(client.get('/api/categories').json())==1000
    assert client.get('/api/records?limit=0').status_code==422
    assert client.get('/api/records/missing').status_code==404
    assert client.post('/api/recognize',json=photo(crop=[.9,0,.5,1])).status_code==422
    assert client.post('/api/recognize',json=photo(crop=[0,0,.001,.001])).status_code==422

@pytest.mark.parametrize('data',['not base64!','YWJj'])
def test_invalid_files(client,data):
    assert client.post('/api/recognize',json=photo(data=data)).status_code==422

def test_image_normalization_and_blank_rejection():
    # Synthetic images only exercise decoding, never count toward recognition quality.
    out=io.BytesIO();Image.new('RGB',(100,100),'white').save(out,format='PNG')
    with pytest.raises(ValueError,match='纯色'):decode(out.getvalue())
    real=Image.open(ROOT/'artifacts/objects/examples/coffee.png');exif=Image.Exif();exif[274]=6
    out=io.BytesIO();real.save(out,format='JPEG',exif=exif)
    assert decode(out.getvalue()).size==real.size[::-1]
    with pytest.raises(ValueError):selected_region(real,[0,0,0,1])

def test_csv_formula_and_review_validation(client):
    record=client.post('/api/recognize',json=(photo()|{'name':'=danger.png'})).json()
    url=f"/api/records/{record['id']}/review"
    assert client.post(url,json=decision(label=' ')).status_code==422
    assert client.post(url,json=decision(status='unknown')).status_code==422
    assert client.post(url,json=decision(label='=SUM(1,2)')).status_code==200
    rows=list(csv.DictReader(io.StringIO(client.get('/api/export').text.lstrip('\ufeff'))))
    assert rows[0]['name']=="'=danger.png" and rows[0]['final_label']=="'=SUM(1,2)"
    assert rows[0]['model_prediction']=='espresso'

def test_cross_site_writes_rejected(client):
    assert client.post('/api/recognize',json=photo(),headers={'Origin':'https://example.org'}).status_code==403

def test_persistence(tmp_path,engine):
    db=tmp_path/'persist.sqlite3'
    with TestClient(create_app(db,engine)) as c:rid=c.post('/api/recognize',json=photo()).json()['id']
    with TestClient(create_app(db,engine)) as c:assert c.get(f'/api/records/{rid}').json()['name']=='coffee.png'

def test_detector_localizes_real_objects_and_allows_empty(engine):
    for name,expected in [('chelsea','cat'),('coffee','cup'),('astronaut','person')]:
        r=engine.predict(Image.open(ROOT/f'artifacts/objects/examples/{name}.png').convert('RGB'))
        items=r['detection']['items'];assert expected in [d['label'] for d in items]
        for d in items:
            x,y,x2,y2=d['box'];assert 0<=x<x2<=1 and 0<=y<y2<=1
        assert r['detection']['sha256']==engine.detector.sha
    # Falcon 9 is outside COCO's 80 classes. This example must not get a made-up label.
    r=engine.predict(Image.open(ROOT/'artifacts/objects/examples/rocket.png').convert('RGB'))
    assert r['detection']['items']==[]

def test_detection_crop_coordinates_remain_in_original_image(engine):
    image=Image.open(ROOT/'artifacts/objects/examples/chelsea.png').convert('RGB')
    crop=[.1,.1,.8,.8];r=engine.predict(image,crop)
    assert r['detection']['items']
    for d in r['detection']['items']:
        x,y,x2,y2=d['box'];assert .09<=x<x2<=.91 and .09<=y<y2<=.91
