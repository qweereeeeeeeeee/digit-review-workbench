from fastapi.testclient import TestClient
import pytest
from src.api import app


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c: yield c


def test_health_and_actual_prediction(client):
    assert client.get('/health').json()['status']=='ok'
    ex=client.get('/api/examples').json()[0]
    r=client.post('/api/predict',json={'pixels':ex['pixels'],'explain':True})
    assert r.status_code==200
    data=r.json()
    assert len(data['probabilities'])==10
    assert abs(sum(data['probabilities'])-1)<1e-5
    assert len(data['occlusion'])==8
    assert data['model_sha256']==client.get('/api/report').json()['model_sha256']
    assert data['accepted']==(data['confidence']>=data['threshold'])


def test_blank_input_and_extras(client):
    assert client.post('/api/predict',json={'pixels':[[0]*8]*8}).status_code==422
    assert client.post('/api/predict',json={'pixels':[[0]*8]*8,'unknown':1}).status_code==422


def test_request_limit(client):
    assert client.post('/api/predict',content=b' '*17000).status_code==413


def test_site_and_whitelist(client):
    assert client.get('/').status_code==200
    assert client.get('/figures/training.png').status_code==200
    assert client.get('/figures/model.pt').status_code==404
