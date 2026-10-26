"""Independently recompute stored test metrics from checkpoint and split manifest."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import dataset, fingerprint
from src.model import DigitResNet
from src.metrics import softmax, evaluate

run=Path('artifacts/run-001')
r=json.loads((run/'metrics.json').read_text())
m=json.loads((run/'manifest.json').read_text())
x,y=dataset(); torch.set_num_threads(2)
assert fingerprint(x,y)==m['dataset_sha256']
assert hashlib.sha256((run/'model.pt').read_bytes()).hexdigest()==r['model_sha256']
model=DigitResNet(); model.load_state_dict(torch.load(run/'model.pt',weights_only=True,map_location='cpu')); model.eval()
ids=m['splits']['test']
with torch.inference_mode():p=softmax(model(torch.from_numpy(x[ids,None])).numpy(),r['temperature'])
actual=evaluate(p,y[ids],r['threshold'])
for k,v in actual.items():
    assert np.isclose(v,r['test_calibrated'][k]),(k,v,r['test_calibrated'][k])
csv=pd.read_csv(run/'predictions.csv')
assert np.array_equal(csv.sample_id,ids)
assert np.array_equal(csv.prediction,p.argmax(1))
assert np.allclose(csv.confidence,p.max(1))
print(json.dumps({'status':'verified','test_samples':len(ids),'correct':int((p.argmax(1)==y[ids]).sum()),'metrics_recomputed':actual,'model_sha256':r['model_sha256']},indent=2))
