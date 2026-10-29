"""Recompute delivery evidence without overwriting training or experiment outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import torch
from src.data import dataset, fingerprint, split_indices
from src.metrics import softmax, evaluate
from workbench.inference import Engine


def main():
    root=Path(__file__).resolve().parents[1]
    run=root/'artifacts/run-001';out=root/'artifacts/review'
    manifest=json.loads((run/'manifest.json').read_text())
    report=json.loads((run/'metrics.json').read_text())
    x,y=dataset();checks=[]
    def passed(name):checks.append(name)
    assert fingerprint(x,y)==manifest['dataset_sha256']
    table=pd.read_csv(run/'dataset.csv').sort_values('sample_id')
    np.testing.assert_array_equal(table.sample_id,np.arange(len(y)))
    np.testing.assert_array_equal(table.label,y)
    np.testing.assert_array_equal(table[[f'pixel_{i}' for i in range(64)]],x.reshape(-1,64))
    assert np.unique(x.reshape(-1,64),axis=0).shape[0]==len(y)
    expected=split_indices(y,manifest['seed'])
    combined=[]
    for name,ids in manifest['splits'].items():
        np.testing.assert_array_equal(ids,expected[name])
        assert len(ids)==report['splits'][name]
        assert (table.iloc[ids]['split']==name).all()
        combined.extend(ids)
    assert len(set(combined))==len(combined)==len(y)
    passed('1797 images and labels match installed public dataset; no duplicate images; four disjoint reproducible splits cover all samples')
    engine=Engine()
    assert sum(p.numel() for p in engine.model.parameters())==report['parameter_count']
    ids=np.array(manifest['splits']['test']);truth=y[ids]
    for e,i in zip(engine.examples,ids):
        assert e['sample_id']==i and e['label']==y[i]
        np.testing.assert_array_equal(e['pixels'],x[i])
    assert len(engine.examples)==len(ids)
    def predict(arr):
        with torch.inference_mode():return softmax(engine.model(torch.from_numpy(arr[:,None])).numpy(),report['temperature'])
    p=predict(x[ids]);pred=p.argmax(1);confidence=p.max(1)
    stored=pd.read_csv(run/'predictions.csv')
    for col,actual in [('sample_id',ids),('true_label',truth),('prediction',pred),('accepted',confidence>=report['threshold']),('correct',pred==truth)]:
        np.testing.assert_array_equal(stored[col],actual)
    np.testing.assert_allclose(stored.confidence,confidence,rtol=1e-6,atol=1e-7)
    for key,value in evaluate(p,truth,report['threshold']).items():
        np.testing.assert_allclose(value,report['test_calibrated'][key],rtol=1e-6,atol=1e-7)
    errors=pd.read_csv(run/'errors.csv')
    pd.testing.assert_frame_equal(errors.reset_index(drop=True),stored[~stored.correct].reset_index(drop=True))
    passed('checkpoint fingerprint and parameter count match; all 270 predictions, labels, examples, acceptance flags and three errors recomputed')
    summary=json.loads((out/'queue_experiment.json').read_text())
    saved_predictions=pd.read_csv(out/'queue_predictions.csv');curves=pd.read_csv(out/'queue_curves.csv')
    rng=np.random.default_rng(43)
    scenes={'clean':x[ids],'noise_025':np.clip(x[ids]+rng.normal(0,.25,x[ids].shape),0,1).astype(np.float32),'brightness_05':x[ids]*.5}
    assert len(saved_predictions)==len(curves)==810
    budgets20={}
    for name,arr in scenes.items():
        prob=predict(arr);mistakes=prob.argmax(1)!=truth
        saved=saved_predictions[saved_predictions.scene==name]
        np.testing.assert_array_equal(saved.sample_id,ids)
        np.testing.assert_array_equal(saved.label,truth)
        np.testing.assert_array_equal(saved.prediction,prob.argmax(1))
        np.testing.assert_allclose(saved.confidence,prob.max(1),rtol=1e-6,atol=1e-7)
        priority=np.cumsum(mistakes[np.lexsort((ids,prob.max(1)))])
        random=np.random.default_rng(summary['random_seed'])
        trials=np.array([np.cumsum(mistakes[random.permutation(len(ids))]) for _ in range(summary['random_trials'])])
        low,high=np.quantile(trials,[.05,.95],axis=0);mean=trials.mean(0)
        curve=curves[curves.scene==name]
        np.testing.assert_array_equal(curve.budget,np.arange(1,271))
        for col,values in [('priority',priority),('random_mean',mean),('random_p05',low),('random_p95',high)]:
            np.testing.assert_allclose(curve[col],values,rtol=1e-10,atol=1e-10)
        scene=summary['scenes'][name]
        assert scene['samples']==270 and scene['errors']==int(mistakes.sum())
        for row in scene['rows']:
            i=row['budget']-1
            for key,value in [('priority_errors',priority[i]),('random_mean',mean[i]),('random_p05',low[i]),('random_p95',high[i])]:
                np.testing.assert_allclose(row[key],value,rtol=1e-10,atol=1e-10)
        budgets20[name]={'total_errors':int(mistakes.sum()),'priority_errors':int(priority[19]),'random_mean':float(mean[19])}
    passed('all 810 scene predictions and full queue curves match fresh inference and 1000 randomized permutations per scene')
    export=pd.read_csv(out/'demo/演示导出.csv',encoding='utf-8-sig')
    assert len(export)==270 and export.name.is_unique
    expected_map={f'test-{i}.png':int(v) for i,v in zip(ids,pred)}
    assert all(expected_map[row['name']]==row['prediction'] for row in export.to_dict('records'))
    reviewed=export[export.final_label.notna()]
    assert len(reviewed)==1
    row=reviewed.iloc[0]
    assert row['name']=='test-555.png' and row.prediction==3 and row.final_label==y[555]==9 and row.version==1
    passed('recorded CSV contains 270 unchanged model predictions and exactly one explicitly demonstrated correction, sample 555: 3 to 9')
    # Read existing user data without modifying it. The database is intentionally not published.
    db_path=root/'var/review.sqlite3'
    database=None
    if db_path.exists():
        db=sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)
        try:
            assert db.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
            assert db.execute('PRAGMA foreign_key_check').fetchall()==[]
            database={'batches':db.execute('SELECT count(*) FROM batches').fetchone()[0],
                      'items':db.execute('SELECT count(*) FROM items').fetchone()[0],
                      'events':db.execute('SELECT count(*) FROM events').fetchone()[0]}
        finally:db.close()
        passed('local SQLite integrity and foreign keys checked read-only')
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','checks':checks,
            'samples':len(y),'splits':report['splits'],'test_correct':int((pred==truth).sum()),'test_total':len(ids),
            'error_sample_ids':ids[pred!=truth].tolist(),'budget_20':budgets20,'local_database':database,
            'model_sha256':engine.sha,'limits':'This checks consistency and reproducibility against the installed public dataset, not new handwriting performance or human review efficiency.'}
    (out/'final-data-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
