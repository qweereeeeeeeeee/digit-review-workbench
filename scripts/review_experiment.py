"""Retrospective error-discovery experiment. No simulated human correctness claims."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import dataset
from src.metrics import softmax
from workbench.inference import Engine


def main():
    root=Path(__file__).resolve().parents[1];out=root/'artifacts/review';out.mkdir(exist_ok=True)
    engine=Engine(); manifest=json.loads((root/'artifacts/run-001/manifest.json').read_text())
    images,labels=dataset();ids=np.asarray(manifest['splits']['test']);x=images[ids];y=labels[ids]
    rng=np.random.default_rng(43)
    scenes={'clean':x,'noise_025':np.clip(x+rng.normal(0,.25,x.shape),0,1).astype(np.float32),'brightness_05':x*.5}
    result={'method':'retrospective analysis; error discovery, not human correction','random_trials':1000,'random_seed':2026,'budgets':[5,10,20,40,80],'scenes':{}}
    curves=[];predictions=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for ax,(name,arr) in zip(axes,scenes.items()):
        with torch.inference_mode():p=softmax(engine.model(torch.from_numpy(arr[:,None])).numpy(),engine.report['temperature'])
        errors=p.argmax(1)!=y
        order=np.lexsort((ids,p.max(1)))
        prioritized=np.cumsum(errors[order])
        generator=np.random.default_rng(2026)
        random=np.asarray([np.cumsum(errors[generator.permutation(len(y))]) for _ in range(1000)])
        mean=random.mean(0);low,high=np.quantile(random,[.05,.95],axis=0)
        rows=[]
        for budget in result['budgets']:
            rows.append(dict(budget=budget,priority_errors=int(prioritized[budget-1]),random_mean=float(mean[budget-1]),random_p05=float(low[budget-1]),random_p95=float(high[budget-1])))
        result['scenes'][name]={'samples':len(y),'errors':int(errors.sum()),'rows':rows}
        for k in range(len(y)):
            curves.append(dict(scene=name,budget=k+1,priority=int(prioritized[k]),random_mean=mean[k],random_p05=low[k],random_p95=high[k]))
        for i in range(len(y)):
            predictions.append(dict(scene=name,sample_id=int(ids[i]),label=int(y[i]),prediction=int(p[i].argmax()),confidence=float(p[i].max())))
        budgets=np.arange(1,len(y)+1)
        ax.plot(budgets,prioritized,label='Lowest confidence first',color='#215e9b')
        ax.plot(budgets,mean,label='Random mean',color='#8c949c');ax.fill_between(budgets,low,high,color='#bbc2c9',alpha=.3,label='Random 5%-95%')
        ax.set(title=f'{name}: {int(errors.sum())} errors / {len(y)}',xlabel='Images reviewed',ylabel='Errors encountered')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'queue_experiment.png',dpi=160);plt.close(fig)
    (out/'queue_experiment.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(curves).to_csv(out/'queue_curves.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'queue_predictions.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
