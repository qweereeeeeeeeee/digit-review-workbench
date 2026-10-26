"""Run with python -m src.train. Writes real evidence, never placeholder metrics."""
import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from .data import dataset, split_indices, fingerprint
from .model import DigitResNet
from .metrics import softmax, evaluate, wilson_interval


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/run-001'))
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    images, labels = dataset()
    splits = split_indices(labels)
    write_json(out/'manifest.json', dict(seed=42, dataset_sha256=fingerprint(images, labels),
               source='sklearn.datasets.load_digits / UCI Optical Recognition of Handwritten Digits',
               splits={k:v.tolist() for k,v in splits.items()}, started_utc=started,
               epochs=args.epochs, device='cpu', python=platform.python_version(), torch=torch.__version__))
    # Audit all rows, and store actual pixel data for review.
    frame = pd.DataFrame(images.reshape(-1, 64), columns=[f'pixel_{i}' for i in range(64)])
    frame['label'] = labels
    frame['split'] = ''
    for name, ids in splits.items(): frame.loc[ids, 'split'] = name
    frame.index.name = 'sample_id'
    frame.to_csv(out/'dataset.csv')
    counts = frame.groupby(['split', 'label']).size().unstack(fill_value=0)
    counts.to_csv(out/'class_counts.csv')
    write_json(out/'data_quality.json', dict(samples=len(labels), missing=int(frame.isna().sum().sum()),
               duplicate_images=int(frame.iloc[:, :64].duplicated().sum()), min=float(images.min()), max=float(images.max())))
    x = torch.from_numpy(images[:, None]); y = torch.from_numpy(labels)
    loader = DataLoader(TensorDataset(x[splits['train']], y[splits['train']]), batch_size=64,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    model = DigitResNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = float('inf'); history = []; best_epoch = 0
    for epoch in range(1, args.epochs+1):
        model.train(); loss_sum = 0; correct = 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb); loss = loss_fn(logits, yb)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*len(yb); correct += int((logits.argmax(1)==yb).sum())
        model.eval()
        with torch.inference_mode():
            vl = model(x[splits['validation']]); vy = y[splits['validation']]
            val_loss = loss_fn(vl, vy).item(); val_acc = (vl.argmax(1)==vy).float().mean().item()
        row = dict(epoch=epoch, train_loss=loss_sum/len(splits['train']), train_accuracy=correct/len(splits['train']),
                   validation_loss=val_loss, validation_accuracy=val_acc, lr=optimizer.param_groups[0]['lr'])
        history.append(row); pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        if val_loss < best:
            best = val_loss; best_epoch = epoch
            torch.save(model.state_dict(), out/'model.pt')
        print(json.dumps(row), flush=True)
        scheduler.step()
    model.load_state_dict(torch.load(out/'model.pt', map_location='cpu', weights_only=True)); model.eval()
    with torch.inference_mode(): cal_logits = model(x[splits['calibration']]).numpy()
    # A prespecified bounded temperature grid is optimized ONLY on calibration NLL.
    temperatures = np.geomspace(.25, 4, 241)
    cal_y = labels[splits['calibration']]
    losses = [-np.log(np.maximum(softmax(cal_logits, t)[np.arange(len(cal_y)), cal_y], 1e-15)).mean() for t in temperatures]
    temperature = float(temperatures[np.argmin(losses)])
    cal_probs = softmax(cal_logits, temperature)
    threshold = float(np.quantile(cal_probs.max(1), .10))
    write_json(out/'calibration.json', dict(temperature=temperature, threshold=threshold, target_coverage=.90,
                fitted_on='calibration', temperature_grid=[.25,4,241], calibration_nll_before=float(losses[120]),
                calibration_nll_after=float(min(losses))))
    test_ids = splits['test']; test_x = x[test_ids]; test_y = labels[test_ids]
    with torch.inference_mode(): logits = model(test_x).numpy()
    probs = softmax(logits, temperature)
    raw = evaluate(softmax(logits), test_y, threshold)
    calibrated = evaluate(probs, test_y, threshold)
    calibrated['accuracy_95pct_wilson'] = wilson_interval(int((probs.argmax(1)==test_y).sum()), len(test_y))
    baseline = LogisticRegression(max_iter=2000, C=1, random_state=42)
    baseline.fit(images[splits['train']].reshape(-1,64), labels[splits['train']])
    baseline_probs = baseline.predict_proba(images[test_ids].reshape(-1,64))
    rng = np.random.default_rng(43)
    corrupted = dict(noise_sigma_0_25=np.clip(images[test_ids]+rng.normal(0,.25,images[test_ids].shape),0,1).astype(np.float32),
                     brightness_0_5=(images[test_ids]*.5).copy(), center_occlusion=images[test_ids].copy())
    corrupted['center_occlusion'][:,3:5,3:5] = 0
    robustness = {}
    for name, arr in corrupted.items():
        with torch.inference_mode(): cp = softmax(model(torch.from_numpy(arr[:,None])).numpy(), temperature)
        robustness[name] = evaluate(cp, test_y, threshold)
    pred = probs.argmax(1)
    predictions = pd.DataFrame(dict(sample_id=test_ids, true_label=test_y, prediction=pred,
                confidence=probs.max(1), accepted=probs.max(1)>=threshold, correct=pred==test_y))
    predictions.to_csv(out/'predictions.csv', index=False)
    predictions[~predictions.correct].to_csv(out/'errors.csv', index=False)
    report = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  duration_seconds=time.perf_counter()-start, best_epoch=best_epoch,
                  parameter_count=sum(p.numel() for p in model.parameters()), splits={k:len(v) for k,v in splits.items()},
                  test_raw=raw, test_calibrated=calibrated, baseline_logistic=evaluate(baseline_probs,test_y,0),
                  robustness=robustness, model_sha256=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),
                  temperature=temperature, threshold=threshold)
    write_json(out/'metrics.json', report)
    # Only held-out test examples are exposed in the gallery, visibly labelled.
    write_json(out/'examples.json', [dict(sample_id=int(i), label=int(labels[i]), pixels=images[i].tolist()) for i in test_ids])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':150, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    h = pd.DataFrame(history)
    axes[0].plot(h.epoch,h.train_loss,label='Train'); axes[0].plot(h.epoch,h.validation_loss,label='Validation')
    axes[0].set(title='Learning curves (no test-set tuning)',xlabel='Epoch',ylabel='Cross entropy'); axes[0].legend()
    axes[1].plot(h.epoch,h.train_accuracy,label='Train'); axes[1].plot(h.epoch,h.validation_accuracy,label='Validation')
    axes[1].set(title='Accuracy',xlabel='Epoch',ylim=(0,1.02)); axes[1].legend(); fig.tight_layout(); fig.savefig(out/'training.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,5)); cm=confusion_matrix(test_y,pred,labels=list(range(10)))
    ax.imshow(cm,cmap='Blues'); ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted',ylabel='True',title='Held-out test confusion matrix')
    for i in range(10):
        for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#203050',fontsize=8)
    fig.tight_layout(); fig.savefig(out/'confusion.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4)); counts.T.plot.bar(stacked=True,ax=ax,colormap='viridis')
    ax.set(title='Actual class distribution by split',ylabel='Images',xlabel='Digit'); fig.tight_layout(); fig.savefig(out/'distribution.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,p in [('Raw',softmax(logits)),('Temperature scaled',probs)]:
        conf=p.max(1); ok=p.argmax(1)==test_y; xs=[]; ys=[]
        for lo in np.arange(0,1,.1):
            mask=(conf>lo)&(conf<=lo+.1)
            if mask.any(): xs.append(conf[mask].mean()); ys.append(ok[mask].mean())
        axes[0].plot(xs,ys,'o-',label=name)
    axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].set(title='Test reliability (10 bins)',xlabel='Confidence',ylabel='Observed accuracy'); axes[0].legend()
    rows=[('Clean',calibrated)]+list(robustness.items()); xx=np.arange(len(rows))
    axes[1].bar(xx-.18,[r['accuracy'] for _,r in rows],.36,label='Accuracy'); axes[1].bar(xx+.18,[r['coverage'] for _,r in rows],.36,label='Coverage')
    axes[1].set(xticks=xx,xticklabels=['Clean','Noise','Dim','Occlusion'],ylim=(0,1.05),title='Fixed corruption stress test'); axes[1].legend()
    fig.tight_layout(); fig.savefig(out/'reliability.png'); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__': main()
