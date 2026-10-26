import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def softmax(logits, temperature=1.0):
    z = logits / temperature
    exp = np.exp(z - z.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def ece(probs, labels, bins=10):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    total = 0.0
    for i in range(bins):
        mask = (confidence > i/bins) & (confidence <= (i+1)/bins)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def evaluate(probs, labels, threshold):
    pred = probs.argmax(axis=1)
    accepted = probs.max(axis=1) >= threshold
    # Null, not 0 or 100%, when no examples are accepted.
    return dict(n=len(labels), accuracy=float(accuracy_score(labels, pred)),
                macro_f1=float(f1_score(labels, pred, average='macro', zero_division=0)),
                nll=float(log_loss(labels, probs, labels=list(range(10)))),
                ece=ece(probs, labels), coverage=float(accepted.mean()),
                accepted_count=int(accepted.sum()),
                accepted_accuracy=float((pred[accepted] == labels[accepted]).mean()) if accepted.any() else None)


def wilson_interval(correct, n):
    z = 1.96
    p = correct/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [float(center-radius), float(center+radius)]
