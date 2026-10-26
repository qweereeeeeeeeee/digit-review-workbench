import hashlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def dataset():
    data = load_digits()
    images = data.images.astype(np.float32) / 16.0
    labels = data.target.astype(np.int64)
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError('Dataset failed finite/range validation')
    return images, labels


def split_indices(labels, seed=42):
    ids = np.arange(len(labels))
    train, rest = train_test_split(ids, test_size=.45, random_state=seed, stratify=labels)
    val, rest = train_test_split(rest, test_size=2/3, random_state=seed, stratify=labels[rest])
    cal, test = train_test_split(rest, test_size=.5, random_state=seed, stratify=labels[rest])
    return dict(train=train, validation=val, calibration=cal, test=test)


def fingerprint(images, labels):
    return hashlib.sha256(images.tobytes() + labels.tobytes()).hexdigest()
