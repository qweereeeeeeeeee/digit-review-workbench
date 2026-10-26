import numpy as np
import pytest
import torch
from src.model import ResidualBlock, DigitResNet
from src.data import dataset, split_indices
from src.metrics import softmax, evaluate
from src.api import PredictionRequest
from pydantic import ValidationError


def test_residual_projection_and_gradient():
    for cin, cout, stride in [(16,16,1),(16,32,2),(32,64,2)]:
        x=torch.randn(3,cin,8,8,requires_grad=True)
        y=ResidualBlock(cin,cout,stride)(x)
        assert y.shape == (3,cout,8//stride,8//stride)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


def test_full_model_backward():
    m=DigitResNet(); x=torch.randn(4,1,8,8)
    logits=m(x)
    assert logits.shape==(4,10)
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_split_integrity_and_duplicates():
    x,y=dataset(); splits=split_indices(y)
    sets=[set(v) for v in splits.values()]
    assert len(set.union(*sets))==len(y)
    assert sum(map(len,sets))==len(y)
    for a in range(len(sets)):
        for b in range(a): assert not sets[a]&sets[b]
    assert len(np.unique(x.reshape(len(x),-1),axis=0))==len(x)
    for ids in splits.values(): assert set(y[ids])==set(range(10))


def test_temperature_preserves_argmax_and_normalization():
    logits=np.array([[2.,-4,5,0,1,3,2,8,-1,1]])
    for t in [.25,1,4]:
        p=softmax(logits,t)
        assert p.argmax()==logits.argmax()
        assert np.allclose(p.sum(1),1)


def test_no_accepted_examples_is_null():
    p=np.full((2,10),.1)
    assert evaluate(p,np.array([0,1]),.9)['accepted_accuracy'] is None


@pytest.mark.parametrize('bad',[[],[[0]*7]*8,[[2]*8]*8,[[float('nan')]*8]*8,[[True]*8]*8, [['0']*8]*8])
def test_invalid_input(bad):
    with pytest.raises(ValidationError): PredictionRequest(pixels=bad)
