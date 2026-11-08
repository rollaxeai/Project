"""Модели: форма выхода при разных длинах окна и входных представлениях."""

import pytest
import torch

from amc.augment import IN_CHANNELS, augment, normalize_power, represent
from amc.models import build


@pytest.mark.parametrize("name", ["vtcnn2", "resnet", "cnnlstm"])
@pytest.mark.parametrize("kind", ["iq", "ap", "iqap"])
def test_forward_shape(name, kind):
    n = 128
    x = normalize_power(torch.randn(4, 2, n))
    m = build(name, IN_CHANNELS[kind], 11, n).eval()
    assert m(represent(x, kind)).shape == (4, 11)
    assert m.encode(represent(x, kind)).shape == (4, m.emb_dim)


def test_resnet_any_length():
    m = build("resnet", 2, 11, 0).eval()
    for n in (128, 1024, 2048):
        assert m(torch.randn(2, 2, n)).shape == (2, 11)


def test_augment_keeps_power():
    x = normalize_power(torch.randn(8, 2, 256))
    y = augment(x, torch.Generator().manual_seed(0))
    assert torch.allclose(x.pow(2).sum(1).mean(1), y.pow(2).sum(1).mean(1), atol=1e-4)
