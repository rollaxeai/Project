"""Тесты продукта: перенос на ноль/передискретизация и сквозной анализ WAV (если есть ONNX-модель)."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal as sps

from amc import infer
from amc.generator import GenConfig, generate_one


def _wav_signal(cls: str, seed: int = 0):
    cfg = replace(GenConfig(), n=8192, p_multipath=0.0, cfo=0.0, sps=(8.0, 8.0), snr_db=(30.0, 30.0))
    _, m, x = generate_one(cls, np.random.default_rng(seed), cfg, return_clean=True)
    y = sps.resample_poly(x.astype(np.complex128), 4, 1) * np.exp(2j * np.pi * 0.2 * np.arange(4 * len(x)))
    y += 0.01 * (np.random.default_rng(1).standard_normal(len(y)) + 1j * np.random.default_rng(2).standard_normal(len(y)))
    return y, m


def test_to_baseband_center_and_rate():
    y, m = _wav_signal("QPSK")
    z, fs2, center, bw = infer.to_baseband(y, 48000.0)
    assert abs(center - 0.2 * 48000) < 100              # несущая 9600 Гц
    assert abs(bw - m["bw99"] * 12000) / (m["bw99"] * 12000) < 0.15
    assert 0.1 < bw / fs2 < 0.25                         # полоса ≈ 15 % новой fs
    assert len(z) == pytest.approx(len(y) * fs2 / 48000, rel=0.01)


def test_real_signal_same_center():
    y, _ = _wav_signal("BPSK")
    _, _, c_iq, _ = infer.to_baseband(y, 48000.0)
    _, _, c_re, _ = infer.to_baseband(y.real, 48000.0)
    assert abs(c_iq - c_re) < 50


def test_windows_shape_and_power():
    w = infer.windows(np.exp(1j * np.arange(5000) * 0.1) * 3)
    assert w.shape[1:] == (2, 1024) and w.dtype == np.float32
    assert np.allclose((w ** 2).sum(1).mean(1), 1, atol=1e-3)
    assert infer.windows(np.ones(100, complex)).shape == (1, 2, 1024)


@pytest.mark.skipif(not infer.DEFAULT_MODEL.exists(), reason="нет models/onnx/amc_multitask.onnx")
def test_analyze_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")
    y, m = _wav_signal("BPSK")
    y /= np.abs(y).max() * 1.1
    f = tmp_path / "bpsk.wav"
    sf.write(f, np.stack([y.real, y.imag], 1).astype(np.float32), 48000, subtype="FLOAT")
    r = infer.analyze(f)
    assert r["modulation"] == "BPSK"
    assert abs(r["symbol_rate_Bd"] - 1500) / 1500 < 0.05   # 12000 / 8
    assert abs(r["carrier_Hz"] - 9600) < 50


def test_real_signal_noisy_band_not_full():
    """Регрессия: вещественный сигнал с сильным белым шумом — полоса не должна «растекаться» на всю fs/2."""
    y, m = _wav_signal("QPSK")
    rng = np.random.default_rng(3)
    y = y + 0.05 * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    _, fs2, _, bw = infer.to_baseband(y.real, 48000.0)
    assert fs2 < 24000 and bw < 3 * m["bw99"] * 12000
