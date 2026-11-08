"""Классические оценщики на сигналах с известными параметрами (высокое ОСШ, без многолучёвости)."""

from dataclasses import replace

import numpy as np
import pytest

from amc import estimators as est
from amc.generator import GenConfig, generate_one

CFG = GenConfig(n=4096, snr_db=(25.0, 25.0), p_multipath=0.0, phase_noise=(0.0, 0.0), sro_ppm=0.0)


def batch(cls: str, cfg: GenConfig = CFG, k: int = 8):
    zs, ms = zip(*[generate_one(cls, np.random.default_rng(i), cfg) for i in range(k)], strict=True)
    return np.stack(zs).astype(complex), ms


@pytest.mark.parametrize("cls", ["BPSK", "QPSK", "8PSK", "QAM16"])
def test_cfo_xm(cls):
    z, ms = batch(cls)
    f = est.cfo_xm(z, est.M_POWER[cls])
    true = np.array([m["cfo"] for m in ms])
    assert np.median(np.abs(f - true)) < 1e-4


@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_snr_m2m4_true_ka(snr):
    """M2M4 точен, если ka известен точно (по чистому сигналу)."""
    zs, kas = [], []
    for i in range(8):
        z, _, s = generate_one("QPSK", np.random.default_rng(i), replace(CFG, snr_db=(snr, snr)), return_clean=True)
        s = s.astype(complex)
        zs.append(z)
        kas.append(np.mean(np.abs(s) ** 4) / np.mean(np.abs(s) ** 2) ** 2)
    assert abs(np.median(est.snr_m2m4(np.stack(zs).astype(complex), np.array(kas))) - snr) < 1.0


def test_m2m4_sensitive_to_ka_at_high_snr():
    """Известное ограничение: при ОСШ 20 дБ ошибка ka ≈ 0,05 даёт ошибку оценки > 10 дБ (см. tools/m2m4_check.py)."""
    z, _ = batch("QPSK", replace(CFG, snr_db=(20.0, 20.0)))
    assert abs(np.median(est.snr_m2m4(z, 1.265)) - 20) > 10


@pytest.mark.parametrize("snr", [0.0, 10.0, 20.0])
def test_snr_spectral_psk(snr):
    """Спектральный метод: ошибка ≤ 1,5 дБ (при 0 дБ смещение ≈ −1 дБ: сигнал занимает часть «полки»)."""
    z, _ = batch("QPSK", replace(CFG, snr_db=(snr, snr)))
    assert abs(np.median(est.snr_spectral(z)) - snr) < 1.5


def test_snr_spectral_analog():
    z, _ = batch("AM-SSB", replace(CFG, snr_db=(10.0, 10.0)))
    assert abs(np.median(est.snr_spectral(z)) - 10) < 2.0


@pytest.mark.parametrize("cls", ["BPSK", "QPSK", "QAM16"])
def test_rate_envelope(cls):
    z, ms = batch(cls)
    r = est.rate_envelope(z)
    true = np.array([m["symbol_rate"] for m in ms])
    assert np.median(np.abs(r / true - 1)) < 0.02


def test_rate_freqdiff_cpfsk():
    z, ms = batch("CPFSK")
    r = est.rate_freqdiff(z)
    true = np.array([m["symbol_rate"] for m in ms])
    assert np.median(np.abs(r / true - 1)) < 0.05


def test_bandwidth_linear():
    z, ms = batch("QPSK")
    bw = est.bandwidth99(z)
    true = np.array([m["bw99"] for m in ms])
    assert np.median(np.abs(bw / true - 1)) < 0.15
