"""Проверки генератора: метки должны соответствовать реальному сигналу."""

from dataclasses import replace

import numpy as np
import pytest

from amc.generator import CLASSES, LINEAR, GenConfig, constellation, generate_one

IDEAL = GenConfig(n=4096, snr_db=(30.0, 30.0), p_multipath=0.0, phase_noise=(0.0, 0.0), sro_ppm=0.0)


@pytest.mark.parametrize("cls", CLASSES)
def test_shape_power_finite(cls):
    z, m = generate_one(cls, np.random.default_rng(1))
    assert z.shape == (1024,) and z.dtype == np.complex64
    assert np.all(np.isfinite(z))
    assert abs(np.mean(np.abs(z) ** 2) - 1) < 1e-3
    assert m["cls"] == cls and m["label"] == CLASSES.index(cls)


def test_deterministic():
    a, _ = generate_one("QAM16", np.random.default_rng(7))
    b, _ = generate_one("QAM16", np.random.default_rng(7))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("cls", ["BPSK", "QAM64", "GFSK", "WBFM", "AM-SSB"])
def test_snr_label(cls):
    """Мощность шума соответствует заданному ОСШ."""
    for seed in range(5):
        z, m, clean = generate_one(cls, np.random.default_rng(seed), return_clean=True)
        k = np.sqrt(1 + 10 ** (-m["snr_db"] / 10))
        noise = z * k - clean
        snr_meas = 10 * np.log10(1 / np.mean(np.abs(noise) ** 2))
        assert abs(snr_meas - m["snr_db"]) < 0.5


def test_constellation_unit_power():
    for c in LINEAR:
        assert abs(np.mean(np.abs(constellation(c)) ** 2) - 1) < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_cfo_label_qpsk(seed):
    """Пик спектра z⁴ у QPSK находится на частоте 4·cfo."""
    z, m = generate_one("QPSK", np.random.default_rng(seed), IDEAL)
    n = 1 << 16
    s = np.abs(np.fft.fft(z.astype(complex) ** 4, n))
    f = np.fft.fftfreq(n)[np.argmax(s)]
    assert abs(f / 4 - m["cfo"]) < 2e-4


@pytest.mark.parametrize("cls", sorted(LINEAR))
def test_bandwidth_linear(cls):
    """Полоса 99 % у линейных модуляций ≈ (1 + roll-off) / sps."""
    for seed in range(3):
        _, m = generate_one(cls, np.random.default_rng(seed), IDEAL)
        theory = (1 + m["rolloff"]) / m["sps"]
        assert 0.75 * theory < m["bw99"] < 1.1 * theory


def test_ssb_one_sided():
    """AM-SSB: почти вся мощность по одну сторону от несущей (в отличие от RadioML, где это шум)."""
    cfg = replace(IDEAL, cfo=0.0)
    for seed in range(4):
        _, m, clean = generate_one("AM-SSB", np.random.default_rng(seed), cfg, return_clean=True)
        s = np.abs(np.fft.fft(clean)) ** 2
        f = np.fft.fftfreq(len(clean))
        pos = s[f > 0].sum() / s.sum()
        assert (pos > 0.95) if m["sideband"] == "USB" else (pos < 0.05)


def test_fsk_shift():
    """CPFSK: два тона разнесены на h/sps."""
    cfg = replace(IDEAL, cfo=0.0, n=8192)
    _, m, clean = generate_one("CPFSK", np.random.default_rng(3), cfg, return_clean=True)
    inst_f = np.diff(np.unwrap(np.angle(clean))) / (2 * np.pi)
    hi, lo = np.percentile(inst_f, 90), np.percentile(inst_f, 10)
    assert abs((hi - lo) - m["fsk_shift"]) < 0.15 * m["fsk_shift"]
