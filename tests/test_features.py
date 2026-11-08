"""Проверка признаков на сигналах с известными теоретическими значениями."""

import numpy as np
import pytest

from amc.features import FEATURES, compute_features
from amc.generator import GenConfig, constellation, generate_one

RNG = np.random.default_rng(0)


def symbols(name: str, n: int = 4096, k: int = 50) -> np.ndarray:
    c = constellation(name)
    z = c[RNG.integers(0, len(c), (k, n))]
    return z * np.exp(1j * RNG.uniform(0, 2 * np.pi, (k, 1)))  # случайная фаза — признаки к ней инвариантны


@pytest.mark.parametrize("name,c42", [("BPSK", -2.0), ("QPSK", -1.0), ("8PSK", -1.0), ("QAM16", -0.68),
                                      ("QAM64", -0.619)])
def test_c42_theory(name, c42):
    f = compute_features(symbols(name))
    assert abs(f["C42"].mean() - c42) < 0.03


def test_c40_separates_8psk():
    """|C40|: QPSK = 1, 8PSK = 0."""
    assert abs(compute_features(symbols("QPSK"))["C40"].mean() - 1) < 0.03
    assert compute_features(symbols("8PSK"))["C40"].mean() < 0.05


def test_noise_cumulants_zero():
    z = (RNG.standard_normal((50, 4096)) + 1j * RNG.standard_normal((50, 4096))) / np.sqrt(2)
    f = compute_features(z)
    assert abs(f["C42"].mean()) < 0.05 and f["C40"].mean() < 0.1


def test_constant_envelope_small_gmax():
    cfg = GenConfig(snr_db=(40.0, 40.0), p_multipath=0.0)
    fsk = np.stack([generate_one("CPFSK", np.random.default_rng(i), cfg)[0] for i in range(20)])
    am = np.stack([generate_one("AM-DSB", np.random.default_rng(i), cfg)[0] for i in range(20)])
    assert compute_features(fsk)["s_aa"].median() < compute_features(am)["s_aa"].median()


def test_x4_line_for_qpsk_not_for_8psk():
    assert compute_features(symbols("QPSK"))["x4_peak"].mean() > compute_features(symbols("8PSK"))["x4_peak"].mean() + 1


def test_shape_and_finite():
    z, _ = generate_one("QAM16", np.random.default_rng(1))
    f = compute_features(np.stack([z, z]))
    assert list(f.columns) == FEATURES and f.shape == (2, len(FEATURES))
    assert np.isfinite(f.to_numpy()).all()


def test_phase_invariance():
    z, _ = generate_one("8PSK", np.random.default_rng(2))
    a = compute_features(z[None]).to_numpy()
    b = compute_features((z * np.exp(1j * 1.234))[None]).to_numpy()
    assert np.allclose(a, b, rtol=1e-3, atol=1e-4)
