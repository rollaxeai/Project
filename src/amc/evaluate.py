"""Метрики качества классификации и их доверительные интервалы."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def summary(y_true: np.ndarray, y_pred: np.ndarray, snr: np.ndarray, classes: list[str],
            exclude: str | None = "AM-SSB") -> dict[str, float]:
    """Сводные метрики: accuracy (вся, ОСШ ≥ 0, ≥ 10), macro-F1, accuracy без класса exclude."""
    y_true, y_pred, snr = np.asarray(y_true), np.asarray(y_pred), np.asarray(snr)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "acc_snr>=0": accuracy_score(y_true[snr >= 0], y_pred[snr >= 0]),
        "acc_snr>=10": accuracy_score(y_true[snr >= 10], y_pred[snr >= 10]),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    if exclude in classes:
        m = y_true != classes.index(exclude)
        out[f"acc_без_{exclude}"] = accuracy_score(y_true[m], y_pred[m])
    return out


def acc_by_snr(y_true: np.ndarray, y_pred: np.ndarray, snr: np.ndarray, bin_db: float = 2.0) -> pd.Series:
    """Accuracy по интервалам ОСШ (для непрерывного ОСШ — округление вниз к сетке bin_db)."""
    b = np.floor(np.asarray(snr) / bin_db) * bin_db
    return pd.Series(np.asarray(y_true) == np.asarray(y_pred)).groupby(b).mean()


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Бутстреп-доверительный интервал для доли верных ответов."""
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, dtype=float)
    means = np.array([c[rng.integers(0, len(c), len(c))].mean() for _ in range(n_boot)])
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
