"""Загрузка датасетов проекта и фиксированные разбиения train/val/test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amc.radioml import load_npz

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def rml_split(y: np.ndarray, snr: np.ndarray, seed: int = 42, frac: tuple[float, float] = (0.6, 0.2)) -> np.ndarray:
    """Стратифицированное по (класс × ОСШ) разбиение 60/20/20. Возвращает массив строк 'train'/'val'/'test'.

    EDA (01_eda, п. 5) показал, что окна RadioML не перекрываются, поэтому случайное разбиение без утечки.
    """
    rng = np.random.default_rng(seed)
    split = np.empty(len(y), dtype=object)
    for c in np.unique(y):
        for s in np.unique(snr):
            idx = rng.permutation(np.flatnonzero((y == c) & (snr == s)))
            a = int(round(frac[0] * len(idx)))
            b = a + int(round(frac[1] * len(idx)))
            split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    return split.astype(str)


def load_rml2016a() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """RadioML 2016.10a: X[N, 2, 128] и meta (label, cls, snr_db, split)."""
    X, y, snr, classes = load_npz(DATA / "radioml" / "rml2016a.npz")
    meta = pd.DataFrame({"label": y, "cls": [classes[i] for i in y], "snr_db": snr.astype(float),
                         "split": rml_split(y, snr)})
    return X, meta, classes


def load_synth(name: str = "v1") -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Свой датасет: X[N, 2, n] (memmap) и meta.parquet."""
    from amc.generator import CLASSES

    d = DATA / "synth" / name
    return np.load(d / "X.npy", mmap_mode="r"), pd.read_parquet(d / "meta.parquet"), CLASSES
