"""Безопасная загрузка RadioML 2016.10a и конвертация в npz.

Файл ``RML2016.10a_dict.pkl`` — это pickle (Python 2), скачанный со стороннего зеркала.
Обычный ``pickle.load`` может выполнить произвольный код, поэтому используется
ограниченный ``Unpickler``: разрешены только классы, нужные для восстановления
массивов numpy. Всё остальное — ошибка.

Запуск::

    python -m amc.radioml data/radioml/RML2016.10a_dict.pkl data/radioml/rml2016a.npz
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_ALLOWED: set[tuple[str, str]] = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler с белым списком классов numpy."""

    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _ALLOWED:
            # numpy 2: старый путь numpy.core → numpy._core
            if module.startswith("numpy.core"):
                module = module.replace("numpy.core", "numpy._core")
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"запрещённый класс в pickle: {module}.{name}")


def load_rml2016_pkl(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Загрузить RML2016.10a из pickle.

    Returns:
        X: float32 [N, 2, 128] — I и Q;
        y: int64 [N] — номер класса (индекс в ``classes``);
        snr: int64 [N] — ОСШ, дБ;
        classes: отсортированный список названий модуляций.

    """
    with open(path, "rb") as f:
        data = _SafeUnpickler(f, encoding="latin1").load()
    if not isinstance(data, dict):
        raise ValueError("ожидался dict {(модуляция, ОСШ): массив}")
    classes = sorted({k[0] for k in data})
    xs, ys, snrs = [], [], []
    for (mod, snr), arr in sorted(data.items()):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2:
            raise ValueError(f"неожиданная форма {arr.shape} для {(mod, snr)}")
        xs.append(arr)
        ys.append(np.full(len(arr), classes.index(mod), dtype=np.int64))
        snrs.append(np.full(len(arr), int(snr), dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(snrs), classes


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Загрузить уже сконвертированный npz (без pickle)."""
    d = np.load(path, allow_pickle=False)
    return d["X"], d["y"], d["snr"], [str(c) for c in d["classes"]]


def main(argv: list[str]) -> None:
    """Командная строка: конвертация RadioML в npz."""
    src, dst = argv[1], argv[2]
    X, y, snr, classes = load_rml2016_pkl(src)
    np.savez(dst, X=X, y=y, snr=snr, classes=np.array(classes))
    print(f"X {X.shape} {X.dtype}; классов {len(classes)}: {classes}")
    print(f"ОСШ: {sorted(set(snr.tolist()))}")
    print(f"сохранено: {dst}")


if __name__ == "__main__":
    main(sys.argv)
