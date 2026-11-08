"""Загрузка обученных моделей и предсказание на произвольных данных."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from amc.augment import IN_CHANNELS, normalize_power, represent
from amc.data import ROOT
from amc.models import build

DEEP = ROOT / "models" / "deep"


def load_model(run: str, device: str | torch.device | None = None) -> tuple[torch.nn.Module, dict]:
    """Модель запуска models/deep/<run> (веса model.pt + конфигурация из metrics.json)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = json.loads((DEEP / run / "metrics.json").read_text(encoding="utf-8"))["config"]
    length = cfg["length"] or (128 if cfg["data"] == "rml2016a" else 1024)
    model = build(cfg["model"], IN_CHANNELS[cfg["input"]], 11, length).to(dev)
    model.load_state_dict(torch.load(DEEP / run / "model.pt", map_location=dev))
    model.eval()
    cfg["_length"] = length
    return model, cfg


@torch.no_grad()
def predict_proba(model: torch.nn.Module, cfg: dict, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Вероятности классов для X[N, 2, n] (n должно совпадать с длиной обучения для VT-CNN2)."""
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(X), batch):
        xb = normalize_power(torch.from_numpy(np.ascontiguousarray(X[s:s + batch], dtype=np.float32)).to(dev))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(F.softmax(model(represent(xb, cfg["input"])).float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def perturb(X: np.ndarray, cfo: float = 0.0, conj: bool = False, phase: float = 0.0) -> np.ndarray:
    """Искажение I/Q: сдвиг частоты cfo (доли fs), инверсия спектра, поворот фазы."""
    z = X[:, 0].astype(np.float64) + 1j * X[:, 1]
    n = np.arange(z.shape[1])
    z = z * np.exp(1j * (2 * np.pi * cfo * n + phase))
    if conj:
        z = np.conj(z)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def run_dir(run: str) -> Path:
    """Папка запуска."""
    return DEEP / run
