"""Многозадачная модель: вид модуляции (классификация) + параметры сигнала (регрессия), этап 4а.

Общий энкодер (ResNet1D) → голова классификации + голова регрессии параметров.
Потеря = CE + Σ_k w_k · Huber(нормированный параметр k), параметр учитывается только там, где он определён
(например, символьная скорость — только у цифровых сигналов): маска NaN.

Цели регрессии (из метаданных своего генератора):

| цель | преобразование | у каких классов |
|---|---|---|
| snr_db | как есть | все |
| cfo | ×100 (доли fs → проценты) | все |
| log2_rate | log2(1/sps) | цифровые |
| log2_bw | log2(bw99) | все |
| rolloff | как есть | линейные |
| log2_shift | log2(разнос ЧМн) | CPFSK, GFSK |

Аугментации согласованы с метками: поворот фазы метки не меняет; сдвиг частоты прибавляется к cfo;
инверсия спектра (сопряжение) меняет знак cfo.

Запуск::

    python -m amc.multitask --length 1024 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from amc.augment import normalize_power, represent, to_complex, to_iq
from amc.data import ROOT, load_synth
from amc.models import ResNet1D, n_params

TARGETS = ["snr_db", "cfo", "log2_rate", "log2_bw", "rolloff", "log2_shift"]


def make_targets(meta: pd.DataFrame) -> np.ndarray:
    """Матрица целей [N, 6] (NaN — параметр не определён для этого класса)."""
    t = pd.DataFrame({
        "snr_db": meta["snr_db"], "cfo": meta["cfo"] * 100, "log2_rate": np.log2(meta["symbol_rate"]),
        "log2_bw": np.log2(meta["bw99"]), "rolloff": meta["rolloff"], "log2_shift": np.log2(meta["fsk_shift"]),
    })
    return t[TARGETS].to_numpy(dtype=np.float32)


class MultiTaskNet(nn.Module):
    """ResNet1D-энкодер + классификация + регрессия."""

    def __init__(self, in_ch: int, n_classes: int, n_targets: int = len(TARGETS)):
        super().__init__()
        self.backbone = ResNet1D(in_ch, n_classes)
        d = self.backbone.emb_dim
        self.reg = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True), nn.Linear(128, n_targets))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        e = self.backbone.encode(x)
        return self.backbone.head(e), self.reg(e)


def augment_with_labels(x: torch.Tensor, t: torch.Tensor, gen: torch.Generator, cfo_max: float = 0.005) -> tuple[
        torch.Tensor, torch.Tensor]:
    """Поворот фазы, сдвиг частоты (+ к метке cfo), сопряжение (знак cfo); t — нормированные цели."""
    b, _, n = x.shape
    dev = x.device
    z = to_complex(x)
    theta = torch.rand(b, 1, device=dev, generator=gen) * 2 * math.pi
    f = (torch.rand(b, 1, device=dev, generator=gen) * 2 - 1) * cfo_max
    z = z * torch.exp(1j * (theta + 2 * math.pi * f * torch.arange(n, device=dev)[None]))
    conj = torch.rand(b, device=dev, generator=gen) < 0.5
    z = torch.where(conj[:, None], z.conj(), z)
    t = t.clone()
    cfo = t[:, 1] + f[:, 0] * 100
    t[:, 1] = torch.where(conj, -cfo, cfo)
    return to_iq(z), t


def masked_huber(pred: torch.Tensor, target: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Huber по нормированным целям, только для определённых (не NaN) значений; среднее по целям."""
    m = ~torch.isnan(target)
    tn = torch.nan_to_num(target) / scale
    loss = F.smooth_l1_loss(pred, tn, reduction="none") * m
    return (loss.sum(0) / m.sum(0).clamp_min(1)).mean()


def train_mt(length: int = 1024, seed: int = 0, epochs: int = 25, batch: int = 256, lr: float = 2e-3,
             w_reg: float = 1.0, data: str = "v1", tag: str = "") -> dict:
    """Обучить многозадачную модель на своём датасете."""
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"synth_{data}_{length}_multitask_s{seed}{tag}"
    out = ROOT / "models" / "deep" / name
    out.mkdir(parents=True, exist_ok=True)
    Xm, meta, classes = load_synth(data)
    X = normalize_power(torch.from_numpy(np.ascontiguousarray(Xm[:, :, :length], dtype=np.float32)).to(dev))
    y = torch.from_numpy(meta["label"].to_numpy().copy()).to(dev)
    T = torch.from_numpy(make_targets(meta)).to(dev)
    split = meta["split"].to_numpy()
    tr = torch.from_numpy(np.flatnonzero(split == "train")).to(dev)
    va = torch.from_numpy(np.flatnonzero(split == "val")).to(dev)
    # масштаб целей — СКО по train (cfo — с учётом аугментации)
    scale = torch.nanmean((T[tr] - torch.nanmean(T[tr], 0)) ** 2, 0).sqrt().clamp_min(1e-3)
    model = MultiTaskNet(2, len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = (len(tr) + batch - 1) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    gen = torch.Generator(device=dev).manual_seed(seed)
    print(f"{name}: {n_params(model):,} параметров")

    def evaluate(idx: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
        model.eval()
        ps, rs = [], []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lo, r = model(represent(X[idx[s:s + 2048]], "iq"))
                ps.append(F.softmax(lo.float(), 1))
                rs.append(r.float() * scale)
        p, r = torch.cat(ps), torch.cat(rs)
        loss = F.nll_loss(p.clamp_min(1e-12).log(), y[idx]).item() + w_reg * masked_huber(r, T[idx], scale).item()
        return loss, p, r

    best, best_loss, bad, hist, t0 = None, float("inf"), 0, [], time.time()
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev, generator=gen)]
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            xb, tb = augment_with_labels(X[idx], T[idx], gen)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lo, r = model(represent(xb, "iq"))
            loss = F.cross_entropy(lo.float(), y[idx]) + w_reg * masked_huber(r.float(), tb, scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        vloss, pv, _ = evaluate(va)
        vacc = (pv.argmax(1) == y[va]).float().mean().item()
        hist.append({"epoch": ep + 1, "val_loss": vloss, "val_acc": vacc})
        el = time.time() - t0
        print(f"эпоха {ep + 1}/{epochs}: val {vloss:.4f}, acc {vacc:.4f}; прошло {el:.0f} с, "
              f"осталось ≤ {el / (ep + 1) * (epochs - ep - 1):.0f} с", flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, bad, best = vloss, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(best)
    allidx = torch.arange(len(X), device=dev)
    _, p, r = evaluate(allidx)
    torch.save({"state": model.state_dict(), "scale": scale.cpu(), "targets": TARGETS}, out / "model.pt")
    np.savez(out / "pred.npz", proba=p.cpu().numpy().astype(np.float32), reg=r.cpu().numpy().astype(np.float32),
             split=split, label=meta["label"].to_numpy())
    (out / "history.json").write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    print("готово:", out, f"{time.time() - t0:.0f} с")
    return {"name": name, "best_val_loss": best_loss}


def main() -> None:
    """Командная строка: обучение многозадачной сети."""
    ap = argparse.ArgumentParser(description="Многозадачная модель: модуляция + параметры")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--w_reg", type=float, default=1.0)
    a = ap.parse_args()
    train_mt(a.length, a.seed, a.epochs, w_reg=a.w_reg)


if __name__ == "__main__":
    main()
