"""Обучение нейросетевого классификатора вида модуляции.

Весь датасет загружается в память видеокарты (RadioML — 0,2 ГБ, свой — 1,7 ГБ), аугментации выполняются
на GPU пачками — без DataLoader и копирований через CPU.

Схема: AdamW + OneCycle, смешанная точность (bfloat16), ранняя остановка по потере на val,
лучшие по val веса. Выбор модели — только по val; тест считается и сохраняется, но для выбора не используется.
Результаты: веса, вероятности классов для всех примеров (для ансамбля), метрики → models/deep/<run>/ и MLflow.

Запуск::

    python -m amc.train --data rml2016a --model resnet --input iq --seed 0
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from amc.augment import IN_CHANNELS, augment, normalize_power, represent
from amc.data import ROOT, load_rml2016a, load_synth
from amc.evaluate import summary
from amc.models import build, n_params


@dataclass
class TrainConfig:
    """Параметры обучения."""

    data: str = "rml2016a"         # rml2016a | synth_v1
    length: int = 0                # 0 — вся длина; иначе первые length отсчётов
    model: str = "resnet"          # vtcnn2 | resnet | cnnlstm
    input: str = "iq"              # iq | ap | iqap
    aug: bool = True
    epochs: int = 40
    batch: int = 512
    lr: float = 2e-3
    wd: float = 1e-4
    patience: int = 7
    seed: int = 0
    tag: str = ""

    @property
    def run_name(self) -> str:  # noqa: D102
        ln = f"_{self.length}" if self.length else ""
        return f"{self.data}{ln}_{self.model}_{self.input}_{'aug' if self.aug else 'noaug'}_s{self.seed}{self.tag}"


def load(cfg: TrainConfig) -> tuple[np.ndarray, pd.DataFrame, list[str]]:  # noqa: F821
    """Данные по имени: X[N, 2, n] float32, meta (label, snr_db, split), классы."""
    if cfg.data == "rml2016a":
        X, meta, classes = load_rml2016a()
    elif cfg.data.startswith("synth_"):
        X, meta, classes = load_synth(cfg.data.split("_", 1)[1])
    else:
        raise ValueError(cfg.data)
    if cfg.length:
        X = X[:, :, : cfg.length]
    return np.ascontiguousarray(X, dtype=np.float32), meta.reset_index(drop=True), classes


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: torch.Tensor, kind: str, batch: int = 4096) -> torch.Tensor:
    """Вероятности классов для всех примеров X (без аугментаций)."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out.append(F.softmax(model(represent(X[s:s + batch], kind)).float(), dim=1))
    return torch.cat(out)


def train(cfg: TrainConfig) -> dict:
    """Обучить модель по конфигурации; вернуть словарь метрик."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = ROOT / "models" / "deep" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    Xn, meta, classes = load(cfg)
    X = normalize_power(torch.from_numpy(Xn).to(dev))
    del Xn
    y = torch.from_numpy(meta["label"].to_numpy().copy()).to(dev)
    split = meta["split"].to_numpy()
    tr = torch.from_numpy(np.flatnonzero(split == "train")).to(dev)
    va = torch.from_numpy(np.flatnonzero(split == "val")).to(dev)

    model = build(cfg.model, IN_CHANNELS[cfg.input], len(classes), X.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = (len(tr) + cfg.batch - 1) // cfg.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)

    print(f"{cfg.run_name}: {n_params(model):,} параметров, train {len(tr)}, val {len(va)}, устройство {dev}")
    best, best_loss, bad, hist = None, float("inf"), 0, []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev, generator=gen)]
        tot = 0.0
        for s in range(0, len(perm), cfg.batch):
            idx = perm[s:s + cfg.batch]
            xb = X[idx]
            if cfg.aug:
                xb = augment(xb, gen)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(represent(xb, cfg.input)).float(), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        pv = predict_proba(model, X[va], cfg.input)
        vloss = F.nll_loss(pv.clamp_min(1e-12).log(), y[va]).item()
        vacc = (pv.argmax(1) == y[va]).float().mean().item()
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_loss": vloss, "val_acc": vacc})
        el = time.time() - t0
        print(f"эпоха {ep + 1}/{cfg.epochs}: train {tot / len(tr):.4f}, val {vloss:.4f}, val acc {vacc:.4f}; "
              f"прошло {el:.0f} с, осталось ≤ {el / (ep + 1) * (cfg.epochs - ep - 1):.0f} с", flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, bad, best = vloss, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                print("ранняя остановка")
                break

    model.load_state_dict(best)
    proba = predict_proba(model, X, cfg.input).cpu().numpy()
    pred = proba.argmax(1)
    yy, snr = meta["label"].to_numpy(), meta["snr_db"].to_numpy()
    metrics = {f"{part}_{k}": v for part in ("val", "test")
               for k, v in summary(yy[split == part], pred[split == part], snr[split == part], classes).items()}
    metrics.update(params=n_params(model), epochs_run=len(hist), train_time_s=time.time() - t0)
    torch.save(model.state_dict(), out_dir / "model.pt")
    np.savez(out_dir / "proba.npz", proba=proba.astype(np.float32), split=split, label=yy, snr=snr)
    (out_dir / "metrics.json").write_text(json.dumps({"config": asdict(cfg), "metrics": metrics, "history": hist},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    _log_mlflow(cfg, metrics, hist, out_dir)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, ensure_ascii=False))
    return metrics


def _safe(name: str) -> str:
    """Имя метрики для MLflow: только латиница, цифры, _ - . (кириллицу и >= заменить)."""
    return name.replace(">=", "_ge").replace("без_", "no_")


def _log_mlflow(cfg: TrainConfig, metrics: dict, hist: list[dict], out_dir: Path) -> None:
    """Записать запуск в MLflow (локальная база mlflow.db); ошибки MLflow не прерывают обучение."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("amc-classification")
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(asdict(cfg))
            for h in hist:
                mlflow.log_metrics({k: v for k, v in h.items() if k != "epoch"}, step=h["epoch"])
            mlflow.log_metrics({_safe(k): float(v) for k, v in metrics.items()})
            mlflow.log_artifact(str(out_dir / "metrics.json"))
    except Exception as e:  # noqa: BLE001
        print("MLflow: запись не удалась:", e)


def main() -> None:
    """Командная строка: обучение классификатора."""
    ap = argparse.ArgumentParser(description="Обучение классификатора вида модуляции")
    for k, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    train(TrainConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
