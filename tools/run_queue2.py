"""Очередь 2 (этап 4): дообучение моделей своего датасета на 60 эпох (в очереди 1 все остановились на 25-й).

Пропускает готовые запуски — после обрыва просто запустить снова.
"""
import traceback

from amc.data import ROOT
from amc.multitask import train_mt
from amc.train import TrainConfig, train

Q = []
for seed in (0, 1, 2):
    Q.append(TrainConfig(data="synth_v1", model="cnnlstm", seed=seed, epochs=60, batch=256, patience=10, tag="_e60"))
for seed in (0, 1, 2):
    Q.append(TrainConfig(data="synth_v1", model="resnet", seed=seed, epochs=60, batch=256, patience=10, tag="_e60"))
MT = [dict(length=1024, seed=s, epochs=60, tag="_e60") for s in (0, 1, 2)]

if __name__ == "__main__":
    for i, cfg in enumerate(Q, 1):
        if (ROOT / "models" / "deep" / cfg.run_name / "metrics.json").exists():
            continue
        print(f"[{i}/{len(Q)}] {cfg.run_name}: старт", flush=True)
        try:
            train(cfg)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    for kw in MT:
        if (ROOT / "models" / "deep" / f"synth_v1_1024_multitask_s{kw['seed']}{kw['tag']}" / "pred.npz").exists():
            continue
        print(f"многозадачная {kw}: старт", flush=True)
        try:
            train_mt(**kw)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    print("ОЧЕРЕДЬ 2 ЗАВЕРШЕНА", flush=True)
