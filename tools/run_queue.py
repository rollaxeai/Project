"""Очередь обучения этапа 4: запуски по порядку, уже выполненные (есть metrics.json) пропускаются.

Устойчиво к перерывам: после обрыва достаточно запустить снова — продолжит с невыполненного.
    python -X utf8 tools/run_queue.py
"""
import traceback

from amc.data import ROOT
from amc.train import TrainConfig, train

Q: list[TrainConfig] = []
# RadioML 2016.10a: три архитектуры × 3 seed'а (для статистического сравнения)
for model in ["resnet", "cnnlstm", "vtcnn2"]:
    for seed in [0, 1, 2]:
        Q.append(TrainConfig(data="rml2016a", model=model, seed=seed))
# абляции на RadioML (ResNet, seed 0): без аугментаций, другие входы
Q += [TrainConfig(data="rml2016a", model="resnet", aug=False),
      TrainConfig(data="rml2016a", model="resnet", input="ap"),
      TrainConfig(data="rml2016a", model="resnet", input="iqap")]
# свой датасет: 128 отсчётов (сравнение с RadioML) и 1024 отсчёта
Q += [TrainConfig(data="synth_v1", length=128, model="resnet", epochs=30)]
for seed in [0, 1, 2]:
    Q.append(TrainConfig(data="synth_v1", model="resnet", seed=seed, epochs=25, batch=256))
Q += [TrainConfig(data="synth_v1", model="cnnlstm", epochs=25, batch=256)]

# этап 4а: многозадачная модель (модуляция + параметры), свой датасет
MT = [dict(length=1024, seed=s) for s in (0, 1, 2)] + [dict(length=128, seed=0)]

if __name__ == "__main__":
    from amc.multitask import train_mt

    for i, cfg in enumerate(Q, 1):
        done = ROOT / "models" / "deep" / cfg.run_name / "metrics.json"
        if done.exists():
            print(f"[{i}/{len(Q)}] {cfg.run_name}: уже есть, пропуск", flush=True)
            continue
        print(f"[{i}/{len(Q)}] {cfg.run_name}: старт", flush=True)
        try:
            train(cfg)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    for kw in MT:
        done = ROOT / "models" / "deep" / f"synth_v1_{kw['length']}_multitask_s{kw['seed']}" / "pred.npz"
        if done.exists():
            continue
        print(f"многозадачная {kw}: старт", flush=True)
        try:
            train_mt(**kw)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    print("ОЧЕРЕДЬ ЗАВЕРШЕНА", flush=True)
