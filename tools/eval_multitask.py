"""Быстрая оценка многозадачной модели против классических оценщиков (тест своего датасета, 1024 отсчёта)."""
import numpy as np
import pandas as pd

from amc import estimators as est
from amc.data import ROOT, load_synth
from amc.multitask import TARGETS, make_targets

X, meta, _ = load_synth("v1")
te = np.flatnonzero(meta["split"].to_numpy() == "test")
T = make_targets(meta)[te]
snr = meta["snr_db"].to_numpy()[te]
rows = []
for s in (0, 1, 2):
    d = np.load(ROOT / "models" / "deep" / f"synth_v1_1024_multitask_s{s}" / "pred.npz", allow_pickle=True)
    p, r = d["proba"][te], d["reg"][te]
    acc = (p.argmax(1) == meta["label"].to_numpy()[te])
    rows.append({"seed": s, "acc": acc.mean(), "acc>=0": acc[snr >= 0].mean(), **{
        t: np.nanmean(np.abs(r[:, i] - T[:, i])) for i, t in enumerate(TARGETS)}})
mt = pd.DataFrame(rows)
print("многозадачная модель (тест), MAE в единицах целей:")
print(mt.round(4).to_string(index=False))

# классические оценщики с истинным классом (оракул) — верхняя граница для каскада «класс → оценщик»
z = np.asarray(X[te]).astype(np.float64)
z = z[:, 0] + 1j * z[:, 1]
cls = meta["cls"].to_numpy()[te]
parts = [est.estimate_all(z[i:i + 4000], cls[i:i + 4000]) for i in range(0, len(z), 4000)]
E = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
C = np.stack([E["snr_db"], E["cfo"] * 100, np.log2(np.clip(E["symbol_rate"], 1e-6, None)),
              np.log2(np.clip(E["bw99"], 1e-6, None)), E["rolloff"],
              np.log2(np.clip(E["fsk_shift"], 1e-6, None))], axis=1)
mask = ~np.isnan(T)
cl = {t: np.mean(np.abs(C[mask[:, i], i] - T[mask[:, i], i])) for i, t in enumerate(TARGETS)}
d = np.load(ROOT / "models" / "deep" / "synth_v1_1024_multitask_s0" / "pred.npz", allow_pickle=True)
r = d["reg"][te]
out = []
for lo, hi in [(-20, -10), (-10, 0), (0, 10), (10, 20)]:
    b = (snr >= lo) & (snr < hi)
    row = {"ОСШ": f"{lo}…{hi}"}
    for i, t in enumerate(TARGETS):
        m = b & mask[:, i]
        row[f"{t} ML"] = np.mean(np.abs(r[m, i] - T[m, i]))
        row[f"{t} ЦОС"] = np.mean(np.abs(C[m, i] - T[m, i]))
    out.append(row)
print("\nклассика (истинный класс), MAE всего:", {k: round(v, 4) for k, v in cl.items()})
print("\nпо ОСШ (seed 0):")
print(pd.DataFrame(out).round(3).T.to_string())
