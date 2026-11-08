"""Диагностика классических оценщиков по классам и многолучёвости (тест, ОСШ ≥ 10 дБ)."""
import numpy as np

from amc import estimators as est
from amc.data import load_synth

X, meta, _ = load_synth("v1")
m = meta[(meta.split == "test") & (meta.snr_db >= 10)].sample(6000, random_state=0)
z = np.asarray(X[np.sort(m.index.to_numpy())]).astype(np.float64)
m = m.sort_index()
z = z[:, 0] + 1j * z[:, 1]
E = est.estimate_all(z, m["cls"].to_numpy())
m = m.assign(rate_err=np.abs(np.log2(np.clip(E["symbol_rate"], 1e-6, None) / m["symbol_rate"])),
             bw_err=np.abs(np.log2(np.clip(E["bw99"], 1e-6, None) / m["bw99"])),
             bw_ratio=E["bw99"] / m["bw99"], rate_ratio=E["symbol_rate"] / m["symbol_rate"])
print(m.groupby(["cls", "multipath"])[["rate_err", "bw_err"]].median().round(3).unstack())
print(m.groupby("cls")[["bw_ratio", "rate_ratio"]].median().round(3))
