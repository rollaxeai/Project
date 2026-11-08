"""Тестовые WAV с известными параметрами для проверки продукта (amc.infer).

Чистый сигнал генератора (fs_base = 12 кГц, 16384 отсчёта, без многолучёвости) передискретизируется в 4 раза
(fs = 48 кГц), переносится на несущую 9 кГц; добавляется **белый шум по всей полосе записи** (как в реальном
приёмнике) так, что ОСШ в полосе сигнала (99 %) равно 15 дБ. Записывается:
- стерео float32 (I/Q) — ``<класс>_iq.wav``;
- моно PCM16 (вещественный) — ``<класс>_real.wav``.
Истинные параметры — в ``truth.csv``.

    python -X utf8 tools/make_test_wavs.py data/test_wavs
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal as sps

from amc.generator import CLASSES, GenConfig, generate_one

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test_wavs")
out.mkdir(parents=True, exist_ok=True)
FS0, UP, FC, SNR = 12000, 4, 9000.0, 15.0
cfg = replace(GenConfig(), n=16384, snr_db=(15.0, 15.0), p_multipath=0.0, cfo=0.01, sps=(6.0, 10.0))
rng = np.random.default_rng(7)
rows = []
for cls in CLASSES:
    _, m, x = generate_one(cls, rng, cfg, return_clean=True)
    y = sps.resample_poly(x.astype(np.complex128), UP, 1)
    p_sig = np.mean(np.abs(y) ** 2)
    n0 = p_sig / 10 ** (SNR / 10) / (m["bw99"] * FS0)        # СПМ шума, чтобы ОСШ в полосе 99 % = SNR
    sigma = np.sqrt(n0 * FS0 * UP / 2)
    n = np.arange(len(y))
    y = y * np.exp(2j * np.pi * FC / (FS0 * UP) * n)
    y += sigma * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    y /= np.abs(y).max() * 1.05
    sf.write(out / f"{cls}_iq.wav", np.stack([y.real, y.imag], 1).astype(np.float32), FS0 * UP, subtype="FLOAT")
    sf.write(out / f"{cls}_real.wav", y.real, FS0 * UP, subtype="PCM_16")
    rows.append({"cls": cls, "carrier_Hz": FC + m["cfo"] * FS0,
                 "symbol_rate_Bd": FS0 / m["sps"] if cls not in ("AM-DSB", "AM-SSB", "WBFM") else np.nan,
                 "bw99_Hz": m["bw99"] * FS0, "rolloff": m.get("rolloff", np.nan),
                 "fsk_shift_Hz": m["fsk_shift"] * FS0, "snr_inband_db": SNR})
pd.DataFrame(rows).to_csv(out / "truth.csv", index=False)
print(pd.DataFrame(rows).round(2).to_string())
