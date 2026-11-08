"""Калибровка ka = E|s|⁴/(E|s|²)² для M2M4 на чистых сигналах генератора (с формированием спектра)."""
import numpy as np

from amc.generator import CLASSES, GenConfig, generate_one

cfg = GenConfig(snr_db=(60.0, 60.0))
for c in CLASSES:
    ka = []
    for i in range(300):
        _, _, s = generate_one(c, np.random.default_rng(10_000 + i), cfg, return_clean=True)
        s = s.astype(complex)
        ka.append(np.mean(np.abs(s) ** 4) / np.mean(np.abs(s) ** 2) ** 2)
    print(f'"{c}": {np.median(ka):.3f},  # IQR {np.percentile(ka, 25):.3f}–{np.percentile(ka, 75):.3f}')
