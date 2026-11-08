"""Как ведёт себя M2M4 на QPSK без многолучёвости (ka 1,0 / 1,2 / откалиброванный 1,265)."""
from dataclasses import replace

import numpy as np

from amc import estimators as est
from amc.generator import GenConfig, generate_one

CFG = GenConfig(n=4096, p_multipath=0.0, phase_noise=(0.0, 0.0), sro_ppm=0.0)
for snr in [0, 5, 10, 15, 20]:
    zs, ka_true = [], []
    for i in range(8):
        z, m, s = generate_one("QPSK", np.random.default_rng(i), replace(CFG, snr_db=(snr, snr)), return_clean=True)
        zs.append(z)
        s = s.astype(complex)
        ka_true.append(np.mean(np.abs(s) ** 4) / np.mean(np.abs(s) ** 2) ** 2)
    z = np.stack(zs).astype(complex)
    r = {ka: np.median(est.snr_m2m4(z, ka)) for ka in (1.0, 1.2, 1.265)}
    print(snr, "истинный ka", np.round(np.median(ka_true), 3), {k: round(v, 2) for k, v in r.items()},
          "spectral", round(float(np.median(est.snr_spectral(z))), 2))
