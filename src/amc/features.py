"""Экспертные признаки сигнала для классических моделей (baseline).

Все признаки инвариантны к начальной фазе и масштабу (сигнал нормируется на единичную мощность,
от комплексных кумулянтов берутся модули). Расчёт векторизован по пачке сигналов ``z[N, n]``.

Группы признаков:

* **кумулянты высших порядков** C20…C63 — теоретически различают созвездия ФМн/КАМ
  (BPSK: C42 = −2, QPSK/8PSK: −1, QAM16: −0,68, QAM64: −0,62, гауссов шум: 0);
* **признаки Азаза–Нанди** (γmax, σaa, σap, σdp, σaf) — классика различения АМ/ЧМ/ФМн
  по мгновенным амплитуде, фазе и частоте;
* **статистики огибающей и мгновенной частоты** (эксцесс, пик-фактор, постоянная составляющая);
* **спектральные**: плоскостность, концентрация, асимметрия, ширина;
* **линии спектров z², z⁴, z⁸** — признак кратности ФМн (возведение в M-ю степень снимает M-ФМн);
* **линия спектра огибающей** — признак символьной скорости у линейных модуляций.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "C20", "C40", "C41", "C42", "C60", "C61", "C62", "C63",
    "gmax", "s_aa", "s_ap", "s_dp", "s_af", "kurt_a", "kurt_f", "papr", "dc",
    "flat", "peak_frac", "occ90", "spec_skew", "spec_kurt",
    "x2_peak", "x4_peak", "x8_peak", "env_peak",
]


def normalize(z: np.ndarray) -> np.ndarray:
    """Привести каждый сигнал к единичной средней мощности."""
    z = np.asarray(z, dtype=np.complex128)
    return z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True) + 1e-20)


def cumulants(z: np.ndarray) -> dict[str, np.ndarray]:
    """Нормированные кумулянты 2–6 порядка (модули комплексных, вещественные C42 и C63)."""
    zc = np.conj(z)
    m = lambda a: np.mean(a, axis=1)  # noqa: E731
    z2, z4 = z**2, z**4
    M20, M21 = m(z2), m(z * zc)
    M40, M41, M42 = m(z4), m(z2 * z * zc), m(z2 * zc**2)
    M60, M61, M62, M63 = m(z4 * z2), m(z4 * z * zc), m(z4 * zc**2), m(z2 * z * zc**3)
    M22, M43 = np.conj(M20), np.conj(M41)
    C40 = M40 - 3 * M20**2
    C41 = M41 - 3 * M20 * M21
    C42 = M42 - np.abs(M20) ** 2 - 2 * M21**2
    C60 = M60 - 15 * M20 * M40 + 30 * M20**3
    C61 = M61 - 5 * M21 * M40 - 10 * M20 * M41 + 30 * M20**2 * M21
    C62 = M62 - 6 * M20 * M42 - 8 * M21 * M41 - M22 * M40 + 6 * M20**2 * M22 + 24 * M21**2 * M20
    C63 = M63 - 9 * M21 * M42 + 12 * M21**3 - 3 * M20 * M43 - 3 * M22 * M41 + 18 * M20 * M21 * M22
    c21 = M21.real
    return {
        "C20": np.abs(M20) / c21,
        "C40": np.abs(C40) / c21**2, "C41": np.abs(C41) / c21**2, "C42": C42.real / c21**2,
        "C60": np.abs(C60) / c21**3, "C61": np.abs(C61) / c21**3, "C62": np.abs(C62) / c21**3,
        "C63": C63.real / c21**3,
    }


def _masked_std(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """СКО по строкам только для отсчётов mask; если отсчётов < 2 — 0."""
    cnt = mask.sum(axis=1)
    xs = np.where(mask, x, 0.0)
    mean = xs.sum(axis=1) / np.maximum(cnt, 1)
    var = np.where(mask, (x - mean[:, None]) ** 2, 0.0).sum(axis=1) / np.maximum(cnt - 1, 1)
    return np.where(cnt >= 2, np.sqrt(var), 0.0)


def _kurt(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    return np.mean(xc**4, axis=1) / (np.mean(xc**2, axis=1) ** 2 + 1e-20)


def _line_strength(y: np.ndarray, skip_dc: bool = False) -> np.ndarray:
    """Насколько выражена спектральная линия: log(max / среднее) спектра |FFT(y)|²."""
    n = y.shape[1]
    s = np.abs(np.fft.fft(y, 2 * n, axis=1)) ** 2
    if skip_dc:
        s[:, :3] = 0
        s[:, -2:] = 0
    return np.log10(s.max(axis=1) / (s.mean(axis=1) + 1e-20) + 1e-20)


def compute_features(z: np.ndarray) -> pd.DataFrame:
    """Признаки для пачки сигналов z[N, n] (комплексные). Возвращает DataFrame [N, len(FEATURES)]."""
    z = normalize(z)
    n = z.shape[1]
    out = cumulants(z)

    # --- Азаз–Нанди
    a = np.abs(z)
    ma = a.mean(axis=1, keepdims=True)
    acn = a / ma - 1
    out["gmax"] = np.max(np.abs(np.fft.fft(acn, axis=1)) ** 2, axis=1) / n
    out["s_aa"] = np.sqrt(np.maximum(np.mean(acn**2, axis=1) - np.mean(np.abs(acn), axis=1) ** 2, 0))
    strong = (a / ma) > 1.0  # «сильные» отсчёты: фаза там надёжна
    phi = np.unwrap(np.angle(z), axis=1)
    t = np.arange(n) - (n - 1) / 2
    slope = (phi * t).sum(axis=1, keepdims=True) / (t**2).sum()
    phi_nl = phi - phi.mean(axis=1, keepdims=True) - slope * t  # убран линейный набег (сдвиг частоты)
    phi_nl = (phi_nl + np.pi) % (2 * np.pi) - np.pi
    out["s_ap"] = _masked_std(np.abs(phi_nl), strong)
    out["s_dp"] = _masked_std(phi_nl, strong)
    f = np.diff(phi, axis=1) / (2 * np.pi)
    fc = f - f.mean(axis=1, keepdims=True)
    out["s_af"] = _masked_std(np.abs(fc), strong[:, 1:])
    out["kurt_a"] = _kurt(acn)
    out["kurt_f"] = _kurt(fc)
    out["papr"] = 10 * np.log10(np.max(a**2, axis=1))
    out["dc"] = np.abs(z.mean(axis=1))

    # --- спектр
    P = np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)) ** 2 + 1e-20
    Pn = P / P.sum(axis=1, keepdims=True)
    fr = np.linspace(-0.5, 0.5, n, endpoint=False)
    out["flat"] = np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)
    out["peak_frac"] = Pn.max(axis=1)
    srt = -np.sort(-Pn, axis=1)
    out["occ90"] = (np.argmax(np.cumsum(srt, axis=1) >= 0.9, axis=1) + 1) / n
    mu = (Pn * fr).sum(axis=1, keepdims=True)
    var = (Pn * (fr - mu) ** 2).sum(axis=1)
    out["spec_skew"] = (Pn * (fr - mu) ** 3).sum(axis=1) / (var**1.5 + 1e-20)
    out["spec_kurt"] = (Pn * (fr - mu) ** 4).sum(axis=1) / (var**2 + 1e-20)

    # --- спектральные линии
    out["x2_peak"] = _line_strength(z**2)
    out["x4_peak"] = _line_strength(z**4)
    out["x8_peak"] = _line_strength(z**8)
    out["env_peak"] = _line_strength(a**2 - (a**2).mean(axis=1, keepdims=True), skip_dc=True)
    return pd.DataFrame({k: out[k].astype(np.float32) for k in FEATURES})


def features_from_iq(X: np.ndarray, batch: int = 8192, length: int | None = None) -> pd.DataFrame:
    """Признаки для массива X[N, 2, n] (I/Q, в т.ч. np.memmap) пачками; length — обрезать окно."""
    parts = []
    for s in range(0, len(X), batch):
        xb = np.asarray(X[s:s + batch, :, :length] if length else X[s:s + batch])
        parts.append(compute_features(xb[:, 0].astype(np.float64) + 1j * xb[:, 1]))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    """Рассчитать и сохранить признаки: data/features/{rml2016a,synth_v1,synth_v1_128}.parquet."""
    import time

    from amc.data import DATA, load_rml2016a, load_synth

    out = DATA / "features"
    out.mkdir(exist_ok=True)
    jobs = [("rml2016a", load_rml2016a, None), ("synth_v1", load_synth, None), ("synth_v1_128", load_synth, 128)]
    for name, loader, length in jobs:
        t0 = time.time()
        X, meta, _ = loader()
        f = features_from_iq(X, length=length)
        f = pd.concat([meta[["label", "cls", "snr_db", "split"]].reset_index(drop=True), f], axis=1)
        f.to_parquet(out / f"{name}.parquet")
        print(f"{name}: {f.shape}, {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
