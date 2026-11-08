"""Классические (ЦОС) оценщики параметров сигнала — baseline для регрессии параметров (этап 4а).

Все частоты — в долях частоты дискретизации (циклов на отсчёт). Векторизовано по пачке z[N, n].

| Параметр | Метод | Идея |
|---|---|---|
| ОСШ | M2M4 | моменты 2-го и 4-го порядка смеси «сигнал + гауссов шум» дают S и N отдельно |
| ОСШ | спектральный | уровень шумовой «полки» по нижним квантилям спектра |
| сдвиг частоты | линия спектра z^M | M-я степень снимает M-ФМн → линия на M·cfo |
| сдвиг частоты | центр тяжести спектра | для симметричного спектра совпадает с несущей |
| символьная скорость | линия спектра огибающей |z|² | у линейных модуляций огибающая периодична с Rs |
| символьная скорость | линия спектра |Δf| | у ЧМн мгновенная частота скачет на границах символов |
| полоса 99 % | спектр Уэлча за вычетом шумовой полки | как определена метка `bw99` |
| roll-off | (полоса · sps) − 1 | для RRC полоса ≈ (1 + β)·Rs |
| разнос ЧМн | квантили мгновенной частоты | два тона ±Δ/2 |
"""

from __future__ import annotations

import numpy as np

from amc.features import normalize

# ka = E|s|⁴ / (E|s|²)² — эксцесс модуля сигнала. Теоретические значения (BPSK/QPSK = 1, QAM16 = 1,32)
# верны только для отсчётов в моменты символов; у сигнала с формированием RRC и многолучёвостью огибающая
# между символами меняется, поэтому ka откалиброван по чистым сигналам генератора (tools/calib_ka.py, медиана).
KA = {"8PSK": 1.260, "AM-DSB": 1.174, "AM-SSB": 2.179, "BPSK": 1.393, "CPFSK": 1.078, "GFSK": 1.045,
      "PAM4": 1.903, "QAM16": 1.519, "QAM64": 1.565, "QPSK": 1.265, "WBFM": 1.030}
M_POWER = {"BPSK": 2, "PAM4": 2, "QPSK": 4, "QAM16": 4, "QAM64": 4, "8PSK": 8}


def _peak_freq(s: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Частота максимума спектра s[N, K] с параболической интерполяцией по трём точкам."""
    k = np.argmax(s, axis=1)
    i = np.arange(len(s))
    km, kp = np.clip(k - 1, 0, s.shape[1] - 1), np.clip(k + 1, 0, s.shape[1] - 1)
    a, b, c = np.log(s[i, km] + 1e-30), np.log(s[i, k] + 1e-30), np.log(s[i, kp] + 1e-30)
    d = 0.5 * (a - c) / (a - 2 * b + c - 1e-30)
    df = freqs[1] - freqs[0]
    return freqs[k] + np.clip(d, -0.5, 0.5) * df


def snr_m2m4(z: np.ndarray, ka: float | np.ndarray = 1.0) -> np.ndarray:
    """ОСШ (дБ) по моментам M2, M4: M4 = ka·S² + 4·S·N + 2·N² (комплексный гауссов шум, kw = 2)."""
    m2 = np.mean(np.abs(z) ** 2, axis=1)
    m4 = np.mean(np.abs(z) ** 4, axis=1)
    ka = np.broadcast_to(np.asarray(ka, dtype=float), m2.shape)
    # из M2 = S + N и M4: (ka − 2)·S² + 2·M2² − M4 = 0  →  S = sqrt((2·M2² − M4) / (2 − ka))
    s = np.sqrt(np.clip((2 * m2**2 - m4) / np.where(np.abs(2 - ka) < 1e-6, 1e-6, 2 - ka), 1e-12, None))
    s = np.minimum(s, m2 * (1 - 1e-6))
    return 10 * np.log10(s / (m2 - s))


def _welch(z: np.ndarray, nper: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Спектр Уэлча (окно Ханна, перекрытие 50 %), частоты от −0,5 до 0,5."""
    n = z.shape[1]
    nper = min(nper, n)
    step = nper // 2
    starts = range(0, n - nper + 1, step)
    w = np.hanning(nper)
    p = np.mean([np.abs(np.fft.fft(z[:, s:s + nper] * w, axis=1)) ** 2 for s in starts], axis=0)
    p = np.fft.fftshift(p, axes=1) / np.sum(w**2)
    return np.fft.fftshift(np.fft.fftfreq(nper)), p


def noise_floor(z: np.ndarray, q: float = 0.2) -> np.ndarray:
    """Доля мощности шума по «полке» спектра: среднее нижних q·100 % бинов, умноженное на число бинов."""
    _, p = _welch(z)
    srt = np.sort(p, axis=1)
    k = max(1, int(q * p.shape[1]))
    floor = srt[:, :k].mean(axis=1) * 1.25  # поправка: нижние квантили χ²-распределённых бинов ниже среднего
    return np.clip(floor * p.shape[1] / p.sum(axis=1), 1e-6, 1 - 1e-6)


def snr_spectral(z: np.ndarray) -> np.ndarray:
    """ОСШ (дБ) по шумовой полке спектра (работает и для аналоговых сигналов; ломается, если сигнал занимает всю полосу)."""
    nf = noise_floor(z)
    return 10 * np.log10((1 - nf) / nf)


def cfo_xm(z: np.ndarray, m: int | np.ndarray, pad: int = 8) -> np.ndarray:
    """Сдвиг частоты по линии спектра z^M (M = 2 / 4 / 8 по виду ФМн/КАМ)."""
    z = normalize(z)
    n = z.shape[1]
    nfft = pad * n
    freqs = np.fft.fftfreq(nfft)
    out = np.empty(len(z))
    m_arr = np.broadcast_to(np.asarray(m), (len(z),))
    for mm in np.unique(m_arr):
        idx = np.flatnonzero(m_arr == mm)
        s = np.abs(np.fft.fft(z[idx] ** int(mm), nfft, axis=1)) ** 2
        f = _peak_freq(np.fft.fftshift(s, axes=1), np.fft.fftshift(freqs))
        out[idx] = f / mm
    return out


def cfo_centroid(z: np.ndarray) -> np.ndarray:
    """Сдвиг частоты как центр тяжести спектра (за вычетом шумовой полки)."""
    f, p = _welch(normalize(z))
    p = np.clip(p - np.sort(p, axis=1)[:, : max(1, p.shape[1] // 5)].mean(axis=1, keepdims=True) * 1.25, 0, None)
    return (p * f).sum(axis=1) / (p.sum(axis=1) + 1e-20)


def rate_envelope(z: np.ndarray, fmin: float = 1 / 20, fmax: float = 1 / 2.5, pad: int = 4) -> np.ndarray:
    """Символьная скорость (1/sps) по линии спектра |z|² (линейные модуляции)."""
    e = np.abs(normalize(z)) ** 2
    e = e - e.mean(axis=1, keepdims=True)
    return _line_in_band(e, fmin, fmax, pad)


def rate_freqdiff(z: np.ndarray, fmin: float = 1 / 20, fmax: float = 1 / 2.5, pad: int = 4) -> np.ndarray:
    """Символьная скорость по линии спектра |Δ мгновенной частоты| (ЧМн: скачки на границах символов)."""
    ph = np.unwrap(np.angle(z), axis=1)
    f = np.diff(ph, axis=1)
    d = np.abs(np.diff(f, axis=1))
    d = d - d.mean(axis=1, keepdims=True)
    # без отбеливания: у |Δf| острые гармоники 2Rs…5Rs, отбеливание выводит их наверх
    return _line_in_band(d, fmin, fmax, pad, whiten=False)


def _line_in_band(x: np.ndarray, fmin: float, fmax: float, pad: int, whiten: bool = True) -> np.ndarray:
    """Частота самой выраженной спектральной линии в полосе [fmin, fmax].

    Спектр «отбеливается» — делится на локальный фон (скользящая медиана): у огибающей PAM4/BPSK и при
    замираниях есть сильная случайная низкочастотная составляющая, на фоне которой абсолютный максимум
    оказывается не на символьной линии (tools/diag_estimators.py: ошибки до 30–50 % без отбеливания).
    """
    from scipy.ndimage import median_filter

    nfft = pad * x.shape[1]
    s = np.abs(np.fft.rfft(x, nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft)
    w = s / (median_filter(s, size=(1, 16 * pad + 1), mode="nearest") + 1e-20) if whiten else s
    band = (freqs >= fmin) & (freqs <= fmax)
    return _peak_freq(w[:, band], freqs[band])


def bandwidth99(z: np.ndarray, frac: float = 0.99) -> np.ndarray:
    """Полоса, содержащая долю frac мощности сигнала (шумовая полка вычтена)."""
    f, p = _welch(normalize(z), nper=256 if z.shape[1] >= 256 else z.shape[1])
    from scipy.ndimage import uniform_filter1d

    # спектр Уэлча по 1024 отсчётам шумный (≈7 усреднений): сглаживание по частоте, чтобы отдельные
    # шумовые бины не проходили порог
    p = uniform_filter1d(p, 7, axis=1, mode="wrap")
    floor = np.sort(p, axis=1)[:, : max(1, p.shape[1] // 5)].mean(axis=1, keepdims=True) * 1.1
    # бины ниже уровня «полка + 3 дБ» считаются шумом и обнуляются: иначе остаток шума по всей полосе
    # растягивает полосу 99 % (аналоговые сигналы: завышение в 2–3 раза, tools/diag_estimators.py)
    p = np.where(p > 2 * floor, p - floor, 0.0) + 1e-20
    c = np.cumsum(p, axis=1) / p.sum(axis=1, keepdims=True)
    lo = np.array([f[np.searchsorted(ci, (1 - frac) / 2)] for ci in c])
    hi = np.array([f[min(np.searchsorted(ci, 1 - (1 - frac) / 2), len(f) - 1)] for ci in c])
    return hi - lo


def rolloff_from(bw: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """roll-off ≈ полоса / Rs − 1 (для RRC), ограничен [0, 1]."""
    return np.clip(bw / np.maximum(rate, 1e-6) - 1, 0, 1)


def fsk_shift(z: np.ndarray) -> np.ndarray:
    """Разнос частот двухпозиционной ЧМн: разность 90-го и 10-го процентилей мгновенной частоты."""
    ph = np.unwrap(np.angle(z), axis=1)
    f = np.diff(ph, axis=1) / (2 * np.pi)
    k = 3  # сглаживание скользящим средним против шума
    f = np.apply_along_axis(lambda r: np.convolve(r, np.ones(k) / k, mode="valid"), 1, f)
    return np.percentile(f, 90, axis=1) - np.percentile(f, 10, axis=1)


def estimate_all(z: np.ndarray, classes: list[str] | np.ndarray) -> dict[str, np.ndarray]:
    """Все классические оценки; там, где метод зависит от вида модуляции, используется класс classes[i]."""
    classes = np.asarray(classes)
    ka = np.array([KA.get(c, 1.0) for c in classes])
    m = np.array([M_POWER.get(c, 4) for c in classes])
    linear = np.isin(classes, list(M_POWER))
    fsk = np.isin(classes, ["CPFSK", "GFSK"])
    # ОСШ — спектральным методом: M2M4 при высоком ОСШ «взрывается» от малейшей ошибки ka
    # (tools/m2m4_check.py: ka 1,2 вместо 1,216 → 16,8 дБ вместо 20; ka 1,265 → 60 дБ); M2M4 — отдельной оценкой
    snr = snr_spectral(z)
    snr_m2 = snr_m2m4(z, ka)
    cfo = np.where(linear, cfo_xm(z, m), cfo_centroid(z))
    rate = np.where(fsk, rate_freqdiff(z), rate_envelope(z))
    bw = bandwidth99(z)
    return {"snr_db": snr, "snr_db_m2m4": snr_m2, "cfo": cfo, "symbol_rate": rate, "bw99": bw, "rolloff": rolloff_from(bw, rate),
            "fsk_shift": fsk_shift(z)}
