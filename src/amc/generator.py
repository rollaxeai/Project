"""Собственный генератор радиосигналов с известными параметрами.

Генерирует те же 11 классов, что и RadioML 2016.10a, но:

* все параметры сигнала и канала случайны и **записываются в метаданные** — это метки для регрессии
  (ОСШ, сдвиг частоты, символьная скорость, roll-off, ширина полосы, разнос частот ЧМн и т.д.);
* AM-SSB — настоящий однополосный сигнал (в RadioML 2016.10a этот класс — чистый шум, см. 01_eda);
* модель канала: многолучёвость с замираниями (Рэлей/Райс, медленная доплеровская вариация),
  ошибка частоты дискретизации, фазовый шум, сдвиг частоты, AWGN.

Сигнал строится в непрерывном времени: символы накладываются формирующим импульсом, вычисленным
в моменты отсчётов, поэтому число отсчётов на символ может быть дробным, а ошибка частоты
дискретизации и задержка — произвольными.

Все частоты нормированы на частоту дискретизации (циклов на отсчёт): f/fs.
ОСШ — отношение мощности сигнала к мощности шума во всей полосе fs (как в RadioML).

Запуск::

    python -m amc.generator --out data/synth/v1 --per-class 20000 --len 1024 --seed 2026
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps_signal
from scipy.special import erf

CLASSES: list[str] = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK",
                      "WBFM"]  # тот же порядок, что в RadioML (по алфавиту)
LINEAR = {"BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64"}
FSK = {"CPFSK", "GFSK"}
ANALOG = {"AM-DSB", "AM-SSB", "WBFM"}
PAD = 128  # запас отсчётов с каждой стороны (переходные процессы фильтров и канала), обрезается


@dataclass(frozen=True)
class GenConfig:
    """Диапазоны случайных параметров генератора (равномерные, если не сказано иное)."""

    n: int = 1024                                   # длина примера, отсчётов
    snr_db: tuple[float, float] = (-20.0, 20.0)     # ОСШ во всей полосе fs
    sps: tuple[float, float] = (4.0, 16.0)          # отсчётов на символ (лог-равномерно)
    rolloff: tuple[float, float] = (0.2, 0.5)       # RRC
    rrc_span: int = 8                               # полуширина импульса RRC, символов
    cfo: float = 0.05                               # |сдвиг частоты| ≤ cfo, циклов/отсчёт
    sro_ppm: float = 100.0                          # ошибка частоты дискретизации, ppm
    fsk_h: tuple[float, float] = (0.4, 1.0)         # индекс модуляции ЧМн
    gfsk_bt: tuple[float, float] = (0.3, 0.5)       # BT гауссова фильтра
    # audio_bw: полосы аналоговых и цифровых классов перекрываются (иначе полоса — обходной признак, см. 00_generator)
    audio_bw: tuple[float, float] = (0.01, 0.12)  # верхняя частота модулирующего сигнала, f/fs
    am_index: tuple[float, float] = (0.3, 1.0)      # глубина АМ
    fm_dev: tuple[float, float] = (0.03, 0.12)      # девиация ЧМ, f/fs
    p_multipath: float = 0.6                        # вероятность многолучёвости
    max_paths: int = 4
    max_delay: float = 1.5                          # макс. задержка луча, символов (для аналоговых — 8 отсч.)
    doppler: tuple[float, float] = (0.0, 2e-4)      # доплеровское рассеяние, f/fs
    p_rician: float = 0.5                           # доля райсовских каналов (иначе Рэлей)
    rician_k_db: tuple[float, float] = (0.0, 10.0)
    phase_noise: tuple[float, float] = (0.0, 1e-4)  # дисперсия приращения фазы за отсчёт, рад²


# ---------------------------------------------------------------- созвездия и импульсы

def constellation(name: str) -> np.ndarray:
    """Точки созвездия с единичной средней мощностью."""
    if name == "BPSK":
        c = np.array([-1, 1], dtype=complex)
    elif name == "QPSK":
        c = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    elif name == "8PSK":
        c = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    elif name == "PAM4":
        c = np.array([-3, -1, 1, 3], dtype=complex)
    elif name in ("QAM16", "QAM64"):
        m = 4 if name == "QAM16" else 8
        lv = np.arange(-(m - 1), m, 2)
        c = (lv[:, None] + 1j * lv[None, :]).ravel()
    else:
        raise ValueError(name)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def rrc(t: np.ndarray, beta: float) -> np.ndarray:
    """Импульс «корень из приподнятого косинуса», T = 1, в произвольные моменты t."""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    eps = 1e-8
    z = np.abs(t) < eps
    s = np.abs(np.abs(t) - 1 / (4 * beta)) < eps
    g = ~(z | s)
    tg = t[g]
    out[g] = (np.sin(np.pi * tg * (1 - beta)) + 4 * beta * tg * np.cos(np.pi * tg * (1 + beta))) / (
        np.pi * tg * (1 - (4 * beta * tg) ** 2))
    out[z] = 1 - beta + 4 * beta / np.pi
    out[s] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                  + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    return out


def gauss_freq_pulse(t: np.ndarray, bt: float) -> np.ndarray:
    """Частотный импульс GFSK (прямоугольник длительностью T, сглаженный гауссом), интеграл = 1."""
    a = 2 * np.pi * bt / np.sqrt(np.log(2))
    return 0.5 * (erf(a * (t + 0.5) / np.sqrt(2)) - erf(a * (t - 0.5) / np.sqrt(2)))


def _symbol_times(n_total: int, sps: float, sro: float, tau: float) -> np.ndarray:
    """Моменты отсчётов в единицах символьного интервала (с ошибкой частоты дискретизации и задержкой)."""
    return (np.arange(n_total) * (1 + sro) + tau) / sps


def _superpose(t: np.ndarray, symbols_fn, pulse, span: int, rng: np.random.Generator) -> np.ndarray:
    """s(t) = Σ_k a_k p(t − k) в моменты t; a_k — случайные символы."""
    k0 = int(np.floor(t[0])) - span
    k1 = int(np.ceil(t[-1])) + span
    a = symbols_fn(k1 - k0 + 1)
    base = np.floor(t).astype(int)
    offs = np.arange(-span, span + 1)
    k = base[:, None] + offs[None, :]
    return np.sum(a[k - k0] * pulse(t[:, None] - k), axis=1)


# ---------------------------------------------------------------- модулирующий сигнал (аналоговые классы)

def audio(n_total: int, bw: float, rng: np.random.Generator) -> np.ndarray:
    """Модулирующий сигнал речевого/музыкального типа, |a| ≤ 1, полоса ≈ (0.05·bw, bw).

    «Речь» — окрашенный шум с огибающей слогового темпа; «музыка» — сумма тонов с гармониками.
    Пауз нет: в RadioML паузы делают AM-DSB/WBFM неотличимыми от несущей (см. 01_eda).
    """
    warm = 512
    m = n_total + warm
    if rng.random() < 0.5:
        sos = sps_signal.butter(4, [0.05 * bw * 2, bw * 2], btype="band", output="sos")
        a = sps_signal.sosfilt(sos, rng.standard_normal(m))[warm:]
        syl = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.5, 3) / n_total * np.arange(n_total)
                                 + rng.uniform(0, 2 * np.pi))
        a = a * syl
    else:
        nn = np.arange(n_total)
        a = np.zeros(n_total)
        for _ in range(rng.integers(1, 4)):
            f0 = rng.uniform(0.1 * bw, 0.5 * bw)
            for h in range(1, 4):
                if h * f0 < bw:
                    a += rng.uniform(0.2, 1) / h * np.cos(2 * np.pi * h * f0 * nn + rng.uniform(0, 2 * np.pi))
    a = a - a.mean()
    return a / (np.max(np.abs(a)) + 1e-12)


# ---------------------------------------------------------------- модуляторы

def modulate(cls: str, n_total: int, rng: np.random.Generator, cfg: GenConfig, sro: float) -> tuple[np.ndarray, dict]:
    """Чистый (без канала) комплексный сигнал класса cls и его параметры."""
    p: dict = {"sps": np.nan, "symbol_rate": np.nan, "rolloff": np.nan, "fsk_h": np.nan, "fsk_shift": np.nan,
               "gfsk_bt": np.nan, "audio_bw": np.nan, "am_index": np.nan, "fm_dev": np.nan, "sideband": ""}
    if cls in LINEAR or cls in FSK:
        sps = float(np.exp(rng.uniform(np.log(cfg.sps[0]), np.log(cfg.sps[1]))))
        tau = rng.uniform(0, sps)
        t = _symbol_times(n_total, sps, sro, tau)
        p.update(sps=sps, symbol_rate=1 / sps)
    if cls in LINEAR:
        beta = rng.uniform(*cfg.rolloff)
        c = constellation(cls)
        x = _superpose(t, lambda k: c[rng.integers(0, len(c), k)], lambda u: rrc(u, beta), cfg.rrc_span, rng)
        p.update(rolloff=beta)
    elif cls in FSK:
        h = rng.uniform(*cfg.fsk_h)
        if cls == "CPFSK":
            def pulse(u):
                return ((u >= -0.5) & (u < 0.5)).astype(float)
            span = 1
        else:
            bt = rng.uniform(*cfg.gfsk_bt)
            p.update(gfsk_bt=bt)

            def pulse(u):
                return gauss_freq_pulse(u, bt)
            span = 3
        f = _superpose(t, lambda k: rng.choice([-1.0, 1.0], k), pulse, span, rng) * (h / 2) / sps
        x = np.exp(1j * 2 * np.pi * np.cumsum(f))
        p.update(fsk_h=h, fsk_shift=h / sps)
    elif cls in ANALOG:
        bw = rng.uniform(*cfg.audio_bw)
        a = audio(n_total, bw, rng)
        p.update(audio_bw=bw)
        if cls == "AM-DSB":
            mi = rng.uniform(*cfg.am_index)
            x = (1 + mi * a).astype(complex)
            p.update(am_index=mi)
        elif cls == "AM-SSB":
            xa = sps_signal.hilbert(a)
            sb = "USB" if rng.random() < 0.5 else "LSB"
            x = xa if sb == "USB" else np.conj(xa)
            p.update(sideband=sb)
        else:  # WBFM
            dev = rng.uniform(*cfg.fm_dev)
            x = np.exp(1j * 2 * np.pi * dev * np.cumsum(a))
            p.update(fm_dev=dev)
    else:
        raise ValueError(cls)
    return x, p


def occupied_bw(x: np.ndarray, frac: float = 0.99) -> float:
    """Ширина полосы, содержащей долю frac мощности (f/fs), по периодограмме Уэлча."""
    f, pxx = sps_signal.welch(x, fs=1.0, nperseg=256, return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    c = np.cumsum(pxx) / np.sum(pxx)
    lo = f[np.searchsorted(c, (1 - frac) / 2)]
    hi = f[min(np.searchsorted(c, 1 - (1 - frac) / 2), len(f) - 1)]
    return float(hi - lo)


# ---------------------------------------------------------------- канал

def _fading_gain(n_total: int, fd: float, rng: np.random.Generator, n_sin: int = 8) -> np.ndarray:
    """Комплексный рэлеевский коэффициент, меняющийся во времени (модель суммы синусоид), E|g|² = 1."""
    nn = np.arange(n_total)
    theta = rng.uniform(0, 2 * np.pi, n_sin)
    phi = rng.uniform(0, 2 * np.pi, n_sin)
    return np.sum(np.exp(1j * (2 * np.pi * fd * np.cos(theta)[:, None] * nn + phi[:, None])), axis=0) / np.sqrt(n_sin)


def _frac_delay(x: np.ndarray, d: float) -> np.ndarray:
    """Задержка на d отсчётов (дробная) через фазовый сдвиг в частотной области."""
    n = len(x)
    f = np.fft.fftfreq(n)
    return np.fft.ifft(np.fft.fft(x) * np.exp(-1j * 2 * np.pi * f * d))


def channel(x: np.ndarray, sym_len: float, rng: np.random.Generator, cfg: GenConfig) -> tuple[np.ndarray, dict]:
    """Многолучёвость с замираниями + фазовый шум + сдвиг частоты (без шума). Возвращает y и параметры."""
    n_total = len(x)
    p = {"multipath": False, "n_paths": 1, "delay_spread": 0.0, "fading": "none", "rician_k_db": np.nan,
         "doppler": 0.0}
    if rng.random() < cfg.p_multipath:
        L = int(rng.integers(2, cfg.max_paths + 1))
        delays = np.sort(np.concatenate([[0.0], rng.uniform(0, cfg.max_delay * sym_len, L - 1)]))
        pw = np.exp(-delays / (0.5 * cfg.max_delay * sym_len + 1e-9))
        pw /= pw.sum()
        fd = rng.uniform(*cfg.doppler)
        rician = rng.random() < cfg.p_rician
        kdb = rng.uniform(*cfg.rician_k_db) if rician else np.nan
        y = np.zeros(n_total, dtype=complex)
        for i, (d, w) in enumerate(zip(delays, pw, strict=True)):
            g = _fading_gain(n_total, fd, rng)
            if i == 0 and rician:
                k = 10 ** (kdb / 10)
                g = np.sqrt(k / (k + 1)) * np.exp(1j * rng.uniform(0, 2 * np.pi)) + np.sqrt(1 / (k + 1)) * g
            y += np.sqrt(w) * g * _frac_delay(x, d)
        x = y
        p.update(multipath=True, n_paths=L, delay_spread=float(np.sqrt(np.sum(pw * delays**2) - np.sum(pw * delays) ** 2)),
                 fading="rician" if rician else "rayleigh", rician_k_db=kdb, doppler=fd)
    pn = rng.uniform(*cfg.phase_noise)
    cfo = rng.uniform(-cfg.cfo, cfg.cfo)
    phase0 = rng.uniform(0, 2 * np.pi)
    ph = np.cumsum(rng.standard_normal(n_total) * np.sqrt(pn)) + 2 * np.pi * cfo * np.arange(n_total) + phase0
    p.update(phase_noise=pn, cfo=cfo, phase0=phase0)
    return x * np.exp(1j * ph), p


# ---------------------------------------------------------------- один пример

def generate_one(cls: str, rng: np.random.Generator, cfg: GenConfig = GenConfig(),
                 return_clean: bool = False) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """Один пример: комплексный сигнал длины cfg.n с единичной средней мощностью и словарь меток."""
    n_total = cfg.n + 2 * PAD
    sro = rng.uniform(-cfg.sro_ppm, cfg.sro_ppm) * 1e-6
    x, meta = modulate(cls, n_total, rng, cfg, sro)
    bw = occupied_bw(x[PAD:-PAD])
    sym_len = meta["sps"] if cls not in ANALOG else 8.0
    y, chp = channel(x, sym_len, rng, cfg)
    y = y[PAD:-PAD]
    clean = y / np.sqrt(np.mean(np.abs(y) ** 2))
    snr_db = rng.uniform(*cfg.snr_db)
    noise = (rng.standard_normal(cfg.n) + 1j * rng.standard_normal(cfg.n)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    out = clean + noise
    out = out / np.sqrt(np.mean(np.abs(out) ** 2))
    meta.update(chp)
    meta.update(cls=cls, label=CLASSES.index(cls), snr_db=snr_db, bw99=bw, sro_ppm=sro * 1e6,
                esn0_db=snr_db + 10 * np.log10(meta["sps"]) if cls not in ANALOG else np.nan)
    if return_clean:
        return out.astype(np.complex64), meta, clean.astype(np.complex64)
    return out.astype(np.complex64), meta


# ---------------------------------------------------------------- датасет

def _chunk(args: tuple) -> tuple[int, np.ndarray, list[dict]]:
    """Сгенерировать кусок датасета (выполняется в отдельном процессе)."""
    start, cls_list, seed_entropy, chunk_id, cfg_dict = args
    cfg = GenConfig(**cfg_dict)
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy, spawn_key=(chunk_id,)))
    X = np.empty((len(cls_list), 2, cfg.n), dtype=np.float32)
    metas = []
    for i, c in enumerate(cls_list):
        z, m = generate_one(c, rng, cfg)
        X[i, 0], X[i, 1] = z.real, z.imag
        metas.append(m)
    return start, X, metas


def _low_priority() -> None:
    """Пониженный приоритет рабочих процессов (Windows: BELOW_NORMAL), чтобы ПК не «тормозил»."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    else:
        os.nice(5)


def available_memory_gb() -> float:
    """Свободная физическая память, ГБ (Windows API / /proc/meminfo); при ошибке — 4 ГБ."""
    try:
        if os.name == "nt":
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("p", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("t", "a", "tp", "ap", "tv", "av", "e")]

            m = _Mem()
            m.l = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.a / 2**30
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 2**20
    except Exception:  # noqa: BLE001
        pass
    return 4.0


def memory_limited_workers(per_worker_gb: float = 0.4, reserve_gb: float = 3.0) -> int:
    """Сколько процессов помещается в свободную память (≈0,4 ГБ на процесс + запас системе)."""
    return max(1, int((available_memory_gb() - reserve_gb) / per_worker_gb))


def default_workers() -> int:
    """Ядра минус max(1, четверть)."""
    n = os.cpu_count() or 1
    return max(1, n - max(1, n // 4))


def make_dataset(out: str | Path, per_class: int, cfg: GenConfig, seed: int, workers: int,
                 chunk: int = 500, split: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> None:
    """Сгенерировать датасет: X.npy [N, 2, n] float32, meta.parquet (метки и параметры, колонка split)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(CLASSES)), per_class)
    rng.shuffle(labels)
    cls_seq = [CLASSES[i] for i in labels]
    N = len(cls_seq)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(N, 2, cfg.n))
    metas: list[dict | None] = [None] * N
    tasks = [(s, cls_seq[s:s + chunk], seed, s // chunk, asdict(cfg)) for s in range(0, N, chunk)]
    t0 = time.time()
    done = 0
    # OpenBLAS по умолчанию резервирует буферы на все ядра — ≈1,6 ГБ commit на каждый процесс
    # (измерено tools/peakmem.py); при 18 процессах память кончалась. Рабочим процессам — 1 поток BLAS.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    workers = min(workers, memory_limited_workers())
    print(f"процессов: {workers}, кусков: {len(tasks)}")
    pending = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, initializer=_low_priority) as ex:
        # в работе не больше 2×workers кусков — результаты не накапливаются в памяти
        running = {ex.submit(_chunk, t) for t in itertools.islice(pending, 2 * workers)}
        while running:
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                s, xb, mb = f.result()
                X[s:s + len(xb)] = xb
                metas[s:s + len(mb)] = mb
                done += len(xb)
                del xb
            for t in itertools.islice(pending, len(finished)):
                running.add(ex.submit(_chunk, t))
            el = time.time() - t0
            print(f"\r{done}/{N} ({100 * done / N:.1f} %), прошло {el:.0f} с, осталось ≈ {el / done * (N - done):.0f} с",
                  end="", flush=True)
    print()
    X.flush()
    meta = pd.DataFrame(metas)
    # стратифицированное разбиение по (класс × интервал ОСШ)
    meta["snr_bin"] = (np.floor(meta["snr_db"] / 4) * 4).astype(int)
    meta["split"] = ""
    for _, idx in meta.groupby(["label", "snr_bin"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        a = int(round(split[0] * len(idx)))
        b = a + int(round(split[1] * len(idx)))
        meta.loc[idx[:a], "split"] = "train"
        meta.loc[idx[a:b], "split"] = "val"
        meta.loc[idx[b:], "split"] = "test"
    meta.to_parquet(out / "meta.parquet")
    (out / "config.txt").write_text(f"seed={seed}\nper_class={per_class}\n{asdict(cfg)}\n", encoding="utf-8")
    print(f"готово: {N} примеров → {out}")


def main() -> None:
    """Командная строка: генерация датасета."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=20000)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=default_workers())
    a = ap.parse_args()
    make_dataset(a.out, a.per_class, GenConfig(n=a.len), a.seed, a.threads)


if __name__ == "__main__":
    main()
