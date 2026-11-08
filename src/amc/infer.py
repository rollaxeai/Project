"""Распознавание вида модуляции и параметров сигнала в WAV/I/Q-файле (продукт, этап 6).

Конвейер:

1. **Чтение** WAV любого формата (PCM 8/16/24/32, float 32/64, WAVE_EXTENSIBLE, RF64 — через soundfile):
   моно → вещественный сигнал, 2 канала → I/Q.
2. **Перенос на нулевую частоту и передискретизация.** Модель обучена на сигналах, занимающих 7–40 % полосы fs
   (4–16 отсчётов на символ). Поэтому по спектру Уэлча находятся центр и полоса сигнала; сигнал переносится
   в ноль и передискретизируется так, чтобы полоса занимала ≈ 15 % новой частоты дискретизации.
   Поиск масштаба: если сеть сама оценивает занятость полосы вне обучающего диапазона, проверяются также
   fs обработки ×4 и ÷4 (оценка полосы у АМ может ошибаться в разы) и берётся самосогласованный вариант.
   Вещественный сигнал предварительно превращается в аналитический (преобразование Гильберта).
3. **Окна** по 1024 отсчёта с шагом 512 → многозадачная сеть (ONNX Runtime): вероятности классов и параметры.
4. **Агрегация:** вероятности усредняются по окнам, параметры — медиана по окнам.
5. **Гибрид с ЦОС** (по результатам 03b_parameters, выбор на val): ОСШ, сдвиг частоты, скорость и полоса берутся
   у классических оценщиков, если ОСШ по оценке сети выше порога для данного параметра (HYBRID_SNR), иначе —
   у сети (полоса ЦОС — по спектру всего файла); roll-off и разнос ЧМн — всегда сеть.
6. **Перевод в физические единицы** с учётом новой частоты дискретизации и сдвига центра: Гц, Бод.

Запуск::

    python -m amc.infer файл.wav [файл2.wav папка ...] [--model models/onnx/amc_multitask.onnx]
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sps

from amc import estimators as est
from amc.generator import CLASSES, LINEAR

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "onnx" / "amc_multitask.onnx"
WIN, HOP = 1024, 512
# ЦОС используется, если ОСШ по оценке сети выше порога (дБ, в полосе обработки); иначе — нейросеть.
# Пороги — из выбора на val (03b_parameters, п. 4): ОСШ — ЦОС от −10 дБ, сдвиг и полоса — от 0, скорость — от 10.
HYBRID_SNR = {"snr": -10.0, "cfo": 0.0, "bw": 0.0, "rate": 10.0}
RATE_AGREE = 1.25  # ниже порога скорость ЦОС принимается, если отличается от оценки сети не более чем в 1,25 раза
TARGET_OCC = 0.15  # доля полосы сигнала от новой fs (медиана обучающего датасета ≈ 0,15)


def read_signal(path: str | Path) -> tuple[np.ndarray, float]:
    """Прочитать WAV: (сигнал, fs). Моно — вещественный, 2 канала — комплексный I + jQ."""
    import soundfile as sf

    data, fs = sf.read(str(path), dtype="float64", always_2d=True)
    if data.shape[1] >= 2:
        return data[:, 0] + 1j * data[:, 1], float(fs)
    return data[:, 0], float(fs)


def find_signal(x: np.ndarray, fs: float) -> tuple[np.ndarray, float, float, float]:
    """Аналитический сигнал, центр и полоса сигнала (Гц) по спектру Уэлча всего файла, разрешение спектра (Гц)."""
    real = not np.iscomplexobj(x)
    if real:
        x = sps.hilbert(x - x.mean())
    x = x - x.mean()
    nper = min(4096, 1 << int(np.log2(max(len(x) // 8, 256))))
    f, p = sps.welch(x, fs=fs, nperseg=nper, return_onesided=False, detrend=False)
    order = np.argsort(f)
    f, p = f[order], p[order]
    if real:
        # у аналитического сигнала отрицательные частоты пусты: без этого «пол шума» ≈ 0 и весь шум
        # положительной половины принимается за сигнал
        keep = (f > 2 * fs / nper) & (f < fs / 2 - 2 * fs / nper)
        f, p = f[keep], p[keep]
    p = sps.medfilt(p, 5)
    floor = np.sort(p)[: max(1, len(p) // 5)].mean() * 1.1
    # учитываются только бины выше 2·пола шума и не ниже −20 дБ от пика; пик — по спектру, сглаженному медианой
    # по 9 бинам, чтобы одиночная линия (несущая АМ) не задирала порог
    q = np.where(p > max(2 * floor, sps.medfilt(p, 9).max() / 100), p - floor, 0.0)
    if q.sum() <= 0:
        q = p
    c = np.cumsum(q) / q.sum()
    lo, hi = f[np.searchsorted(c, 0.01)], f[min(np.searchsorted(c, 0.99), len(f) - 1)]
    return x, float((lo + hi) / 2), float(max(hi - lo, fs / nper)), fs / nper


def shift_resample(x: np.ndarray, fs: float, center: float, fs_target: float) -> tuple[np.ndarray, float]:
    """Перенос центра в ноль и понижение частоты дискретизации до ≈ fs_target (не выше исходной)."""
    z = x * np.exp(-2j * np.pi * center / fs * np.arange(len(x)))
    ratio = Fraction(min(fs_target, fs) / fs).limit_denominator(1000)
    if 0 < ratio < 1:
        return sps.resample_poly(z, ratio.numerator, ratio.denominator), fs * ratio.numerator / ratio.denominator
    return z, fs


def to_baseband(x: np.ndarray, fs: float) -> tuple[np.ndarray, float, float, float]:
    """Перенос сигнала в ноль и передискретизация. Возвращает (z, новая fs, центр сигнала Гц, полоса Гц)."""
    xa, center, bw, _ = find_signal(x, fs)
    z, fs2 = shift_resample(xa, fs, center, bw / TARGET_OCC)
    return z, float(fs2), center, bw


def windows(z: np.ndarray, max_windows: int = 512) -> np.ndarray:
    """Окна [K, 2, 1024] с шагом 512 (если сигнал короче — дополняется нулями)."""
    if len(z) < WIN:
        z = np.pad(z, (0, WIN - len(z)))
    starts = np.arange(0, len(z) - WIN + 1, HOP)
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    w = np.stack([z[s:s + WIN] for s in starts])
    w = w / np.sqrt(np.mean(np.abs(w) ** 2, axis=1, keepdims=True) + 1e-20)
    return np.stack([w.real, w.imag], axis=1).astype(np.float32)


class Classifier:
    """Обёртка над ONNX-моделью: окна → (вероятности [K, 11], параметры [K, 6] в единицах целей)."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(model_path), providers=ort.get_available_providers())

    def __call__(self, w: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:  # noqa: D102
        ps, rs = [], []
        for s in range(0, len(w), batch):
            p, r = self.sess.run(None, {"iq": w[s:s + batch]})
            ps.append(p)
            rs.append(r)
        return np.concatenate(ps), np.concatenate(rs)


def analyze(path: str | Path, clf: Classifier | None = None) -> dict:
    """Полный анализ файла → словарь с видом модуляции, вероятностями и параметрами в физических единицах."""
    clf = clf or Classifier()
    x, fs = read_signal(path)
    xa, center, bw_est, res = find_signal(x, fs)
    # полоса ≤ 6 бинов спектра — видна только линия (несущая АМ, боковые в шуме): это не полоса сигнала,
    # поэтому масштаб ищется шире и проверка проводится всегда
    line = bw_est <= 6 * res
    mults = (1.0, 4.0, 16.0, 64.0) if line else (1.0, 4.0, 0.25)
    # поиск масштаба: оценка полосы может ошибаться в разы (АМ с малой глубиной: видна только несущая, или
    # боковые полосы тонут в шуме), а сеть обучена на занятости полосы 7–40 % fs. Критерий — самосогласованность:
    # регрессионная голова сети сама оценивает долю полосы; если на основном масштабе она вне [5 %; 45 %],
    # проверяются fs обработки ×4 и ÷4 (для «линии» — ×4, ×16, ×64) и берётся вариант, где оценка сети
    # ближе всего к 15 % (в логарифме).
    # (Выбор по максимуму уверенности проверен и отвергнут: сеть бывает уверенно неправа на чужом масштабе.)
    cands = []
    for mult in mults:
        z, fs2 = shift_resample(xa, fs, center, bw_est / TARGET_OCC * mult)
        if any(abs(fs2 - c[1]) < 1e-6 * fs for c in cands):
            continue
        w = windows(z)
        proba, reg = clf(w)
        occ = float(2.0 ** np.median(reg[:, 3]))
        cands.append((abs(np.log2(occ / TARGET_OCC)), fs2, w, proba, reg, mult))
        if mult == 1.0 and 0.05 <= occ <= 0.45 and not line:
            break
    best = min(cands, key=lambda c: c[0])
    _, fs2, w, proba, reg, scale = best
    pm = proba.mean(axis=0)
    k = int(pm.argmax())
    cls = CLASSES[k]
    snr_net, cfo_net, lrate, lbw, roll, lshift = np.median(reg, axis=0)
    zz = (w[:, 0] + 1j * w[:, 1]).astype(np.complex128)
    dsp = est.estimate_all(zz, [cls] * len(zz))
    # гибрид: метод для каждого параметра выбран на val в 03b_parameters (по интервалам ОСШ, оценённого сетью)
    src = {k: ("ЦОС" if snr_net > thr else "нейросеть") for k, thr in HYBRID_SNR.items()}
    use_dsp = src["bw"] == "ЦОС"
    snr = float(np.median(dsp["snr_db"])) if src["snr"] == "ЦОС" else float(snr_net)
    cfo = float(np.median(dsp["cfo"])) if src["cfo"] == "ЦОС" else float(cfo_net) / 100
    rate_net, rate_dsp = float(2.0 ** lrate), float(np.median(dsp["symbol_rate"]))
    if src["rate"] != "ЦОС" and abs(np.log2(max(rate_dsp, 1e-9) / rate_net)) < np.log2(RATE_AGREE):
        # каскад «сеть → ЦОС»: сеть даёт грубую оценку (±10 %), линия ЦОС её уточняет, если найдена рядом;
        # при расхождении > 25 % ЦОС, скорее всего, нашла не ту линию (многолучёвость, гармоники ЧМн)
        src["rate"] = "ЦОС (уточнение)"
    rate = rate_dsp if src["rate"].startswith("ЦОС") else rate_net
    # полоса при высоком ОСШ — по спектру всего файла (to_baseband): данных больше, чем в окне 1024
    if not use_dsp:
        bw = float(2.0 ** lbw)
    elif scale == 1.0:
        bw = bw_est / fs2
    else:  # оценка полосы по всему файлу признана ошибочной — берётся оценка ЦОС по окнам
        bw = float(np.median(dsp["bw99"]))
    digital = cls not in ("AM-DSB", "AM-SSB", "WBFM")
    out = {
        "file": str(path), "fs_Hz": fs, "duration_s": len(x) / fs, "windows": len(w),
        "modulation": cls, "confidence": float(pm[k]),
        "probabilities": {c: round(float(v), 4) for c, v in sorted(zip(CLASSES, pm, strict=True), key=lambda t: -t[1])},
        "window_votes": {c: int(np.sum(proba.argmax(1) == i)) for i, c in enumerate(CLASSES) if np.any(proba.argmax(1) == i)},
        "snr_dB": round(snr, 1),  # ОСШ во всей полосе обработки fs2 (так определён ОСШ при обучении)
        "snr_inband_dB": round(snr + 10 * np.log10(1 / max(bw, 1e-3)), 1),  # пересчёт в полосу сигнала (99 %)
        "carrier_Hz": round(center + cfo * fs2, 2),
        "bandwidth99_Hz": round(bw * fs2, 2),
        "symbol_rate_Bd": round(rate * fs2, 2) if digital else None,
        "rolloff": round(float(roll), 3) if cls in LINEAR else None,
        "fsk_shift_Hz": round(float(2.0 ** lshift) * fs2, 2) if cls in ("CPFSK", "GFSK") else None,
        "method": {**src, "rolloff": "нейросеть", "fsk_shift": "нейросеть"},
        "processing_fs_Hz": round(fs2, 3),
        "scale": scale,  # множитель fs обработки, выбранный поиском масштаба
    }
    return out


def collect(paths: list[str]) -> list[Path]:
    """Файлы и папки (рекурсивно *.wav)."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.wav")) if p.is_dir() else [p]
    return out


def main() -> None:
    """Командная строка: анализ WAV-файлов и папок."""
    ap = argparse.ArgumentParser(description="Вид модуляции и параметры сигнала в WAV-файлах")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    a = ap.parse_args()
    paths = a.paths or [input("Путь к файлу или папке: ").strip().strip('"')]
    clf = Classifier(a.model)
    for f in collect(paths):
        try:
            r = analyze(f, clf)
        except Exception as e:  # noqa: BLE001
            print(f"{f}: ошибка: {e}", file=sys.stderr)
            continue
        f.with_suffix(".amc.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{f.name}: {r['modulation']} ({r['confidence']:.0%}), ОСШ в полосе {r['snr_inband_dB']} дБ, "
              f"несущая {r['carrier_Hz']} Гц, полоса {r['bandwidth99_Hz']} Гц"
              + (f", скорость {r['symbol_rate_Bd']} Бод" if r["symbol_rate_Bd"] else ""))


if __name__ == "__main__":
    main()
