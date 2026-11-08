"""Веб-интерфейс: загрузка WAV → вид модуляции, вероятности, параметры, спектр.

streamlit run app/streamlit_app.py
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal as sps

from amc import __version__
from amc.infer import Classifier, analyze, read_signal

st.set_page_config(page_title="AMC — вид модуляции", layout="wide")
st.title("Распознавание вида модуляции и параметров радиосигнала")
st.caption(f"amc {__version__} · многозадачная сеть ResNet-1D (ONNX) + классическая ЦОС · "
           "WAV: моно — вещественный сигнал, стерео — I/Q")


@st.cache_resource
def model() -> Classifier:
    """Модель загружается один раз."""
    return Classifier()


up = st.file_uploader("WAV-файл", type=["wav"])
if up is not None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signal.wav"
        p.write_bytes(up.getvalue())
        with st.spinner("анализ…"):
            r = analyze(p, model())
            x, fs = read_signal(p)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Вид модуляции", r["modulation"], f"уверенность {r['confidence']:.0%}")
        params = {"ОСШ в полосе сигнала, дБ": r["snr_inband_dB"], "Несущая, Гц": r["carrier_Hz"],
                  "Полоса 99 %, Гц": r["bandwidth99_Hz"], "Символьная скорость, Бод": r["symbol_rate_Bd"],
                  "Roll-off": r["rolloff"], "Разнос ЧМн, Гц": r["fsk_shift_Hz"]}
        st.table(pd.DataFrame({"значение": {k: v for k, v in params.items() if v is not None}}))
        st.caption(f"Источник оценок: {r['method']}; окон: {r['windows']}; fs записи {r['fs_Hz']:.0f} Гц, "
                   f"fs обработки {r['processing_fs_Hz']:.0f} Гц")
    with c2:
        st.bar_chart(pd.Series(r["probabilities"], name="вероятность"))
        z = x if np.iscomplexobj(x) else sps.hilbert(x)
        f, psd = sps.welch(z, fs=fs, nperseg=min(4096, len(z)), return_onesided=False)
        o = np.argsort(f)
        st.line_chart(pd.DataFrame({"СПМ, дБ": 10 * np.log10(psd[o] + 1e-20)}, index=f[o]), x_label="частота, Гц")
    with st.expander("JSON"):
        st.json(r)
