"""Тест REST API (TestClient FastAPI) на сгенерированном WAV."""
import io
from pathlib import Path

import numpy as np
import pytest

from amc.infer import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="нет ONNX-модели")


def test_health_and_analyze():
    sf = pytest.importorskip("soundfile")
    from fastapi.testclient import TestClient

    from app.api import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    t = np.arange(48000) / 48000
    y = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * 300 * t)) * np.cos(2 * np.pi * 9000 * t)  # АМ, вещественный
    y += 0.01 * np.random.default_rng(0).standard_normal(len(y))
    buf = io.BytesIO()
    sf.write(buf, y, 48000, format="WAV", subtype="PCM_16")
    r = c.post("/analyze", files={"file": ("am.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200
    j = r.json()
    assert j["file"] == "am.wav" and abs(j["carrier_Hz"] - 9000) < 50 and j["modulation"] in {"AM-DSB", "AM-SSB", "WBFM"}
    assert c.post("/analyze", files={"file": ("x.wav", b"not a wav", "audio/wav")}).status_code == 400


def test_streamlit_renders():
    at_mod = pytest.importorskip("streamlit.testing.v1")
    at = at_mod.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert "модуляции" in at.title[0].value
