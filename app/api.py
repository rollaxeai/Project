"""REST API распознавания вида модуляции и параметров сигнала (FastAPI).

    uvicorn app.api:app --host 0.0.0.0 --port 8000

POST /analyze — WAV-файл (multipart, поле ``file``) → JSON с видом модуляции, вероятностями и параметрами.
GET /health — проверка работоспособности. Документация Swagger — /docs.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from amc import __version__
from amc.infer import Classifier, analyze

app = FastAPI(title="AMC: распознавание вида модуляции", version=__version__,
              description="Вид модуляции (11 классов) и параметры сигнала (ОСШ, несущая, полоса, скорость, "
                          "roll-off, разнос ЧМн) по WAV-записи (моно — вещественный сигнал, стерео — I/Q).")
_clf: Classifier | None = None
MAX_BYTES = 200 * 2**20


def classifier() -> Classifier:
    """Модель загружается один раз при первом запросе."""
    global _clf
    if _clf is None:
        _clf = Classifier()
    return _clf


@app.get("/health")
def health() -> dict:
    """Статус сервиса."""
    return {"status": "ok", "version": __version__}


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)) -> dict:
    """Анализ загруженного WAV-файла."""
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "файл больше 200 МБ")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "signal.wav"
        p.write_bytes(data)
        try:
            r = analyze(p, classifier())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"не удалось обработать файл: {e}") from e
    r["file"] = file.filename
    return r
