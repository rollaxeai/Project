# Продукт: REST API (по умолчанию) или веб-интерфейс Streamlit.
#   docker build -t amc .
#   docker run -p 8000:8000 amc                                   # API, Swagger: http://localhost:8000/docs
#   docker run -p 8501:8501 amc streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
FROM python:3.12-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ src/
COPY app/ app/
COPY models/onnx/ models/onnx/
ENV PYTHONPATH=/srv/src:/srv
EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
