FROM python:3.10-slim-buster AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    tensorflow-cpu==2.18.0 \
    && pip install --user --no-cache-dir -r requirements.txt


FROM python:3.10-slim-buster

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY notebooks/transformers.py ./notebooks/
COPY notebooks/functions.py ./notebooks/
COPY romapi/app.py romapi/fast.py romapi/__init__.py ./romapi/
COPY roma_models/stacking_text_model.joblib roma_models/pipeline.joblib roma_models/image_model.keras ./roma_models/
COPY streamlit/static/ /app/streamlit/static/
COPY streamlit/ /app/streamlit
RUN rm -rf /root/nltk_data && \
    python -m nltk.downloader cmudict stopwords wordnet punkt punkt_tab


ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV NLTK_DATA=/root/nltk_data
ENV BASE_URL=https://gaidi1v3-2099061347.europe-west1.run.app

RUN find /usr/local/lib/python3.10 -type d -name __pycache__ -exec rm -rf {} + && \
    rm -rf /root/.cache/pip

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN chmod -R 755 /app/streamlit/static
RUN streamlit config show

EXPOSE 8080

CMD ["streamlit", "run", "streamlit/app.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.enableStaticServing=true", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.maxUploadSize=200", \
    "--browser.gatherUsageStats=false"]
