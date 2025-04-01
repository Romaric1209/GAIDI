# Stage 1: Builder for heavy dependencies
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# System dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies first
COPY requirements.txt .
RUN pip install --user --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    tensorflow-cpu==2.18.0 \
    xgboost==2.1.4 \
    && pip install --user --no-cache-dir -r requirements.txt

# ----------------------------

# Stage 2: Minimal runtime image
FROM python:3.10-slim-buster

WORKDIR /app

# Copy only necessary artifacts
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app
COPY notebooks/transformers.py ./notebooks/
COPY notebooks/functions.py ./notebooks/
COPY notebooks/roma_NTLK_Data_Cache /root/nltk_data

# Application files
COPY romapi/app.py romapi/fast.py romapi/__init__.py ./romapi/
COPY roma_models/XGBoost_model.joblib roma_models/pipeline.joblib roma_models/image_model.keras ./roma_models/

# NLTK data setup
RUN python -m nltk.downloader punkt stopwords && \
    rm -rf /root/nltk_data/corpora/webtext/ /root/nltk_data/tokenizers/

# Environment setup
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV TF_CPP_MIN_LOG_LEVEL=3

# Cleanup
RUN find /usr/local/lib/python3.10 -type d -name __pycache__ -exec rm -rf {} + && \
    rm -rf /root/.cache/pip

EXPOSE 8080
CMD ["uvicorn", "romapi.fast:app", "--host", "0.0.0.0", "--port", "8080"]
