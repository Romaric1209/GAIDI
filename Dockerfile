FROM python:3.10-slim-buster

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgomp1 \
    libgl1 \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy application files
COPY streamlit/ ./streamlit/

ENV PYTHONPATH="/app/streamlit/notebooks"

EXPOSE 8080

CMD ["streamlit", "run", "streamlit/app.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0"]