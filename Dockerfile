# Build Stage
FROM python:3.10-slim-buster AS builder

WORKDIR /app

# Install necessary build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y nginx

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --user --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    tensorflow-cpu==2.18.0 && \
    pip install --user --no-cache-dir -r requirements.txt


# Final Image Setup
FROM python:3.10-slim-buster

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV NLTK_DATA=/root/nltk_data
ENV PYTHONPATH="/app/notebooks:/app"

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy source files and models
COPY romapi/ ./romapi/
COPY notebooks/ ./notebooks/
COPY roma_models/stacking_text_model.joblib roma_models/pipeline.joblib roma_models/image_model.keras ./roma_models/
COPY streamlit/ ./streamlit/

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 libgl1 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

RUN rm -rf /root/nltk_data && \
    python -m nltk.downloader cmudict stopwords wordnet punkt punkt_tab

# Ensure correct permissions for static files
RUN chmod -R 755 /app/streamlit/static

# Add nginx config
COPY nginx.conf /etc/nginx/nginx.conf

COPY start.sh /start.sh
RUN chmod +x /start.sh


# Add a script to run both services
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose the Streamlit port
EXPOSE 8080

# Start the Streamlit app
CMD ["streamlit", "run", "streamlit/app.py", "/bin/bash", "/start.sh", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.enableStaticServing=true", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.maxUploadSize=200", \
    "--browser.gatherUsageStats=false"]
