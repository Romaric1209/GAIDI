# Build Stage
FROM python:3.10-slim-buster AS builder

WORKDIR /app


# Copy requirements.txt and install dependencies
COPY requirements.txt ./
#COPY nginx.conf ./
RUN pip install --no-cache-dir \
    numpy==1.26.0 \
    scipy==1.13.1 \
    pandas==2.2.3 \
    tensorflow-cpu==2.18.0 && \
    pip install --no-cache-dir -r requirements.txt


# Final Image Setup
FROM python:3.10-slim-buster

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app/notebooks:/app"

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy source files and models
COPY notebooks/functions.py notebooks/text_transformers.py notebooks/pipeline.py ./notebooks/
COPY models/xgb_model.joblib models/pipeline.joblib models/image_model.keras ./models/
RUN ls -lah /app/models/
COPY --chmod=755 streamlit/ ./streamlit/
RUN chown -R www-data:www-data /app/streamlit/static

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # nginx \
    libgomp1 libgl1 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip


# Expose the Streamlit port
EXPOSE 8080

# Start the Streamlit app
CMD ["streamlit", "run", "streamlit/app.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.enableStaticServing=true", \
    "--server.headless=true", \
    "--server.enableCORS=false", \
    "--server.maxUploadSize=200", \
    "--browser.gatherUsageStats=false"]
