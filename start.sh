#!/bin/bash


service nginx start


streamlit run streamlit/app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.enableStaticServing=true \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.maxUploadSize=200 \
  --browser.gatherUsageStats=false
