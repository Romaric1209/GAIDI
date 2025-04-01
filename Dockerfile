FROM python:3.10.6-buster

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY . /app
COPY roma_models/XGBoost_model.joblib .
COPY roma_models/pipeline.joblib .
COPY roma_models/image_model.keras .
COPY romapi/app.py .
COPY romapi/fast.py .
COPY romapi/__init__.py .


RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "romapi.fast:app", "--host", "0.0.0.0", "--port", "8000"]
