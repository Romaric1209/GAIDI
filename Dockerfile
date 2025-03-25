FROM python:3.10.6-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "romapp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
