FROM python:3.10.6-buster

WORKDIR /app

COPY . /app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
CMD uvicorn romapp.api.fast:app --reload --host 0.0.0.0 --port $PORT
