from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello"}


@app.get("/predict")
def predict():
    return {"message": "Hello"}
