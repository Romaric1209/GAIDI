from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
from io import BytesIO
from PIL import Image

app = FastAPI()

text_preprocessing = joblib.load("notebooks/roma_pipepline.joblib")
model_texts=tf.keras.models.load_model("notebooks/roma_models/baseline_model.keras")
model_images=tf.keras.models.load_model("notebooks/roma_models/baseline_images.keras")


@app.get("/")
def read_root():
    return {"message": "Welcome to GAIDI!"}


@app.get("/predict_text")
def predict_text(input_data:TextInput):
    try:
        preprocessed_input = text_preprocessing.transform([input_data.text])
        prediction = model_texts.predict(preprocessed_input)

        label = "YOU TYPED IT!" if prediction > 0.5 else "GENAI WROTE THAT"
        return {"result": label, "confidence": round(float(prediction[0][0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
