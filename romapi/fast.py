import uvicorn
import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from notebooks.transformers import (
    InputHandler, HowManyWords, TextPreprocessor, ConsDensity, Stress, Sentiment, Redundance,
    UnusualWord, Coherence, ReadingEase, GunningFog, LogTransform, Tfidf_Vectorizer
)
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from PIL import Image

app = FastAPI()

#Load the text model
text_preprocessing = joblib.load("roma_models/pipeline.joblib")
model_texts = joblib.load("roma_models/XGBoost_model.joblib")

#Load the image model
model_images = tf.keras.models.load_model("roma_models/image_model.keras")

#Define request for text input
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to GAIDI!"}

@app.post("/predict_text")
def predict_text(input_data:TextInput):
    try:
        input_df = pd.DataFrame({"text": [input_data.text]})
        if input_df["text"].str.split().str.len().iloc[0] == 0:
            return {"error": "No text provided."}
        if input_df["text"].str.split().str.len().iloc[0] < 25:
            return {"YOU PROBABLY WROTE THAT"}
        else:
            preprocessed_input = text_preprocessing.transform(input_df)
            prediction = model_texts.predict_proba(preprocessed_input)
            prediction = np.array(prediction)

            if prediction.ndim == 0:
                pred_val = float(prediction)
            elif prediction.ndim == 1:
                pred_val = float(prediction[1])
            elif prediction.ndim == 2:
                pred_val = float(prediction[0][1])
            else:
                raise ValueError(f"Unexpected prediction shape: {prediction.shape}")

            label = "YOU WROTE IT!" if pred_val < 0.5 else "GENAI WROTE THAT TEXT!"  # Text DATA => 0 : HUMAN   1 : AI Generated

            if pred_val < 0.5:
                return {"prediction" : label, "confidence": f"{round(float(1-pred_val)*100, 2)}%"}

            if pred_val>= 0.5:
                return {"prediction" : label, "confidence" : f"{round(float(pred_val)*100, 2)}%"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB').resize((32, 32))
        image_array = np.array(image)  # No normalization (handled by Rescaling layer)
        image_batch = np.expand_dims(image_array, axis=0)

        prediction = model_images.predict(image_batch)
        confidence = float(prediction[0][0])

        # Corrected labels
        label = "REAL" if confidence >= 0.5 else "AI GENERATED"                 # Image DATA => 0 : FAKE    1
        confidence_score = confidence if label == "REAL" else 1 - confidence

        return {
            "prediction": label,
            "confidence": round(confidence_score, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
