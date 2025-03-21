from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
from io import BytesIO
from PIL import Image

app = FastAPI()

#Load the text model
text_preprocessing = joblib.load("notebooks/roma_pipeline.joblib")
model_texts = tf.keras.models.load_model("notebooks/roma_models/baseline_model.keras")

#Load the image model
model_images = tf.keras.models.load_model("notebooks/roma_models/baseline_image.keras")

#Define request for text input
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to GAIDI!"}


@app.get("/predict_text")
def predict_text(input_data:TextInput):
    try:
        preprocessed_input = text_preprocessing.transform([input_data.text])
        predict = model_texts.predict(preprocessed_input)

        label = "YOU WROTE IT!" if predict > 0.5 else "GENAI WROTE THAT TEXT!"
        return {"prediction": label, "confidence": round(float(predict[0][0]), 2)} if predict > 0.5 else {"prediction": label, "confidence": round(float(1-predict[0][0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict_image")
async def predict_image(file: UploadFile= File(...)):
    try:
        contents = await file.read()
        image=Image.open(BytesIO(contents)).resize((32, 32))
        image_array = np.array(image)/255.0
        image_array = np.expand_dims(image_array, axis=0) #add Batch Dimension

        predict = model_images.predict(image_array)

        label = "THAT IS A REAL IMAGE!" if predict > 0.5 else "THIS IS AN AI GENERATED IMAGE!"
        return {"prediction": label, "confidence": round(float(predict[0][0]), 2)} if predict > 0.5 else {"prediction": label, "confidence": round(float(1-predict[0][0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
