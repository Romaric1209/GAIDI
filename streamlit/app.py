import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
  background-size: cover;
}
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
  right: 2rem;
  background-image: url("https://img.freepik.com/premium-vector/burger-icon-isolated-illustration_92753-2926.jpg?w=2000");
  background-size: cover;
}

</style>
"""

st.markdown(page_bg,
    # """
    # <style>
    # .main {
    #     background-image: path("C:\Users\ASUS\Desktop\MyResume-1.0.0\MyResume-1.0.0\assets\img\historia-ia-746x419.jpg");
    #     background-size: cover;
    #     background-position: center;
    #     background-repeat: no-repeat;
    #     height: 100vh;
    #     width: 100vw;
    #     margin: 0;
    # }
    # .block-container {
    #     padding-top: 1rem;
    #     padding-bottom: 1rem;
    #     padding-left: 2rem;
    #     padding-right: 2rem;
    #     background-color: rgba(255, 255, 255, 0.8);
    # }
    # </style>
    # <div class="main">
    #     <div class="block-container">
    # """,
    unsafe_allow_html=True
)

# Load models
text_preprocessing = joblib.load("roma_models/pipeline.joblib")
model_texts = joblib.load("roma_models/XGBoost_model.joblib")
model_images = tf.keras.models.load_model("roma_models/image_model.keras")


st.title("GAIDI - GenAI Data Identificator")


st.header("Text Prediction")
text_input = st.text_area("Enter text for prediction")

if st.button("Predict Text"):
    if not text_input.strip():
        st.error("No text provided.")
    elif len(text_input.split()) < 20:
        st.warning("Warning: Text too short!")
    else:
        input_df = pd.DataFrame({"text": [text_input]})
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
            st.error(f"Unexpected prediction shape: {prediction.shape}")
            pred_val = 0

        label = "YOU WROTE IT!" if pred_val < 0.5 else "GENAI WROTE THAT TEXT!"
        confidence = round(float(pred_val if pred_val >= 0.5 else 1 - pred_val) * 100, 2)

        st.success(f"Prediction: {label} with confidence {confidence}%")



st.header("Image Prediction")
uploaded_file = st.file_uploader("Choose an image...", type="image")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB').resize((32, 32))
    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)

    if st.button("Predict Image"):
        prediction = model_images.predict(image_batch)
        confidence = float(prediction[0][0])

        label = "REAL" if confidence >= 0.5 else "AI GENERATED"
        confidence_score = confidence if label == "REAL" else 1 - confidence

        st.success(f"Prediction: {label} with confidence {round(confidence_score, 2)}")
