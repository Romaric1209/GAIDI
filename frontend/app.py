import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
from io import BytesIO
import base64

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/Romaric1209/GAIDI/refs/heads/main/frontend/static/media/human_and_robotic_interaction_16031691.jpg");
    background-size: cover;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}/*Header Band*/

[data-testid="stBaseButton-headerNoPadding"] {
    right: 2rem;
    background-image: url("https://www.shutterstock.com/image-vector/burger-line-vector-illustration-isolated-600nw-1946040832.jpg");
    background-size: cover;
    color: rgba(255,255,255,1);
}


[data-testid="stMarkdownContainer"].st-emotion-cache-16tyu1 {
    color: rgba(255, 255, 255, 1);
    font-weight: bold;
    font-size: 1.5rem
} /*Powered by Streamlit...*/

[data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFileName"].st-emotion-cache-1rpn56r,
[data-testid="stBaseButton-minimal"].st-emotion-cache-8ccstr,
[data-testid="stFileUploaderFile"].st-emotion-cache-4mjat2 {
    color: rgba(255, 255, 255, 1);
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(255, 255, 255, 0.2);
    z-index: 0;
} /*background opacity*/

.readable-box {
    display: inline-block;
    background-color: rgba(255, 255, 255, 0.7);
    padding: 0.75rem 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    margin-bottom: 1rem;
    z-index: 1;
    position: relative;
    max-width: 90%;
    color: rgb(40,126,90);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "media")

try:
    image_human = Image.open(os.path.join(STATIC_DIR, "human.png"))
    image_ai = Image.open(os.path.join(STATIC_DIR, "robot.png"))
    image_human = image_human
    image_ai = image_ai

except Exception as e:
    st.error(f"Error loading result images: {e}")


def display_image_inline(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(f'<img src="data:image/png;base64,{data}" width="400"/>', unsafe_allow_html=True)

st.markdown("""
<div class="readable-box">
    <h1>GAIDI - GenAI Data Identificator</h1>
</div>
""", unsafe_allow_html=True)

# Load models
text_preprocessing = joblib.load("models/pipeline.joblib")
model_texts = joblib.load("models/svm_model.joblib")
model_images = tf.keras.models.load_model("models/image_model.keras")
st.markdown("""
<div class="readable-box">
    <h2>📝 Text Analysis</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="readable-box">
    <p>Enter English text for prediction</p>
</div>
""", unsafe_allow_html=True)
text_input = st.text_area(" ", placeholder="Type or paste English text here...")

if st.button("Analyze Text", key="text_analyze"):
    if not text_input.strip():
        st.error("No text provided.")
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

        label = "YOU WROTE IT!" if pred_val <= 0.5 else "GENAI WROTE THAT TEXT!"
        confidence = round(float(pred_val if pred_val >= 0.5 else 1 - pred_val) * 100)

        st.markdown(
                f"""
                <div class="readable-box">
                        <div style="background-color:
                        padding: 0.75rem 1rem;
                        border-radius: 10px;
                        font-weight: 400;
                        font-family: sans-serif;
                        font-size: 1rem;">
                    RESULT: {label} \n With confidence {confidence}%
                </div>
                """,
                unsafe_allow_html=True
            )
        display_image_inline(image_human if pred_val <= 0.5 else image_ai)

st.markdown("""
<div class="readable-box">
    <h2>🖼️ Image Analysis</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="readable-box">
    <p>Choose an image...</p>
</div>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    try:
        uploaded_image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Error displaying uploaded image: {e}")

    try:
        # Preprocess image
        image = uploaded_image.resize((32, 32))
        image = image.convert("RGB")
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")

    if st.button("Analyze Image", key="image_analyze"):
        try:
            prediction = model_images.predict(image_batch)
            confidence = float(prediction[0][0])
            label = "REAL! It is NOT AI GENERATED" if confidence >= 0.5 else "FAKE, It is AI GENERATED"
            confidence_score = confidence if label == "REAL! It is NOT AI GENERATED" else 1 - confidence
            st.markdown(
                f"""
                <div class="readable-box">
                        <div style="background-color:
                        padding: 0.75rem 1rem;
                        border-radius: 10px;
                        font-weight: 400;
                        font-family: sans-serif;
                        font-size: 1rem;">
                    RESULT: {label} with confidence {round(confidence_score * 100)}%
                </div>
                """,
                unsafe_allow_html=True
            )
            display_image_inline(image_human if confidence >= 0.5 else image_ai)

        except Exception as e:
            st.error(f"Error during image analysis: {e}")

st.markdown("---")
st.markdown("🛠️ Powered by Streamlit | GAIDI | Made with ❤️ by F. Romaric Berger")
