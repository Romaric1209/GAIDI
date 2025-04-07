import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import os
from io import BytesIO
import base64

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/Romaric1209/GAIDI/main/streamlit/static/media/human_and_robotic_interaction_16031691.jpg");
    background-size: cover;
} /*background*/

[data-testid="stHeader"] {
     background-color: rgba(0,0,0,0);
} /*No white Band at the top*/

[data-testid="stBaseButton-headerNoPadding"] {
    right: 2rem;
    background-image: url("https://www.shutterstock.com/image-vector/burger-line-vector-illustration-isolated-600nw-1946040832.jpg");
    background-size: cover;
    color: rgba(255,255,255,1);
} /* Hamburger Menu*/

[data-testid="stBaseButton-header"]{
        color: rgba(255,255,255,1);
}

[data-testid="stHeadingWithActionElements"] {
    color: rgba(255, 255, 255, 1);
    -webkit-tap-highlight-color: rgba(0, 0, 0, 0.5);
} /* Text Analysis / image...*/

[data-testid="stMarkdownContainer"] {
    color: rgba(0, 0, 0, 1);
    font-weight: bold;
} /* Analyze Text/ Image button*/

[data-testid="stMarkdownContainer"].st-emotion-cache-br351g {
    color: rgba(255, 255, 255, 1);
    font-weight: bold;
    font-size: 25px
}

[data-testid="stMarkdownContainer"].st-emotion-cache-16tyu1 {
    color: rgba(255, 255, 255, 1);
    font-weight: bold;
    font-size: 1.5rem
} /*result */

[data-testid="stFileUploaderFileName"] {
    color: rgba(255, 255, 255, 1);
} /* .png */

[data-testid="stFileUploaderFileName"].st-emotion-cache-1rpn56r {
    color: rgba(255, 255, 255, 1);
} /* Kb */

[data-testid="stBaseButton-minimal"].st-emotion-cache-8ccstr {
    color: rgba(255, 255, 255, 1);
}

[data-testid="stFileUploaderFile"].st-emotion-cache-4mjat2 {
    color: rgba(255, 255, 255, 1);
} /*Image File Icon /don't work */


[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(255, 255, 255, 0.2);
    z-index: 0;
} /* Fog*/
</style>
"""

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static", "media")

try:
    image_human = Image.open(os.path.join(STATIC_DIR, "human.png"))
    image_ai = Image.open(os.path.join(STATIC_DIR, "robot.png"))
    # st.write("Loaded image_human:", image_human)
    # st.write("Loaded image_ai:", image_ai)
    image_human = image_human
    image_ai = image_ai

except Exception as e:
    st.error(f"Error loading result images: {e}")


def display_image_inline(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(f'<img src="data:image/png;base64,{data}" width="400"/>', unsafe_allow_html=True)

st.title("GAIDI - GenAI Data Identificator")
st.markdown(page_bg, unsafe_allow_html=True)

# Load models
text_preprocessing = joblib.load("roma_models/pipeline.joblib")
model_texts = joblib.load("roma_models/stacking_text_model.joblib")
model_images = tf.keras.models.load_model("roma_models/image_model.keras")

st.header("📝 Text Analysis")
text_input = st.text_area("Enter English text for prediction")

if st.button("Analyze Text", key="text_analyze"):
    if not text_input.strip():
        st.error("No text provided.")
    elif len(text_input.split()) < 6:
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

        label = "YOU WROTE IT!" if pred_val <= 0.5 else "GENAI WROTE THAT TEXT!"
        confidence = round(float(pred_val if pred_val >= 0.5 else 1 - pred_val) * 100)

        st.success(f"RESULT: {label} \n With confidence {confidence}%")
        display_image_inline(image_human if pred_val <= 0.5 else image_ai)


st.header("🖼️ Image Analysis")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.write("Uploaded file:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size (bytes):", uploaded_file.size)
    
    # try:
    #     # Attempt to open and display the uploaded image
    #     uploaded_image = Image.open(uploaded_file)
    #     st.image(uploaded_image, caption="Uploaded Image Preview", use_container_width=True)
    # except Exception as e:
    #     st.error(f"Error displaying uploaded image: {e}")
    
    try:
        # Preprocess image
        image = uploaded_image.convert("RGB").resize((32, 32))
        image_array = np.array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        st.write("Image processed; shape:", image_array.shape)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
    
    if st.button("Analyze Image", key="image_analyze"):
        try:
            prediction = model_images.predict(image_batch)
            confidence = float(prediction[0][0])
            label = "It is a REAL picture" if confidence >= 0.5 else "It is AI GENERATED"
            confidence_score = confidence if label == "It is a REAL picture" else 1 - confidence
            st.success(f"RESULT: {label} with confidence {round(confidence_score * 100)}%")
            display_image_inline(image_human if confidence >= 0.5 else image_ai)

        except Exception as e:
            st.error(f"Error during image analysis: {e}")

st.markdown("---")
st.markdown("🛠️ Powered by Streamlit | GAIDI | Made with ❤️ by F. Romaric Berger")