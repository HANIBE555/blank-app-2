import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64
import os

# --- רקע ---
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            direction: rtl;
            text-align: right;
        }}
        .stButton>button {{
            background-color: #f94ca4;
            color: white;
            font-size: 18px;
            padding: 10px 30px;
            border-radius: 10px;
            border: none;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# קובץ הרקע שלך
set_background("images/PNG2.png")

# --- טעינת המודל המאומן ---
@st.cache_resource
def load_trained_model():
    return load_model("cancer_detector_model.h5")

model = load_trained_model()

# --- ממשק ---
st.title("🧠 בדיקת CT לזיהוי גידול")
st.write("העלה תמונה כדי לבדוק האם זוהה גידול סרטני.")

uploaded_file = st.file_uploader("📤 העלה תמונה (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.image(image, caption="📷 התמונה שהועלתה", use_column_width=True)
    st.subheader("🔍 תוצאה:")
    if prediction > 0.5:
        st.error("🔴 זוהה גידול בתמונה (Cancer)")
    else:
        st.success("🟢 לא זוהה גידול בתמונה (Non-Cancer)")
