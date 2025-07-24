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

# הגדרת רקע
set_background("images/PNG2.png")

# --- טעינת מודל ---
@st.cache_resource
def load_trained_model():
    return load_model("cancer_detector_model.h5")

model = load_trained_model()

# --- ממשק משתמש ---
st.title("🧠 בדיקת תמונת CT לזיהוי גידול")
st.write("העלה תמונה של סריקת CT כדי לבדוק האם מזוהה גידול.")

uploaded_file = st.file_uploader("📤 העלה תמונה", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # הצגת תמונה
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="תמונה שהועלתה", use_column_width=True)

    # עיבוד ותחזית
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # הצגת תוצאה
    st.subheader("🔍 תוצאה:")
    if prediction > 0.5:
        st.error("⚠️ זוהה גידול בתמונה")
    else:
        st.success("✅ לא זוהה גידול בתמונה")

