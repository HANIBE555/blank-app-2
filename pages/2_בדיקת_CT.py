import streamlit as st
import base64
import tempfile
import requests
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# --- רקע ---
def set_background(image_path):
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
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/ING2.png")

# --- כותרת ---
st.title("🧠 בדיקת CT - זיהוי גידול באמצעות מודל מאומן")

# --- טעינת המודל מ-Google Drive ---
@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?id=1wnArqGSS3kJrtehWe6oqyE9uP_8mgLyf"
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(response.content)
        tmp.flush()
        return load_model(tmp.name)

st.info("⏳ טוען את המודל המאומן...")
model = load_model_from_drive()
st.success("✅ המודל נטען בהצלחה!")

# --- העלאת תמונה ---
st.header("🔍 העלה תמונת CT לבדיקה")

image_file = st.file_uploader("בחר קובץ תמונה (JPG, PNG)", type=["jpg", "jpeg", "png"])

if image_file:
    # עיבוד התמונה
    img = load_img(image_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # חיזוי
    prediction = model.predict(img_array)[0][0]

    # תוצאה
    st.image(img, caption="📷 תמונה שהועלתה", use_column_width=True)
    if prediction > 0.5:
        st.error("🔴 גידול זוהה (Cancer)")
    else:
        st.success("🟢 לא זוהה גידול (Non-Cancer)")
