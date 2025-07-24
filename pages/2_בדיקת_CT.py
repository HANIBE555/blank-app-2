import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile
import os
import base64

# --- הגדרת רקע ו-RTL ---
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
            font-family: Arial, sans-serif;
        }}
        h1, h2, h3, h4, h5, h6, p, label, div {{
            direction: rtl !important;
            text-align: right !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ✅ עדכון נתיב לתמונה מתוך ../images כי אנחנו בתוך /pages/
set_background("../images/ING2.png")

# --- הגדרות כלליות לדף ---
st.set_page_config(page_title="בדיקת CT לגידול", layout="centered")
st.title("🧠 מערכת לזיהוי גידול בתמונת CT")

# --- שלב 1: העלאת קובץ מודל ---
st.subheader("1. העלה קובץ מודל מסוג H5")
model_file = st.file_uploader("בחר קובץ מודל (קובץ .h5)", type=["h5"])

if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name

    try:
        model = load_model(tmp_path)
        st.success("✅ המודל נטען בהצלחה!")

        # --- שלב 2: העלאת תמונת CT ---
        st.subheader("2. העלה תמונת CT לבדיקה")
        image_file = st.file_uploader("בחר תמונה", type=["jpg", "jpeg", "png"])

        if image_file is not None:
            st.image(image_file, caption="תמונה שהועלתה", use_container_width=True)

            # עיבוד תמונה
            img = load_img(image_file, target_size=(224, 224))  # שים לב להתאים לפי המודל שלך
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # חיזוי
            prediction = model.predict(img_array)
            prob = float(prediction[0][0])
            result = "🔴 גידול זוהה" if prob >= 0.5 else "🟢 אין גידול"

            # תוצאה
            st.subheader("תוצאה:")
            st.markdown(f"### {result}")
            st.markdown(f"#### סבירות לגידול: {prob * 100:.2f}%")

    except Exception as e:
        st.error("❌ שגיאה בטעינת המודל. ודא שהקובץ הוא תקין מסוג .h5.")
        st.exception(e)
