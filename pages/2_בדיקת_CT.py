import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- פרטים על המודל ---
MODEL_FILE = "cancer_detector_model.h5"
MODEL_DRIVE_ID = "1wnArqGSS3kJrtehWe6oqyE9uP_8mgLyf"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"

# --- הורדת המודל במידת הצורך ---
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("מוריד את המודל מ־Google Drive..."):
            gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    return load_model(MODEL_FILE)

# --- טען את המודל ---
model = download_model()

# --- כותרת הדף ---
st.title("🧠 זיהוי גידול בתמונת CT")
st.write("העלה תמונה והמערכת תזהה אם קיים גידול סרטני.")

# --- העלאת תמונה ---
uploaded_file = st.file_uploader("בחר תמונת CT", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="תמונה שהועלתה", use_column_width=True)

    # עיבוד התמונה
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # חיזוי
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    result = "🔴 גידול זוהה" if prob >= 0.5 else "🟢 אין גידול"

    # תצוגת תוצאה
    st.subheader("תוצאה:")
    st.markdown(f"### {result}")
    st.markdown(f"#### סבירות לגידול: {prob * 100:.2f}%")
