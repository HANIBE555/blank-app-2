import streamlit as st
import zipfile
import os
import shutil
import tempfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image

st.set_page_config(page_title="בדיקת CT", layout="wide")

# --- הגדרת רקע ---
def set_background(image_path):
    import base64
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
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/ING2.png")

st.title("🖼 בדיקת תמונות CT – זיהוי גידולים")

# --- שלב 1: העלאת תיקיית אימון כ- ZIP ---
uploaded_zip = st.file_uploader("📁 העלה תיקיית אימון (ZIP) עם תיקיות Cancer ו־Non-Cancer", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # חיפוש התיקייה שבתוך ה-ZIP
        for root, dirs, files in os.walk(tmpdir):
            if "Cancer" in dirs and "Non-Cancer" in dirs:
                dataset_path = root
                break
        else:
            st.error("❌ לא נמצאו תיקיות 'Cancer' ו־'Non-Cancer' בתוך הקובץ.")
            st.stop()

        # מחולל נתונים
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training',
            seed=42
        )

        val_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            seed=42
        )

        # בניית המודל
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        from tensorflow.keras.layers import Input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        with st.spinner("🔄 מאמן את המודל... המתן מספר שניות"):
            model.fit(train_generator, validation_data=val_generator, epochs=10)
            st.success("✅ המודל אומן בהצלחה!")

        st.session_state['model'] = model

# --- שלב 2: העלאת תמונה לחיזוי ---
if 'model' in st.session_state:
    uploaded_img = st.file_uploader("🖼 העלה תמונת CT בודדת לבדיקת גידול", type=["png", "jpg", "jpeg"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert('RGB')
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = st.session_state['model'].predict(img_array)[0][0]
        if prediction > 0.5:
            st.error("🔴 נמצא חשד לגידול (Cancer)")
        else:
            st.success("🟢 לא זוהה גידול (Non-Cancer)")

        st.image(image, caption="תמונה שנבדקה", width=300)
else:
    st.info("⬆ יש להעלות קובץ אימון תחילה כדי לאמן את המודל")

