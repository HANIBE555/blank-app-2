import streamlit as st
import zipfile
import os
import tempfile
import shutil
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="זיהוי גידול בתמונות CT", layout="centered")
st.markdown(
    """
    <h1 style='text-align: right; color: black;'>🧠 זיהוי גידול בתמונות CT (ZIP העלאת קובץ)</h1>
    <p style='text-align: right;'>יש להעלות קובץ ZIP הכולל שתי תיקיות בשם 'Cancer' ו־'Non-Cancer'</p>
    """,
    unsafe_allow_html=True
)

uploaded_zip = st.file_uploader("📦 העלה קובץ ZIP", type="zip")

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # חיפוש תקיית הבסיס שבתוכה נמצאות Cancer ו־Non-Cancer
        for root, dirs, files in os.walk(temp_dir):
            if 'Cancer' in dirs and 'Non-Cancer' in dirs:
                dataset_path = root
                break
        else:
            st.error("❌ קובץ ה-ZIP לא כולל תיקיות בשם 'Cancer' ו-'Non-Cancer'")
            st.stop()

        st.success("📂 קובץ ה-ZIP הועלה ופוענח בהצלחה. מתחיל אימון...")

        # מחולל תמונות
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.3
        )

        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training',
            seed=42,
            classes=['Non-Cancer', 'Cancer']
        )

        val_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            seed=42,
            classes=['Non-Cancer', 'Cancer']
        )

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10
        )

        # חיזוי וביצועים
        val_generator.reset()
        y_pred = model.predict(val_generator)
        y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
        y_true = val_generator.classes

        st.subheader("📊 מדדים:")
        st.text("Confusion Matrix:")
        st.text(confusion_matrix(y_true, y_pred_classes))
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred_classes, target_names=['Non-Cancer', 'Cancer']))

        # גרפים
        st.subheader("📈 גרפים")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='דיוק אימון')
        ax1.plot(history.history['val_accuracy'], label='דיוק אימות')
        ax1.set_title("דיוק המודל")
        ax1.legend()

        ax2.plot(history.history['loss'], label='איבוד אימון')
        ax2.plot(history.history['val_loss'], label='איבוד אימות')
        ax2.set_title("איבוד המודל")
        ax2.legend()

        st.pyplot(fig)
