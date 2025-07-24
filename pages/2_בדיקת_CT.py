import streamlit as st
import os
import numpy as np
import base64
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report

# --- הגדרת רקע מותאם ---
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
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# 🖼️ החל רקע
set_background("images/bg.png")  # ודא שהקובץ הזה קיים

# --- כותרת ---
st.title("🧠 זיהוי גידול בתמונות CT (הרצה מקומית)")

st.markdown("📂 הזן את הנתיב המקומי לתיקייה המכילה שתי תיקיות בשם 'Cancer' ו-'Non-Cancer':")

# --- קליטת נתיב ---
dataset_path = st.text_input("📁 נתיב לתיקייה:")

# בדיקה אם קיים נתיב
if dataset_path:
    if os.path.exists(dataset_path):
        st.success("📁 תיקייה קיימת! בודק מבנה תיקיות...")

        cancer_path = os.path.join(dataset_path, "Cancer")
        non_cancer_path = os.path.join(dataset_path, "Non-Cancer")

        if os.path.exists(cancer_path) and os.path.exists(non_cancer_path):
            st.success("✅ נמצא מבנה תיקיות תקין. מתחיל לאמן את המודל...")

            # הגדרת מחוללי נתונים
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

            train_gen = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='training',
                seed=42,
                classes=['Non-Cancer', 'Cancer']
            )

            val_gen = datagen.flow_from_directory(
                dataset_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                subset='validation',
                seed=42,
                classes=['Non-Cancer', 'Cancer']
            )

            # מודל ResNet50
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

            # אימון
            history = model.fit(train_gen, validation_data=val_gen, epochs=10)

            # חיזוי וביצועים
            y_pred = model.predict(val_gen)
            y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
            y_true = val_gen.classes

            report = classification_report(
                y_true, y_pred_classes,
                target_names=["Non-Cancer", "Cancer"],
                output_dict=True
            )
            recall_cancer = report["Cancer"]["recall"]
            st.write("📊 **Recall לקבוצת Cancer:**", f"{recall_cancer:.2f}")

            # גרפים
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['accuracy'], label='דיוק אימון')
            ax[0].plot(history.history['val_accuracy'], label='דיוק אימות')
            ax[0].set_title("דיוק")
            ax[0].legend()

            ax[1].plot(history.history['loss'], label='איבוד אימון')
            ax[1].plot(history.history['val_loss'], label='איבוד אימות')
            ax[1].set_title("איבוד")
            ax[1].legend()

            st.pyplot(fig)
        else:
            st.error("❌ לא נמצאו תיקיות בשם 'Cancer' ו-'Non-Cancer' בתוך הנתיב שהוזן.")
    else:
        st.error("❌ הנתיב שהוזן אינו קיים במחשב.")
