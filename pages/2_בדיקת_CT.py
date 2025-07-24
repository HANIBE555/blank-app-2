import streamlit as st
import os
import base64
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report

# --- רקע מותאם ---
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

set_background("images/ING2.png")

# --- כותרת ---
st.title("🧠 זיהוי גידול בתמונות CT (הרצה מקומית)")

st.markdown("📁 הזן את הנתיב המקומי לתיקייה המכילה שתי תיקיות בשם **'Cancer'** ו־**'Non-Cancer'**:")
local_path = st.text_input("📂 נתיב לתיקייה:")

if local_path:
    if not os.path.exists(local_path):
        st.error("❌ הנתיב אינו קיים. ודא שהזנת נתיב תקין.")
    elif not all(os.path.exists(os.path.join(local_path, folder)) for folder in ["Cancer", "Non-Cancer"]):
        st.error("❌ חסרה אחת מהתיקיות: 'Cancer' או 'Non-Cancer'. ודא שהמבנה נכון.")
    else:
        with st.spinner("🔁 מאמן את המודל..."):
            try:
                datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

                train_gen = datagen.flow_from_directory(
                    local_path,
                    target_size=(224, 224),
                    batch_size=32,
                    class_mode='binary',
                    subset='training',
                    seed=42,
                    classes=['Non-Cancer', 'Cancer']
                )

                val_gen = datagen.flow_from_directory(
                    local_path,
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

                history = model.fit(train_gen, validation_data=val_gen, epochs=10)

                # חיזוי וביצועים
                val_gen.reset()
                y_pred = model.predict(val_gen, verbose=0)
                y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
                y_true = val_gen.classes

                report = classification_report(y_true, y_pred_classes, target_names=["Non-Cancer", "Cancer"], output_dict=True)
                recall_cancer = report["Cancer"]["recall"]

                st.success(f"✅ אימון הושלם על {train_gen.samples} תמונות אימון ו־{val_gen.samples} תמונות ולידציה.")
                st.markdown(f"📊 <b>Recall עבור קבוצת Cancer:</b> {recall_cancer:.2f}", unsafe_allow_html=True)

                st.session_state.trained_model = model

                # גרפים
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].plot(history.history['accuracy'], label='Train Accuracy')
                ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
                ax[0].set_title("דיוק")
                ax[0].legend()

                ax[1].plot(history.history['loss'], label='Train Loss')
                ax[1].plot(history.history['val_loss'], label='Val Loss')
                ax[1].set_title("איבוד")
                ax[1].legend()

                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ שגיאה במהלך האימון: {e}")

# --- שלב 2: בדיקת תמונה חדשה ---
if "trained_model" in st.session_state:
    image_file = st.file_uploader("🖼️ העלה תמונת CT לבדיקה", type=["jpg", "jpeg", "png"])

    if image_file:
        img = load_img(image_file, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = st.session_state.trained_model.predict(img_array)[0][0]

        st.image(img, caption="תמונה שנבחרה", use_column_width=True)
        if prediction > 0.5:
            st.error("🔴 גידול זוהה (Cancer)")
        else:
            st.success("🟢 לא זוהה גידול (Non-Cancer)")
