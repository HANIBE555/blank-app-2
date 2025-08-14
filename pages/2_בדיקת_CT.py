import streamlit as st  # ספריית Streamlit לממשק אינטראקטיבי
import numpy as np      # פעולות נומריות
from tensorflow.keras.models import load_model  # טעינת מודל Keras מקובץ .h5
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # טעינת/המרת תמונה
import tempfile         # יצירת קבצים זמניים לשמירת המודל שהועלה
import os               # עבודה עם קבצים ונתיבים
import base64           # קידוד תמונת רקע ל-Base64

# --- פונקציית עזר: רקע ו-RTL ---
def set_background(image_path):
    """מגדיר תמונת רקע, כיוון RTL ועימוד ימין לכל האלמנטים בדף."""
    if os.path.exists(image_path):  # בדיקה שהתמונה קיימת
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()  # קידוד לתצוגה ב-CSS
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;         /* התאמה מלאה לרוחב/גובה */
            background-position: top left;  /* מיקום התמונה */
            background-repeat: no-repeat;   /* ללא חזרה */
            direction: rtl;                 /* כיוון כתיבה מימין לשמאל */
            text-align: right;              /* יישור טקסט לימין */
            font-family: Arial, sans-serif; /* פונטים קריאים */
        }}
        h1, h2, h3, h4, h5, h6, p, label, div {{
            direction: rtl !important;      /* הבטחת RTL לכל רכיב */
            text-align: right !important;   /* יישור לימין */
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ✅ תמונת רקע
set_background("images/ING2.png")


st.set_page_config(page_title="בדיקת CT לגידול", layout="centered")  # הגדרות כלליות של העמוד
st.title("🧠 מערכת לזיהוי גידול בתמונת CT")  # כותרת ראשית

# --- שלב 1: העלאת קובץ מודל ---
st.subheader("1. העלה קובץ מודל ")  # הוראות למשתמש
model_file = st.file_uploader("בחר קובץ מודל (קובץ .h5)", type=["h5"])  # העלאת מודל

# --- פונקציית עזר: הודעת המלצה לפי סבירות ---
def recommendation_by_prob(prob_0_to_1: float) -> str:
    """
    מחזירה הודעת המלצה קלינית בהתאם לסבירות (0–1).
    הנחות:
    - מתחת ל-0.5: אין זיהוי גידול → תדירות מעקב רגילה לפי הנחיות רפואיות.
    - 0.5–0.6: זיהוי בהסתברות נמוכה → מומלץ לערוך בדיקות נוספות.
    - 0.6–0.8: זיהוי כמעט ודאי → יש להמשיך לבדיקה אצל רופא/ה.
    - 0.8–1.0: זיהוי ודאי → לטפל בדחיפות לפי הנחיות רפואיות.
    """
    p = prob_0_to_1
    if p < 0.5:
        return ("**המלצה:** המערכת לא זיהתה גידול. "
                "מומלץ להמשיך במעקב בתדירות **הנדרשת** על פי הנחיות רופא/ה.")
    elif 0.5 <= p < 0.6:
        return ("**המלצה:** המערכת זיהתה גידול בהסתברות **נמוכה** (50–60%). "
                "מומלץ לערוך **בדיקות נוספות** ולהיוועץ ברופא/ה.")
    elif 0.6 <= p < 0.8:
        return ("**המלצה:** המערכת זיהתה גידול בצורה **כמעט ודאית** (60–80%). "
                "יש להמשיך ל**בדיקת רופא/ה** בהקדם.")
    else:  # 0.8–1.0
        return ("**המלצה:** המערכת זיהתה גידול באופן **ודאי** (80–100%). "
                "יש **לטפל בהתאם ובדחיפות** לפי הנחיות רפואיות.")

if model_file is not None:
    # שמירת קובץ המודל לקובץ זמני כדי לאפשר ל-load_model לטעון אותו מהדיסק
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(model_file.read())
        tmp_path = tmp.name  # הנתיב לקובץ המודל הזמני

    try:
        model = load_model(tmp_path)  # טעינת המודל
        st.success("✅ המודל נטען בהצלחה!")  # פידבק חיובי למשתמש

        # --- שלב 2: העלאת תמונת CT ---
        st.subheader("2. העלה תמונת CT לבדיקה")
        image_file = st.file_uploader("בחר תמונה", type=["jpg", "jpeg", "png"])  # העלאת תמונה

        if image_file is not None:
            st.image(image_file, caption="תמונה שהועלתה", use_container_width=True)  # תצוגת התמונה

            # --- עיבוד תמונה לפני חיזוי ---
            # הערה: התאם את target_size לערכי הקלט שהמודל שלך ציפה להם באימון
            img = load_img(image_file, target_size=(224, 224))  # ריסייז לתואם למודל
            img_array = img_to_array(img) / 255.0               # נירמול לערכים 0–1
            img_array = np.expand_dims(img_array, axis=0)       # הוספת ממד Batch

            # --- חיזוי באמצעות המודל ---
            prediction = model.predict(img_array)               # הפקת חיזוי מהמודל
            # הנחה: המודל מחזיר וקטור ציון/הסתברות בגודל 1: [[p]]
            prob = float(prediction[0][0])                      # המרה ל-float (0–1)
            result = "🔴 גידול זוהה" if prob >= 0.5 else "🟢 אין גידול"  # סף 0.5 ברירת מחדל

            # --- תוצאות והמלצות ---
            st.subheader("תוצאה:")
            st.markdown(f"### {result}")                        # הצגת תוצאה בינארית
            st.markdown(f"#### סבירות לגידול: {prob * 100:.2f}%")  # הצגת הסבירות באחוזים

            # המלצה מותאמת לפי סבירות (כולל ארבעת המצבים שביקשת)
            st.markdown(recommendation_by_prob(prob))

            # דיסקליימר רפואי – חשוב לשמור
            st.warning(
                "⚠️ הכלי הוא עזר תומך החלטה ואינו תחליף לשיקול דעת רפואי. "
                "יש לפרש את התוצאות בהקשר קליני ולפי הנחיות רופא/ה."
            )

    except Exception as e:
        # טיפול בשגיאה בטעינת המודל
        st.error("❌ שגיאה בטעינת המודל. ודא שהקובץ הוא תקין מסוג .h5 ושמות/שכבות תואמים.")
        st.exception(e)  # הצגת פרטי החריגה למעקב (בפיתוח/דיבוג)
