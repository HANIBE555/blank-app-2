import streamlit as st  # מייבא את ספריית Streamlit ליצירת אפליקציות ווב
from streamlit_extras.switch_page_button import switch_page  # מייבא פונקציה למעבר בין דפי האפליקציה
import base64  # מייבא ספרייה לקידוד/פענוח נתונים בפורמט Base64

# פונקציה להגדרת רקע האפליקציה
def set_background(image_file):
    with open(image_file, "rb") as f:  # פותח את קובץ התמונה לקריאה בינארית
        data = base64.b64encode(f.read()).decode()  # מקודד את התמונה ל־Base64 וממיר למחרוזת

    # מגדיר CSS לעיצוב הרקע, כותרות וכפתורים
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{data}");  /* הגדרת התמונה כרקע */
        background-size: cover;  /* התאמה לגודל המסך */
        background-position: center;  /* מיקום במרכז */
        background-repeat: no-repeat;  /* ללא חזרה של התמונה */
        direction: rtl;  /* יישור טקסט מימין לשמאל */
    }}
    .title {{
        text-align: center;  /* טקסט ממורכז */
        font-size: 60px;  /* גודל פונט */
        font-weight: bold;  /* טקסט מודגש */
        margin-bottom: 10px;  /* רווח תחתון */
    }}
    .subtitle {{
        text-align: center;  /* טקסט ממורכז */
        font-size: 24px;  /* גודל פונט */
        margin-bottom: 30px;  /* רווח תחתון */
    }}
    .stButton>button {{
        background-color: #f94ca4;  /* צבע רקע הכפתור */
        color: white;  /* צבע טקסט */
        font-size: 18px;  /* גודל טקסט */
        padding: 10px 30px;  /* ריווח פנימי */
        border-radius: 10px;  /* פינות מעוגלות */
        border: none;  /* ללא מסגרת */
        transition: 0.3s;  /* אנימציה בריחוף */
    }}
    .stButton>button:hover {{
        background-color: #ff6fbd;  /* צבע רקע בריחוף */
        transform: scale(1.03);  /* הגדלה קלה בריחוף */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)  # מזריק את ה־CSS לעמוד

set_background("images/IMG.png")  # הפעלת הפונקציה עם תמונת הרקע

# הצגת כותרות
st.markdown('<div class="title">שלום רופא יקר 🩺</div>', unsafe_allow_html=True)  # כותרת ראשית
st.markdown('<div class="subtitle">יש לבחור אחת מהאופציות הבאות:</div>', unsafe_allow_html=True)  # כותרת משנה

# יצירת פריסה של 4 עמודות ביחס 2:1:1:2
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

# כפתור ראשון בתוך העמודה השנייה
with col2:
    if st.button("🔍 חיזוי חזרת מחלה"):  # כפתור לחיזוי חזרת מחלה
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")  # מעבר לדף חיזוי מחלה

# כפתור שני בתוך העמודה השלישית
with col3:
    if st.button("🖼 בדיקת תמונות CT"):  # כפתור לבדיקה של תמונות CT
        st.switch_page("pages/2_בדיקת_CT.py")  # מעבר לדף בדיקת CT
