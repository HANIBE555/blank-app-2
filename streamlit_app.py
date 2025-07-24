import streamlit as st

# הגדרת כיוון ימין-לשמאל
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# כותרת לדף הבית
st.title("🔬 ברוכים הבאים לאפליקציית חיזוי סרטן השד")

# טקסט פתיחה
st.write("""
אפליקציה זו נועדה להציג תחזיות מבוססות מודלים של למידת מכונה:
- חיזוי חזרת מחלה על סמך נתונים טבלאיים
- ניתוח תמונות CT לגילוי גידולים

ניתן לעבור בין הדפים בתפריט הצד בצד שמאל.
""")

# תמונת רקע או הקדמה אם יש לך
st.image("IMG.png", use_column_width=True)

# טקסט הסבר נוסף
st.info("לחצי בתפריט משמאל כדי להתחיל 🔍")
