import streamlit as st  # ספריית Streamlit לבניית אפליקציות ווב
import pandas as pd  # עבודה עם טבלאות ו־CSV
from sklearn.linear_model import LogisticRegression  # מודל רגרסיה לוגיסטית לסיווג
from sklearn.model_selection import StratifiedKFold  # חלוקה לקיפולים עם שמירת יחס קטגוריות
from sklearn.metrics import accuracy_score  # חישוב דיוק (Accuracy)
import numpy as np  # חישובים נומריים
import base64  # קידוד תמונה ל־Base64 עבור רקע

# ---------- רקע אפליקציה ----------
def set_background(image_file):
    with open(image_file, "rb") as f:  # פתיחת קובץ תמונה במצב בינארי
        data = base64.b64encode(f.read()).decode()  # קידוד התמונה ל־Base64 למחרוזת
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{data}");  /* הגדרת התמונה כרקע */
        background-size: cover;  /* התאמה לגודל החלון */
        background-position: top left;  /* מיקום התמונה */
        background-repeat: no-repeat;  /* ללא חזרה */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)  # הזרקת CSS לעמוד

set_background("images/ING2.png")  # הפעלת הרקע

# ---------- כיוון RTL גלובלי ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl !important;       /* יישור מימין לשמאל */
    text-align: right !important;    /* טקסט מיושר לימין */
}
</style>
""", unsafe_allow_html=True)

# ---------- טווחים לנרמול (min-max) ----------
min_max_values = {
    "tumor-size": (2, 52),   # טווח משוער של גודל הגידול
    "inv-nodes": (1, 25),    # טווח משוער של קשריות נגועות
    "deg-malig": (1, 3)      # דרגת ממאירות (1–3)
}

def min_max_normalize(value, min_val, max_val):
    """נרמול ערך לטווח [0,1] עם חסימה לערכים שמחוץ לטווח."""
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def to_binary(val):
    """המרת 'כן'/'לא' ל־1/0."""
    return 1 if val == "כן" else 0

# ---------- כותרות והנחיות ----------
st.title("🔬 תחזית חזרת סרטן - הזנת נתונים לרופא")  # כותרת ראשית של העמוד
st.markdown("""
🧑‍⚕ **הנחיות להזנת ערכים:**
- `tumor-size` ו־`inv-nodes`: הזן את **אמצע הטווח** שנמדד.
- משתנים בינאריים: **"כן" = 1**, **"לא" = 0**.
- התוצאה המוצגת אינה תחליף לשיקול דעת רפואי.  
""")  # טקסט הסבר

# ---------- טעינת קובץ לאימון ----------
uploaded_file = st.file_uploader("📁 העלה קובץ CSV עם עמודת Class", type=["csv"])  # העלאת CSV

if uploaded_file:
    df = pd.read_csv(uploaded_file)  # קריאת הקובץ ל־DataFrame
    if "Class" not in df.columns:  # וידוא שעמודת המטרה קיימת
        st.error("❌ הקובץ חייב לכלול עמודה בשם 'Class'")
    else:
        X = df.drop("Class", axis=1)  # תכונות (פיצ'רים)
        y = df["Class"]  # עמודת יעד

        # חלוקה ל־5 קיפולים לשם הערכת ביצועים יציבה
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []  # לאגירת תוצאות הדיוק בכל קיפול

        # לולאת אימון/בדיקה לכל קיפול
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = LogisticRegression(max_iter=200)  # יצירת מודל רגרסיה לוגיסטית
            model.fit(X_train, y_train)               # אימון המודל
            y_pred = model.predict(X_test)            # חיזוי על סט הבדיקה
            acc_scores.append(accuracy_score(y_test, y_pred))  # חישוב דיוק לכל קיפול

        mean_acc = np.mean(acc_scores)  # ממוצע דיוק על כל הקיפולים
        st.success(f"✅ הנתונים התקבלו בהצלחה")  # דיווח הצלחה
        st.session_state.model = model  # שמירת המודל האחרון ב־session_state (לשימוש בחיזוי)

# ---------- טופס קלט לחיזוי ----------
if "model" in st.session_state:
    model = st.session_state.model  # שליפת המודל המאומן

    # קלט מספרי – מומלץ להזין אמצע טווח נמדד
    tumor_size = st.number_input("tumor-size (אמצע טווח בגודל הגידול)", step=0.1)  # גודל גידול
    inv_nodes = st.number_input("inv-nodes (אמצע טווח בקשריות נגועות)", step=0.1)  # קשריות נגועות
    deg_malig = st.number_input("deg-malig (דרגת ממאירות)", step=1)  # דרגת ממאירות

    # קלטים בינאריים 'כן/לא'
    node_caps = st.selectbox("node-caps (קופסית קשרית נגועה)", options=["לא", "כן"])
    irradiat = st.selectbox("irradiat (טופל בהקרנות)", options=["לא", "כן"])

    # one-hot לסטטוס גיל מעבר (3 אפשרויות)
    menopause_choice = st.radio("מצב גיל המעבר:", ["ge40 (מעל גיל 40)", "lt40 (מתחת לגיל 40)", "premeno (לפני גיל מעבר)"])
    menopause_ge40 = 1 if menopause_choice.startswith("ge40") else 0
    menopause_lt40 = 1 if menopause_choice.startswith("lt40") else 0
    menopause_premeno = 1 if menopause_choice.startswith("premeno") else 0

    # מיקום הגידול בשד (בחירות בינאריות לכל רבע/מרכז)
    breast_quad_central = st.selectbox("גידול במרכז השד (breast-quad_central)", options=["לא", "כן"])
    breast_quad_left_low = st.selectbox("גידול בשד שמאל תחתון (breast-quad_left_low)", options=["לא", "כן"])
    breast_quad_left_up = st.selectbox("גידול בשד שמאל עליון (breast-quad_left_up)", options=["לא", "כן"])
    breast_quad_right_low = st.selectbox("גידול בשד ימין תחתון (breast-quad_right_low)", options=["לא", "כן"])
    breast_quad_right_up = st.selectbox("גידול בשד ימין עליון (breast-quad_right_up)", options=["לא", "כן"])

    # בניית וקטור הקלט: נרמול רציפים + המרת בינאריים + one-hot
    input_data = [
        min_max_normalize(tumor_size, *min_max_values["tumor-size"]),  # נרמול גודל גידול
        min_max_normalize(inv_nodes, *min_max_values["inv-nodes"]),    # נרמול קשריות נגועות
        to_binary(node_caps),                                          # node-caps (0/1)
        min_max_normalize(deg_malig, *min_max_values["deg-malig"]),    # נרמול דרגת ממאירות
        to_binary(irradiat),                                           # irradiat (0/1)
        menopause_ge40, menopause_lt40, menopause_premeno,             # one-hot של מצב גיל מעבר
        to_binary(breast_quad_central),
        to_binary(breast_quad_left_low),
        to_binary(breast_quad_left_up),
        to_binary(breast_quad_right_low),
        to_binary(breast_quad_right_up)
    ]

    # ---------- כפתור חישוב תחזית ----------
    if st.button("🔍 חשב תחזית"):
        # חיזוי קטגוריאלי: 0 = ללא חזרה, 1 = חזרה
        prediction = model.predict([input_data])[0]  # חיזוי מחלקה
        # הסתברות למחלקה 1 (חזרת מחלה) אם זמינה ב־scikit-learn
        if hasattr(model, "predict_proba"):
            prob_recurr = model.predict_proba([input_data])[0][1]  # הסתברות לחזרה
        else:
            prob_recurr = None  # אם אין predict_proba

        # הצגת תוצאה והמלצה קלינית טקסטואלית
        if prediction == 1:
            # הודעת סיכון + הסתברות + המלצה להיבדק בתדירות גבוהה יותר
            if prob_recurr is not None:
                st.error(f"🔴 סיכון לחזרת סרטן  | הסתברות משוערת: {prob_recurr:.1%}")
            else:
                st.error("🔴 סיכון לחזרת סרטן ")
            st.markdown(
                "**המלצה:** המערכת זיהתה צפי לחזרת המחלה. "
                "מומלץ להיוועץ ברופא/ה המטפל/ת, לשקול בדיקות משלימות, "
                "ולהיבדק בתדירות **גבוהה יותר** לפי הנחיות רפואיות."
            )
        else:
            # הודעת ללא סיכון + הסתברות + המלצה לתדירות רגילה
            if prob_recurr is not None:
                st.success(f"🟢 ללא חזרת סרטן  | הסתברות לחזרה: {prob_recurr:.1%}")
            else:
                st.success("🟢 ללא חזרת סרטן")
            st.markdown(
                "**המלצה:** המערכת **לא זיהתה** צפי לחזרת המחלה. "
                "יש להיבדק בתדירות **הנדרשת** על פי הנחיות הרופא/ה המטפל/ת."
            )

        # הבהרה רפואית (Disclaimer)
        st.warning(
            "⚠️ הכלי הוא עזר תומך החלטה ואינו תחליף לשיקול דעת רפואי. "
            "יש לפרש את התוצאות בהקשר קליני מלא."
        )
else:
    # הנחיה להעלות קובץ לפני ביצוע חיזוי
    st.info("⬆ יש להעלות קובץ כדי לאמן את המודל לפני הזנת תחזית.")
