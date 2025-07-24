import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{data}");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/ING2.png")

# עיצוב RTL
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: rtl !important;
    text-align: right !important;
}
</style>
""", unsafe_allow_html=True)

# טווחים לנרמול
min_max_values = {
    "tumor-size": (2, 52),
    "inv-nodes": (1, 25),
    "deg-malig": (1, 3)
}

def min_max_normalize(value, min_val, max_val):
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

def to_binary(val):
    return 1 if val == "כן" else 0

# כותרת
st.title("🔬 תחזית חזרת סרטן - הזנת נתונים לרופא")
st.markdown("""
🧑‍⚕ **הנחיות להזנת ערכים:**
- tumor-size ו־inv-nodes: הזן את אמצע הטווח.
- משתנים בינאריים: "כן"=1, "לא"=0.
""")

uploaded_file = st.file_uploader("📁 העלה קובץ CSV עם עמודת Class", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Class" not in df.columns:
        st.error("❌ הקובץ חייב לכלול עמודה בשם 'Class'")
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc_scores.append(accuracy_score(y_test, y_pred))

        mean_acc = np.mean(acc_scores)
        st.success(f"✅אימון הושלם ")
        st.session_state.model = model  # שומר את המודל האחרון

if "model" in st.session_state:
    model = st.session_state.model

    tumor_size = st.number_input("tumor-size (אמצע טווח בגודל הגידול)", step=0.1)
    inv_nodes = st.number_input("inv-nodes (אמצע טווח בקשריות נגועות)", step=0.1)
    deg_malig = st.number_input("deg-malig (דרגת ממאירות־)", step=1)

    node_caps = st.selectbox("node-caps (קופסית קשרית נגועה)", options=["לא", "כן"])
    irradiat = st.selectbox("irradiat (טופל בהקרנות)", options=["לא", "כן"])

    menopause_choice = st.radio("מצב גיל המעבר:", ["ge40 (מעל גיל 40)", "lt40 (מתחת לגיל 40)", "premeno (לפני גיל מעבר)"])
    menopause_ge40 = 1 if menopause_choice.startswith("ge40") else 0
    menopause_lt40 = 1 if menopause_choice.startswith("lt40") else 0
    menopause_premeno = 1 if menopause_choice.startswith("premeno") else 0

    breast_quad_central = st.selectbox("גידול במרכז השד (breast-quad_central)", options=["לא", "כן"])
    breast_quad_left_low = st.selectbox("גידול בשד שמאל תחתון (breast-quad_left_low)", options=["לא", "כן"])
    breast_quad_left_up = st.selectbox("גידול בשד שמאל עליון (breast-quad_left_up)", options=["לא", "כן"])
    breast_quad_right_low = st.selectbox("גידול בשד ימין תחתון (breast-quad_right_low)", options=["לא", "כן"])
    breast_quad_right_up = st.selectbox("גידול בשד ימין עליון (breast-quad_right_up)", options=["לא", "כן"])

    input_data = [
        min_max_normalize(tumor_size, *min_max_values["tumor-size"]),
        min_max_normalize(inv_nodes, *min_max_values["inv-nodes"]),
        to_binary(node_caps),
        min_max_normalize(deg_malig, *min_max_values["deg-malig"]),
        to_binary(irradiat),
        menopause_ge40,
        menopause_lt40,
        menopause_premeno,
        to_binary(breast_quad_central),
        to_binary(breast_quad_left_low),
        to_binary(breast_quad_left_up),
        to_binary(breast_quad_right_low),
        to_binary(breast_quad_right_up)
    ]

    if st.button("🔍 חשב תחזית"):
        prediction = model.predict([input_data])[0]
        if prediction == 1:
            st.error("🔴 סיכון לחזרת סרטן (1)")
        else:
            st.success("🟢 ללא חזרת סרטן (0)")
else:
    st.info("⬆ יש להעלות קובץ כדי לאמן את המודל לפני הזנת תחזית.")

