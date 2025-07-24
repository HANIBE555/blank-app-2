import streamlit as st
import base64

# תצורת הדף
st.set_page_config(page_title="ברוכים הבאים", layout="wide")

# עיצוב RTL וטעינת רקע תמונה
def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    html, body, [class*="css"] {{
        direction: rtl;
        text-align: right;
        font-family: Arial;
    }}
    .button-style {{
        display: inline-block;
        padding: 0.6em 1.2em;
        margin: 10px;
        font-size: 1.1em;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 10px;
        text-decoration: none;
        cursor: pointer;
    }}
    .button-style:hover {{
        background-color: #45a049;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# טוען רקע
set_bg_from_local("images/IMG.png")

# כותרת ראשית
st.markdown("<h1 style='text-align: center;'>🧬 ברוכים הבאים לאפליקציית חיזוי סרטן השד</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>נא לבחור את הפעולה שתרצו לבצע:</h3>", unsafe_allow_html=True)

# כפתורים למעבר לדפים
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 חיזוי חזרת מחלה", key="page1"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")

    if st.button("🖼️ בדיקת תמונות CT", key="page2"):
        st.switch_page("pages/2_בדיקת_CT.py")
