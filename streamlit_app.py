import streamlit as st
import base64

st.set_page_config(page_title="אפליקציית חיזוי סרטן", layout="wide")

# טוען רקע מהתמונה המקומית
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    html, body, [class*="css"] {{
        direction: rtl;
        text-align: center;
    }}
    .main-title {{
        font-size: 48px;
        font-weight: 700;
        margin-top: 50px;
        color: #2E2E2E;
    }}
    .subtitle {{
        font-size: 24px;
        margin-bottom: 40px;
        color: #333;
    }}
    .button-container {{
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 20px;
    }}
    .stButton>button {{
        font-size: 18px;
        padding: 0.5em 1.5em;
        border-radius: 10px;
        background-color: #ff69b4;
        color: white;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #e754a5;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/IMG.png")

# כותרות
st.markdown("<div class='main-title'>🧬 ברוכים הבאים לאפליקציית חיזוי סרטן השד</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>נא לבחור את הפעולה שתרצו לבצע:</div>", unsafe_allow_html=True)

# כפתורים בשורה
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 חיזוי חזרת מחלה"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")

with col2:
    if st.button("🖼️ בדיקת תמונות CT"):
        st.switch_page("pages/2_בדיקת_CT.py")
st.markdown("</div>", unsafe_allow_html=True)
