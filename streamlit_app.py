import streamlit as st
import base64

st.set_page_config(page_title="חיזוי סרטן השד", layout="wide")

def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}
    html, body, [class*="css"] {{
        direction: rtl;
        text-align: center;
    }}
    .main-title {{
        font-size: 50px;
        font-weight: bold;
        color: #2E2E2E;
        margin-top: 60px;
        margin-bottom: 10px;
    }}
    .subtitle {{
        font-size: 24px;
        color: #444;
        margin-bottom: 40px;
    }}
    .button-container {{
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 20px;
    }}
    .stButton>button {{
        font-size: 18px;
        padding: 0.7em 1.8em;
        border-radius: 12px;
        background-color: #ff69b4;
        color: white;
        border: none;
        font-weight: bold;
        transition: background-color 0.3s ease;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    .stButton>button:hover {{
        background-color: #e754a5;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ✅ רקע
set_background("images/IMG.png")

# ✅ כותרת + תת-כותרת
st.markdown("<div class='main-title'>שלום רופא יקר 🩺</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>יש לבחור אחת מהאופציות הבאות:</div>", unsafe_allow_html=True)

# ✅ כפתורים
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 חיזוי חזרת מחלה"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")
with col2:
    if st.button("🖼️ בדיקת תמונות CT"):
        st.switch_page("pages/2_בדיקת_CT.py")
st.markdow
