import streamlit as st
import base64

# הגדרת רקע ע"י CSS
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .title {{
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .subtitle {{
        text-align: center;
        font-size: 24px;
        margin-bottom: 30px;
    }}
    .button-container {{
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }}
    .stButton>button {{
        background-color: #f94ca4;
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 10px;
        border: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# רקע
set_background("images/IMG.png")

# כותרות
st.markdown('<div class="title">שלום רופא יקר 🩺</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">יש לבחור אחת מהאופציות הבאות:</div>', unsafe_allow_html=True)

# כפתורים
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🔍 חיזוי חזרת מחלה"):
        st.switch_page("pages/1_חיזוי_חזרת_מחלה.py")

with col3:
    if st.button("🖼️ בדיקת תמונות CT"):
        st.switch_page("pages/2_בדיקת_CT.py")
