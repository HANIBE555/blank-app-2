import streamlit as st

# עיצוב RTL
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl !important;
        text-align: right !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>שלום רופא יקר <img src='https://cdn-icons-png.flaticon.com/512/3774/3774299.png' width='40'></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>יש לבחור אחת מהאופציות הבאות:</p>", unsafe_allow_html=True)

# שני כפתורים במרכז
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🖼 בדיקת תמונות CT"):
        switch_page("2_בדיקת_CT")
with col2:
    st.empty()
with col3:
    if st.button("🔍 חיזוי חזרת מחלה"):
        switch_page("1_חיזוי_חזרת_מחלה")
