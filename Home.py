import streamlit as st


st.set_page_config(page_title="CineHeist", layout="wide")

st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>🎬 Welcome to the CineHeist</h1>
            <h3>Your movie matchmaker</h3>
        <p>Discover movies through recommendations, explore the dataset, or learn more about this app!</p>
    </div>
""", unsafe_allow_html=True)

st.write("### Navigate to different pages using the sidebar.")
st.info("Use the sidebar to explore recommendations or learn more about this app.")
