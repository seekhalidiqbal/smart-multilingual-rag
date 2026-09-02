import requests
from PIL import Image
import streamlit as st

# ==========================================================
# FIXED LOGO PATH & DISPLAY FUNCTION
# ==========================================================

# Direct encoded Raw GitHub URL
LOGO_URL = "https://raw.githubusercontent.com/seekhalidiqbal/rag-assets/main/logo%20University%20of%20Okara.png"

# Header with Embedded Base64 / Clean Direct URL
st.markdown(
    f"""
    <div class="gradio-header">
        <div>
            <div class="gradio-title">🤖 Smart_Multilingual_Multi_Document_AI_RAG_Assistant</div>
            <div class="gradio-sub">💻 Department of Computer Science</div>
            <div class="gradio-meta">🎓 University of Okara • MSCS Research Project &nbsp;|&nbsp; ⚙️ Version 6.24</div>
        </div>
        <div>
            <img src="{LOGO_URL}" style="height: 85px; width: auto; object-fit: contain;" alt="University Logo"/>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
