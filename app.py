import os
import shutil
import time
from pathlib import Path

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_groq import ChatGroq


st.set_page_config(
    page_title="Smart Multilingual AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_PATH = "logo.png"

st.markdown(
    """
    <style>
   .stApp { background: #f8fafc; }
   .block-container { max-width: 1400px; padding-top: 1rem; padding-bottom: 2rem; }
   .gradio-title { color: #0f172a; font-size: 28px; font-weight: 800; }
   .gradio-sub { color: #475569; font-size: 15px; font-weight: 600; margin-top: 4px; }
   .gradio-meta { color: #64748b; font-size: 13px; margin-top: 4px; }
   .status-card { padding: 14px 18px; border-radius: 12px; font-weight: 600; margin-bottom: 20px; font-size: 14px; }
   .status-ready { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
   .status-waiting { background: #fffbebfb; border: 1px solid #fef3c7; color: #92400e; }
   .section-title { font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 12px; }
   .footer-gradio { text-align: center; color: #64748b; font-size: 12px; padding: 25px 0 10px 0; border-top: 1px solid #e2e8f0; margin-top: 30px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# SESSION STATE
defaults = {"documents": [], "chunks": [], "embedding_model": None, "vector_db": None, "retriever": None, "llm": None, "available_files": [], "chat_history": [], "processing_time": 0.0, "last_retrieved": 0, "last_response_time": 0.0, "questions_asked": 0, "answers_generated": 0, "last_language": "Auto / Detect", "preset_question": None}
for key, value in defaults.items():
    if key not in st.session_state: st.session_state[key] = value

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
try: GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception: GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# HEADER
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<div class="gradio-title">🤖 Smart_Multilingual_Multi_Document_AI_RAG_Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradio-sub">💻 Department of Computer Science</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradio-meta">🎓 University of Okara • MSCS Research Project &nbsp;|&nbsp; ⚙️ Version 6.24</div>', unsafe_allow_html=True)
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.caption("Logo not found")
st.markdown('<div style="border: 1px solid #e2e8f0; border-radius: 16px; padding: 1px; background: #ffffff; margin: 20px 0; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);"></div>', unsafe_allow_html=True)


# BAQI TUMHARA SARA CODE YAHAN WESA HI RAHEGA
# load_documents, create_chunks, build_vector_database, build_rag, ask_rag...
# Sidebar, Main, Chat, Footer... sab same
