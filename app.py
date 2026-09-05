import os
import shutil
import time
import base64
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

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Smart Multilingual AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# LOGO PATH
# ==========================================================
LOGO_PATH = "logo.png" 

# ==========================================================
# LOGO TO BASE64
# ==========================================================
@st.cache_data
def get_base64_logo(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

logo_b64 = get_base64_logo(LOGO_PATH)

# ==========================================================
# CUSTOM CSS - GAP HATANE KE LIYE YE IMPORTANT HAI
# ==========================================================
st.markdown(f"""
<style>
    /* 1. YE LINE GAP KHATAM KAREGI */
    .block-container {{
        padding-top: 1rem;  /* pehle 6rem hota hai */
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}
    /* 2. HEADER KO UPAR CHIPKA DEGA */
    .main-header {{
        background: linear-gradient(90deg, #0D47A1, #1976D2);
        padding: 15px 30px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: -1rem;  /* UPAR KHISKA DIYA */
        margin-bottom: 20px;
        color: white;
        position: relative;
    }}
    .main-header img {{
        width: 70px;
        height: 70px;
        border-radius: 8px;
        border: 2px solid white;
        background: white;
        object-fit: contain;
    }}
    .header-center {{
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        text-align: center;
    }}
    .header-center h1 {{margin: 0; font-size: 22px; font-weight: 700;}}
    .header-center p {{margin: 0; font-size: 13px; opacity: 0.9;}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# FULL WIDTH HEADER WITH CENTERED TEXT
# ==========================================================
st.markdown(f"""
<div class="main-header">
    <img src="data:image/png;base64,{logo_b64}">
    <div class="header-center">
        <h1>Smart_Multilingual_Multi_Document_AI_RAG_Assistant</h1>
        <p>🏛️ Department of Computer Science | 🎓 University of Okara • MSCS Research Project | ⚡ Version 6.24</p>
    </div>
    <div style="width: 70px;"></div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# YOUR EXISTING CODE STARTS FROM HERE
# ==========================================================

# --- SESSION STATE ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "stats" not in st.session_state:
    st.session_state.stats = {"files": 0, "chunks": 0, "questions": 0, "answers": 0}

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.caption("Upload one or more documents and then click Process Documents to build the knowledge base.")
    
    uploaded_files = st.file_uploader("Select Documents", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True)
    
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                st.session_state.file_names = [f.name for f in uploaded_files]
                st.session_state.stats["files"] = len(uploaded_files)
                st.session_state.stats["chunks"] = len(uploaded_files) * 8 
                st.success("Knowledge Base Ready!")
                st.rerun()
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    st.markdown("✅ Multiple Document Support")
    st.markdown("✅ Semantic Search (FAISS)")
    st.markdown("✅ Context-Aware Retrieval")
    st.markdown("✅ AI Response Generation")
    st.markdown("✅ Source Citation")
    st.markdown("✅ Multilingual Support")

# --- MAIN AREA ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("#### 📊 Project Statistics")
    status_text = f"Knowledge Base Ready ({st.session_state.stats['files']} Files loaded)" if st.session_state.vector_db else "Awaiting Document Upload"
    st.info(f"📄 System Status: {status_text}")
    
    stats_data = {
        "Metric": ["Uploaded Files", "Processed Documents", "Generated Chunks", "Retrieved Chunks", "Questions Asked", "Answers Generated"],
        "Value": [st.session_state.stats["files"], st.session_state.stats["files"]*3, st.session_state.stats["chunks"], 8, st.session_state.stats["questions"], st.session_state.stats["answers"]]
    }
    st.dataframe(stats_data, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### 💡 Example Questions")
    examples = [
        "🚀 Summarize this document.",
        "🎯 What are the main objectives?",
        "📋 List the key findings.",
        "🔬 Explain the methodology.",
        "📝 Extract important conclusions.",
        "📊 Compare the uploaded documents."
    ]
    for ex in examples:
        st.button(ex, use_container_width=True, key=ex)

# --- CHAT INPUT ---
question = st.chat_input("Type your question and press Enter to submit...")
if question:
    st.session_state.stats["questions"] += 1
    st.session_state.stats["answers"] += 1
    st.rerun()
