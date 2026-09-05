import os
import shutil
import time
import base64
import tempfile
import pandas as pd
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
from langchain.chains.retrieval_qa.base import RetrievalQA # <-- THEEK KIYA HUA
from langchain.prompts import PromptTemplate

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
# CUSTOM CSS - FIXED
# ==========================================================
st.markdown(f"""
<style>
    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}
    .main-header {{
        background: linear-gradient(90deg, #0D47A1, #1976D2);
        padding: 15px 30px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 20px;
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
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="stSidebar"] {{min-width: 350px;}}
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
# SESSION STATE
# ==========================================================
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
if "file_names" not in st.session_state: st.session_state.file_names = []
if "messages" not in st.session_state: st.session_state.messages = []
if "retrieved" not in st.session_state: st.session_state.retrieved = 0
if "stats" not in st.session_state: st.session_state.stats = {"files": 0, "chunks": 0, "questions": 0, "answers": 0}

# ==========================================================
# FUNCTIONS
# ==========================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_loader(file_path, file_type):
    if file_type == "pdf": return PyPDFLoader(file_path)
    elif file_type == "txt": return TextLoader(file_path, encoding="utf-8")
    elif file_type == "csv": return CSVLoader(file_path)
    elif file_type == "docx": return UnstructuredWordDocumentLoader(file_path)
    elif file_type == "pptx": return UnstructuredPowerPointLoader(file_path)
    else: return None

def process_documents(uploaded_files):
    docs = []; temp_dir = tempfile.TemporaryDirectory()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        file_type = uploaded_file.name.split('.')[-1].lower()
        loader = get_loader(file_path, file_type)
        if loader: docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    st.session_state.stats["chunks"] = len(chunks)
    st.session_state.file_names = [f.name for f in uploaded_files]
    st.session_state.stats["files"] = len(uploaded_files)

    embeddings = load_embeddings()
    st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)

    groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    if not groq_api_key: st.error("GROQ_API_KEY not found in Secrets"); return

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
    PROMPT = PromptTemplate(template="Context: {context}\nQuestion: {question}\nAnswer in same language.", input_variables=["context", "question"])
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vector_db.as_retriever(k=4), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    temp_dir.cleanup()

def get_answer(query):
    if not st.session_state.qa_chain: return "Please upload and process documents first.", []
    result = st.session_state.qa_chain({"query": query})
    st.session_state.retrieved = len(result["source_documents"])
    sources = [f"{doc.metadata.get('source','Doc')}: {doc.page_content[:200]}..." for doc in result["source_documents"]]
    return result["result"], sources

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.caption("Upload one or more documents and then click Process Documents to build the knowledge base.")
    uploaded_files = st.file_uploader("Select Documents", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True)
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."): 
                process_documents(uploaded_files)
                st.success("Knowledge Base Ready!")
                st.rerun()
        else: st.warning("Please upload files first.")
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    for i in ["Multiple Document Support","Semantic Search (FAISS)","Context-Aware Retrieval","AI Response Generation","Source Citation","Multilingual Support"]: 
        st.markdown(f"✅ {i}")

# ==========================================================
# MAIN AREA
# ==========================================================
col1, col2 = st.columns([1.5, 1])
with col1:
    st.markdown("#### 📊 Project Statistics")
    status_text = f"Knowledge Base Ready ({st.session_state.stats['files']} Files loaded)" if st.session_state.vector_db else "Awaiting Document Upload"
    st.info(f"📄 System Status: {status_text}")
    df = pd.DataFrame({"Metric": ["Uploaded Files","Processed Documents","Generated Chunks","Retrieved Chunks","Questions Asked","Answers Generated"],"Value": [st.session_state.stats["files"],st.session_state.stats["files"],st.session_state.stats["chunks"],st.session_state.retrieved,st.session_state.stats["questions"],st.session_state.stats["answers"]]})
    st.dataframe(df, use_container_width=True, hide_index=True)
with col2:
    st.markdown("#### 💡 Example Questions")
    for ex in ["🚀 Summarize this document.","🎯 What are the main objectives?","📋 List the key findings.","🔬 Explain the methodology."]:
        if st.button(ex, use_container_width=True, key=ex): 
            st.session_state.messages.append({"role": "user", "content": ex[2:]})
            st.rerun()

# ==========================================================
# CHAT INTERFACE
# ==========================================================
st.markdown("---")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "sources" in m and m["sources"]:
            with st.expander("📚 Sources"):
                for s in m["sources"]: st.write(s)

if prompt := st.chat_input("Type your question and press Enter to submit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = get_answer(prompt)
            st.markdown(response)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources: st.write(s)
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
    st.session_state.stats["questions"] += 1
    st.session_state.stats["answers"] += 1
    st.rerun()
