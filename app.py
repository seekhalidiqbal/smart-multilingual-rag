import os
import shutil
import time
from pathlib import Path
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
)
from langchain_groq import ChatGroq

st.set_page_config(page_title="Smart Multilingual AI RAG Assistant", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
LOGO_PATH = "logo.png"

st.markdown("""
    <style>
  .stApp{background:#f8fafc;}
  .block-container{max-width:1400px;padding-top:1rem;}
  .header-box{background:#ffffff;padding:20px;border-radius:16px;border:1px solid #e2e8f0;box-shadow:0 4px 12px rgba(0,0,0,0.03);margin-bottom:20px;}
  .header-title{color:#0f172a;font-size:26px;font-weight:800;line-height:1.3;}
  .header-sub{color:#475569;font-size:15px;font-weight:600;margin-top:4px;}
  .header-meta{color:#64748b;font-size:13px;margin-top:4px;}
  .status-card{padding:14px 18px;border-radius:12px;font-weight:600;margin-bottom:20px;font-size:14px;}
  .status-ready{background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;}
  .status-waiting{background:#fffbeb;border:1px solid #fef3c7;color:#92400e;}
  .section-title{font-size:16px;font-weight:700;color:#0f172a;margin-bottom:12px;}
  .footer-gradio{text-align:center;color:#64748b;font-size:12px;padding:25px 0;border-top:1px solid #e2e8f0;margin-top:30px;}
    </style>
    """, unsafe_allow_html=True)

defaults = {"documents": [], "chunks": [], "vector_db": None, "retriever": None, "llm": None, "available_files": [], "chat_history": [], "last_retrieved": 0, "last_response_time": 0.0, "questions_asked": 0, "answers_generated": 0, "last_language": "Auto / Detect", "preset_question": None}
for key, value in defaults.items():
    if key not in st.session_state: st.session_state[key] = value

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
try: GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception: GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================= HEADER 100% WORKING =================
with st.container():
    st.markdown('<div class="header-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="header-title">🤖 Smart Multilingual Multi Document AI RAG Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-sub">💻 Department of Computer Science</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-meta">🎓 University of Okara • MSCS Research Project | ⚙️ Version 6.24</div>', unsafe_allow_html=True)
    with col2:
        if os.path.exists(LOGO_PATH): 
            st.image(LOGO_PATH, width=110)
        else: 
            st.warning("Logo.png not found")
    st.markdown('</div>', unsafe_allow_html=True)
# =======================================================

# FUNCTIONS
def load_documents(files):
    documents = []
    if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for uploaded_file in files:
        filename = uploaded_file.name
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as file: file.write(uploaded_file.getbuffer())
        extension = Path(filename).suffix.lower()
        loaders = {".pdf": PyPDFLoader, ".docx": UnstructuredWordDocumentLoader, ".txt": TextLoader, ".csv": CSVLoader, ".pptx": UnstructuredPowerPointLoader}
        if extension in loaders: 
            docs = loaders[extension](save_path).load()
            for doc in docs: doc.metadata["source"] = filename
            documents.extend(docs)
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    return splitter.split_documents(documents)

def build_vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return embedding_model, vector_db

def build_rag(vector_db):
    if not GROQ_API_KEY: raise Exception("GROQ_API_KEY is not configured in secrets.")
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.70})
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1400, groq_api_key=GROQ_API_KEY)
    return retriever, llm

def ask_rag(question, selected_lang):
    retriever = st.session_state.retriever
    llm = st.session_state.llm
    source_documents = retriever.invoke(question)
    context_parts = []
    for idx, doc in enumerate(source_documents, start=1):
        src = os.path.basename(str(doc.metadata.get("source", "Unknown")))
        page = doc.metadata.get("page")
        page_str = f" (Page {int(page)+1})" if page is not None else ""
        context_parts.append(f"[{idx}] Document: {src}{page_str}\n{doc.page_content}")
    context = "\n\n".join(context_parts)
    lang_instruction = f"Answer explicitly in {selected_lang} language." if selected_lang!= "Auto / Detect" else "Answer in the same language as the user question."
    prompt = f"You are Smart Multilingual AI RAG Assistant.\nAnswer strictly from the uploaded document context.\nRULES:\n1. Grounding score must be 100%. No outside knowledge.\n2. {lang_instruction}\n3. If not found, reply: 'I couldn't find the answer in the uploaded documents.'\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"
    response = llm.invoke(prompt)
    return {"result": response.content, "source_documents": source_documents}

# SIDEBAR
with st.sidebar:
    st.markdown("### 📂 Document Management")
       uploaded_files = st.file_uploader("📁 Select Documents", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True, key="file_uploader")
        if uploaded_files:
            with st.spinner("Processing..."):
                docs = load_documents(uploaded_files)
                chunks = create_chunks(docs)
                emb, vdb = build_vector_database(chunks)
                ret, llm = build_rag(vdb)
                st.session_state.documents, st.session_state.chunks = docs, chunks
                st.session_state.vector_db, st.session_state.retriever, st.session_state.llm = vdb, ret, llm
                st.session_state.available_files = [f.name for f in uploaded_files]
                st.success("Documents Processed Successfully!")
                st.rerun()
        else: st.warning("Please upload files first.")
    st.divider()
    st.markdown("### ℹ️ System Information")
    st.markdown("📚 **Multiple Document Support**\n🔎 **Semantic Search (FAISS)**\n🧠 **Context-Aware Retrieval**\n🤖 **AI Response Generation**")

# MAIN
if st.session_state.vector_db is not None: 
    st.markdown(f'<div class="status-card status-ready">📊 System Status: Knowledge Base Ready ({len(st.session_state.available_files)} Files loaded)</div>', unsafe_allow_html=True)
else: 
    st.markdown('<div class="status-card status-waiting">📊 System Status: Awaiting document upload & processing...</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown('<div class="section-title">📈 Project Statistics</div>', unsafe_allow_html=True)
    st.table([("📂 Uploaded Files", str(len(st.session_state.available_files))), ("📄 Processed Documents", str(len(st.session_state.documents))), ("🧩 Generated Chunks", str(len(st.session_state.chunks))), ("🤖 Model", "openai/gpt-oss-120b")])
with col_right:
    st.markdown('<div class="section-title">💡 Example Questions</div>', unsafe_allow_html=True)
    for ex in ["📌 Summarize this document.", "🎯 What are the main objectives?", "📋 List the key findings."]:
        if st.button(ex, use_container_width=True): st.session_state.preset_question = ex.split(" ", 1)[1]
    selected_lang = st.selectbox("Select Response Language", ["Auto / Detect", "Urdu", "English", "Arabic", "Spanish"], label_visibility="collapsed")

st.divider()
st.markdown('<div class="section-title">💬 AI Conversation</div>', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

question_input = st.chat_input("🔎 Type your question and press Enter to submit...")
active_question = question_input or st.session_state.preset_question
if active_question:
    st.session_state.preset_question = None
    if st.session_state.vector_db is None: st.warning("⚠️ Please upload and process documents first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": active_question})
        with st.chat_message("user"): st.markdown(active_question)
        with st.chat_message("assistant"):
            with st.spinner("🚀 Getting Answer..."):
                start_time = time.time()
                res = ask_rag(active_question, selected_lang)
                answer = res["result"]
                sources = res["source_documents"]
                elapsed = round(time.time() - start_time, 2)
                response_text = f"{answer}\n\n**📄 Sources**\n" + "\n".join([f"• {os.path.basename(str(doc.metadata.get('source','Unknown')))}" for doc in sources])
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.session_state.questions_asked += 1
                st.session_state.answers_generated += 1
                st.session_state.last_retrieved = len(sources)
                st.session_state.last_response_time = elapsed
                st.session_state.last_language = selected_lang if selected_lang!= "Auto / Detect" else "Detected"
                st.rerun()

if st.session_state.chat_history:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("""<div class="footer-gradio">© 2026 Smart Multilingual AI RAG Assistant. All rights reserved.<br>Department of Computer Science • University of Okara</div>""", unsafe_allow_html=True)
