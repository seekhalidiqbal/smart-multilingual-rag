import streamlit as st
import os
import tempfile
import time
import base64 # 1. ye import add karo
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import shutil
import pandas as pd
from io import BytesIO
from docx import Document
from fpdf import FPDF

load_dotenv()

st.set_page_config(
    page_title="Smart Multilingual AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# LOGO PATH
# ==========================================================
LOGO_PATH = "logo.png"  # logo.png file repo me honi chahiye

# ==========================================================
# LOGO TO BASE64 FUNCTION - YE SABSE IMPORTANT HAI
# ==========================================================
@st.cache_data
def get_base64_logo(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return "" # agar logo na mile to blank

logo_b64 = get_base64_logo(LOGO_PATH)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown(f"""
<style>
    .stApp {{background: #f8fafc;}}
    .header {{background: linear-gradient(90deg, #0D47A1, #1976D2); padding: 20px 40px; margin: -1rem -5rem 1rem -5rem; display: flex; align-items: center; gap: 20px;}}
    .header img {{width: 80px; height: 80px; border-radius: 10px; border: 3px solid white; object-fit: contain; background:white;}}
    .header h1 {{color: white; margin: 0; font-size: 32px;}}
    .header h2 {{color: #BBDEFB; margin: 0; font-size: 18px; font-weight: 400;}}
    .header h3 {{color: #E3F2FD; margin: 0; font-size: 15px; font-weight: 300;}}
    .version-badge {{background: white; border: 3px solid red; border-radius: 15px; padding: 10px 20px; text-align: center; margin-left: auto;}}
    .version-badge b {{font-size: 14px;}} .version-badge p {{margin:0; font-size: 20px; font-weight: bold;}}
    .card {{background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 15px;}}
    .stButton>button[kind="primary"] {{background-color: #FF6F00; color: white; border: none; font-weight: 600; height: 45px; font-size: 16px;}}
    .stButton>button {{height: 45px;}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# FULL WIDTH HEADER WITH EMBEDDED LOGO
# ==========================================================
st.markdown(f"""
<div class="header">
    <img src="data:image/png;base64,{logo_b64}">
    <div>
        <h1>🤖 Smart Multilingual <b>AI RAG</b> Assistant</h1>
        <h2>Department of Computer Science</h2>
        <h3>University of Okara • MSCS Research Project</h3>
    </div>
    <div class="version-badge">
        <b>Version</b>
        <p>6.02</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---
@st.cache_resource
def get_embeddings(): return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_documents(uploaded_files):
    docs = []; temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
        if uploaded_file.name.endswith(".pdf"): loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"): loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".txt"): loader = TextLoader(temp_path, encoding="utf-8")
        elif uploaded_file.name.endswith(".csv"): loader = CSVLoader(temp_path, encoding="utf-8")
        elif uploaded_file.name.endswith(".pptx"): loader = UnstructuredPowerPointLoader(temp_path)
        else: continue
        loaded_docs = loader.load()
        for doc in loaded_docs: doc.metadata["source"] = uploaded_file.name
        docs.extend(loaded_docs)
    st.session_state.documents = docs # for page count
    shutil.rmtree(temp_dir); return docs

def create_chunks(docs): return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
def build_vector_database(chunks): embeddings = get_embeddings(); vector_db = FAISS.from_documents(chunks, embeddings); return embeddings, vector_db
def build_rag(vector_db): retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8}); llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1400, groq_api_key=os.getenv("GROQ_API_KEY")); return retriever, llm

def ask_rag(question, selected_lang):
    start_time = time.time()
    retriever = st.session_state.retriever; llm = st.session_state.llm; source_documents = retriever.invoke(question)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(source_documents)])
    lang_instruction = f"Answer in {selected_lang}." if selected_lang!= "Auto" else "Answer in same language as question."
    prompt = f"Answer strictly from context. {lang_instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    response = llm.invoke(prompt)
    response_time = round(time.time() - start_time, 2)
    return {"result": response.content, "source_documents": source_documents, "response_time": response_time}

def to_txt(text): return text.encode('utf-8')
def to_pdf(text):
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12); pdf.multi_cell(200, 10, txt=text)
    return pdf.output(dest='S').encode('latin-1')
def to_docx(text):
    doc = Document(); doc.add_paragraph(text)
    buf = BytesIO(); doc.save(buf); return buf.getvalue()

# --- SESSION STATE ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "available_files" not in st.session_state: st.session_state.available_files = []
if "chunks" not in st.session_state: st.session_state.chunks = []
if "last_answer" not in st.session_state: st.session_state.last_answer = ""
if "response_time" not in st.session_state: st.session_state.response_time = 0
if "retrieved_chunks" not in st.session_state: st.session_state.retrieved_chunks = 0

# --- LAYOUT ---
left, right = st.columns([1, 1.8])

# --- LEFT COLUMN ---
with left:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📂 Upload Documents")
        uploaded_files = st.file_uploader("", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True, key="file_uploader")
        if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing..."):
                    docs = load_documents(uploaded_files); chunks = create_chunks(docs); emb, vdb = build_vector_database(chunks); ret, llm = build_rag(vdb)
                    st.session_state.chunks = chunks; st.session_state.vector_db, st.session_state.retriever, st.session_state.llm = vdb, ret, llm
                    st.session_state.available_files = [f.name for f in uploaded_files]; st.success(f"{len(uploaded_files)} files processed!"); st.rerun()
            else: st.warning("Please upload files first.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Project Statistics")
        pages = sum([len(d.page_content)//1000 + 1 for d in st.session_state.get('documents', [])]) if 'documents' in st.session_state else 0
        stats_data = {
            "Metric": ["📁 Uploaded Files", "📄 Processed Pages", "🧩 Generated Chunks", "🔍 Retrieved Chunks", "⚡ Response Time", "🧠 Model"],
            "Value": [len(st.session_state.available_files), pages, len(st.session_state.chunks), st.session_state.retrieved_chunks, f"{st.session_state.response_time} sec", "Groq Llama"]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ System Information")
        st.markdown("✅ Multiple Document Support\n✅ Semantic Search FAISS\n✅ Context-Aware RAG\n✅ Multilingual Support")
        st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT COLUMN ---
with right:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Conversation")
        chat_area = st.container(height=250)
        with chat_area:
            if "messages" not in st.session_state: st.session_state.messages = []
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ❓ Ask Your Question")
        question = st.text_input("", placeholder="Example: Summarize the uploaded document...", key="question_input", label_visibility="collapsed")
        selected_lang = st.selectbox("Answer Language", ["Auto", "English", "Urdu", "Roman Urdu", "Arabic"])
        
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("🚀 Get Answer", type="primary", use_container_width=True):
                if question and st.session_state.vector_db:
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("Thinking..."):
                        result = ask_rag(question, selected_lang)
                        st.session_state.last_answer = result["result"]
                        st.session_state.response_time = result["response_time"]
                        st.session_state.retrieved_chunks = len(result["source_documents"])
                        st.session_state.messages.append({"role": "assistant", "content": result["result"]})
                    st.rerun()
                else: st.warning("Please upload and process documents first.")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True): st.session_state.messages = []; st.session_state.last_answer=""; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ⬇️ Download Answer")
        if st.session_state.last_answer:
            col_t, col_p, col_d = st.columns(3)
            with col_t: st.download_button("📄 Download TXT", to_txt(st.session_state.last_answer), "rag_answer.txt", use_container_width=True)
            with col_p: st.download_button("📕 Download PDF", to_pdf(st.session_state.last_answer), "rag_answer.pdf", use_container_width=True)
            with col_d: st.download_button("📘 Download DOCX", to_docx(st.session_state.last_answer), "rag_answer.docx", use_container_width=True)
        else: st.info("Answer will appear here after you ask a question.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<center style='color:gray; font-size:12px; margin-top:20px;'>© 2026 Department of Computer Science • University of Okara</center>", unsafe_allow_html=True)
