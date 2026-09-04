import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import shutil
import pandas as pd
import base64

load_dotenv()
st.set_page_config(page_title="Smart Multilingual RAG", layout="wide", initial_sidebar_state="collapsed")

# --- FULL WIDTH HEADER CSS ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f: data = f.read()
        return base64.b64encode(data).decode()
    except: return ""

logo_b64 = get_base64_of_bin_file("logo.png") # logo.png repo me honi chahiye

st.markdown(f"""
<style>
    .stApp {{background-color: #f0f2f6;}}
    .full-header {{
        background-color: #1e4d8b; 
        padding: 10px 30px; 
        margin: -1rem -5rem 1rem -5rem;
        display: flex; 
        align-items: center; 
        gap: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .full-header img {{width: 45px; height: 45px;}}
    .full-header h3 {{color: white; margin: 0; font-size: 20px; font-weight: 600; flex-grow: 1; text-align: center;}}
    .card {{background: white; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 15px;}}
    .stButton>button[kind="primary"] {{background-color: #ff7f00; color: white; border: none; font-weight: 600;}}
</style>
""", unsafe_allow_html=True)

# --- FULL WIDTH BLUE HEADER ---
st.markdown(f"""
<div class="full-header">
    <img src="data:image/png;base64,{logo_b64}">
    <h3>Smart_Multilingual_Multi_Document_AI_RAG_Assistant</h3>
    <div style="width:45px;"></div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
left, right = st.columns([1, 1.8])

# --- LEFT COLUMN ---
with left:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📂 Document Management")
        uploaded_files = st.file_uploader("Drag and drop file here", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True, key="file_uploader", label_visibility="collapsed")
        st.caption("Limit 200MB per file • PDF, DOCX, TXT, CSV, PPTX")
        if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing..."):
                    docs = load_documents(uploaded_files); chunks = create_chunks(docs); emb, vdb = build_vector_database(chunks); ret, llm = build_rag(vdb)
                    st.session_state.chunks = chunks; st.session_state.vector_db, st.session_state.retriever, st.session_state.llm = vdb, ret, llm
                    st.session_state.available_files = [f.name for f in uploaded_files]; st.success("Processed!"); st.rerun()
            else: st.warning("Please upload files first.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 System Status")
        if st.session_state.vector_db is None: st.info("Status: Not Ready")
        else: st.success(f"Status: Ready - {len(st.session_state.available_files)} files")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Project Statistics")
        stats_data = {"Metric": ["Total Files", "Total Chunks", "Embedding Model", "LLM Model", "Vector Store"], "Value": [len(st.session_state.available_files), len(st.session_state.chunks), "multilingual-MiniLM", "gpt-oss-120b", "FAISS"]}
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ℹ️ System Information")
        st.markdown("✅ Multiple Document Support\n✅ Semantic Search\n✅ Context-Aware Retrieval\n✅ Multilingual Support")
        st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT COLUMN ---
with right:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🤖 AI Research Assistant")
        chat_area = st.container(height=350)
        with chat_area:
            if "messages" not in st.session_state: st.session_state.messages = []
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        st.text_input("Ask a Question", key="question_input", label_visibility="collapsed", placeholder="Type your question and press Enter...")
        
        col_l, col_m, col_r = st.columns([2,2,2])
        with col_l: selected_lang = st.selectbox("Language", ["Auto", "English", "Urdu", "Roman Urdu", "Arabic"], label_visibility="collapsed")
        with col_m: 
            if st.button("Read Answer", use_container_width=True): st.info("TTS coming soon")
        with col_r: 
            if st.button("Get Answer", type="primary", use_container_width=True): 
                q = st.session_state.question_input
                if q and st.session_state.vector_db:
                    st.session_state.messages.append({"role": "user", "content": q})
                    result = ask_rag(q, selected_lang)
                    st.session_state.messages.append({"role": "assistant", "content": result["result"]})
                    st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True): st.session_state.messages = []; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📥 Download Answer")
        st.button("Download Last Answer", disabled=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### ❓ Example Questions")
        examples = ["Summarize this document.", "What are the key findings?", "Explain the methodology.", "Compare the documents.", "List important dates."]
        for ex in examples:
            if st.button(f"• {ex}", key=ex): 
                st.session_state.question_input = ex; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<center style='color:gray; font-size:12px; margin-top:20px;'>© 2026 Department of Computer Science • University of Okara</center>", unsafe_allow_html=True)

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
    shutil.rmtree(temp_dir); return docs
def create_chunks(docs): return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
def build_vector_database(chunks): embeddings = get_embeddings(); vector_db = FAISS.from_documents(chunks, embeddings); return embeddings, vector_db
def build_rag(vector_db): retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8}); llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1400, groq_api_key=os.getenv("GROQ_API_KEY")); return retriever, llm
def ask_rag(question, selected_lang):
    retriever = st.session_state.retriever; llm = st.session_state.llm; source_documents = retriever.invoke(question)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(source_documents)])
    lang_instruction = f"Answer in {selected_lang}." if selected_lang!= "Auto" else "Answer in same language as question."
    prompt = f"Answer strictly from context. {lang_instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    response = llm.invoke(prompt); return {"result": response.content, "source_documents": source_documents}

if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "available_files" not in st.session_state: st.session_state.available_files = []
if "chunks" not in st.session_state: st.session_state.chunks = []
