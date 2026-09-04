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

load_dotenv()

st.set_page_config(page_title="Smart Multilingual RAG", layout="wide", page_icon="🧠")

# --- CUSTOM CSS FOR GRADIO LIKE UI ---
st.markdown("""
<style>
    .stApp {background-color: #f5f5f5;}
    [data-testid="stSidebar"] {background-color: #ffffff; border-right: 1px solid #ddd;}
    .block-container {padding-top: 1rem;}
    .header-bar {background-color: #1e4d8b; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px;}
    .stButton>button[kind="primary"] {background-color: #ff7f00; color: white; border: none;}
</style>
""", unsafe_allow_html=True)

# --- HEADER WITH LOGO ---
col_logo, col_title, col_empty = st.columns([1, 6, 1])
with col_logo:
    st.image("logo.png", width=50)  # GitHub me logo.png upload karna hai
with col_title:
    st.markdown("<div class='header-bar'><h3 style='color:white; text-align:center; margin:0;'>Smart_Multilingual_Multi_Document_AI_RAG_Assistant</h3></div>", unsafe_allow_html=True)

st.title("🧠 AI Research Assistant")

# --- FUNCTIONS ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_documents(uploaded_files):
    docs = []
    temp_dir = tempfile.mkdtemp()
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
    shutil.rmtree(temp_dir)
    return docs

def create_chunks(docs): return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
def build_vector_database(chunks): embeddings = get_embeddings(); vector_db = FAISS.from_documents(chunks, embeddings); return embeddings, vector_db
def build_rag(vector_db): retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25}); llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1400, groq_api_key=os.getenv("GROQ_API_KEY")); return retriever, llm

def ask_rag(question, selected_lang):
    retriever = st.session_state.retriever; llm = st.session_state.llm
    source_documents = retriever.invoke(question)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(source_documents)])
    lang_instruction = f"Answer in {selected_lang} language." if selected_lang!= "Auto" else "Answer in the same language as question."
    prompt = f"Answer strictly from context. {lang_instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    response = llm.invoke(prompt)
    return {"result": response.content, "source_documents": source_documents}

# --- SESSION STATE ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "available_files" not in st.session_state: st.session_state.available_files = []
if "chunks" not in st.session_state: st.session_state.chunks = []

# --- LAYOUT: 2 COLUMNS ---
left_col, right_col = st.columns([1, 2])

# --- LEFT COLUMN: Document Management + Stats ---
with left_col:
    st.markdown("### 📂 Document Management")
    uploaded_files = st.file_uploader("Drag and drop files here", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True, key="file_uploader", label_visibility="collapsed")
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                docs = load_documents(uploaded_files); chunks = create_chunks(docs); emb, vdb = build_vector_database(chunks); ret, llm = build_rag(vdb)
                st.session_state.chunks = chunks; st.session_state.vector_db, st.session_state.retriever, st.session_state.llm = vdb, ret, llm
                st.session_state.available_files = [f.name for f in uploaded_files]; st.success("Done!"); st.rerun()
        else: st.warning("Please upload files first.")
    
    st.markdown("### 📊 System Status")
    if st.session_state.vector_db is None: st.info("Awaiting upload")
    else: st.success(f"{len(st.session_state.available_files)} files loaded")

    st.markdown("### 📈 Project Statistics")
    stats_data = {
        "Metric": ["Files", "Chunks", "Model", "Embeddings"],
        "Count": [len(st.session_state.available_files), len(st.session_state.chunks), "gpt-oss-120b", "multilingual-MiniLM"]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    st.markdown("### ℹ️ System Information")
    st.markdown("- Multiple Document Support\n- Semantic Search FAISS\n- Context-Aware RAG\n- Multilingual Support")

# --- RIGHT COLUMN: Chat + Language + Examples ---
with right_col:
    st.markdown("### 💬 AI Research Assistant")
    
    chat_box = st.container(height=300)
    with chat_box:
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])

    selected_lang = st.selectbox("🌐 Answer Language", ["Auto", "English", "Urdu", "Roman Urdu", "Arabic"])
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1: 
        if st.button("🗑️ Clear Chat", use_container_width=True): st.session_state.messages = []; st.rerun()
    with col_btn2:
        if st.button("📥 Download Answer", use_container_width=True): st.info("Download feature coming soon")
    
    question = st.chat_input("Ask a question about your documents...")
    
    # Example Questions
    st.markdown("### ❓ Example Questions")
    examples = ["Summarize this document", "What are the key findings?", "Explain the methodology", "Compare all documents"]
    for ex in examples:
        if st.button(f"• {ex}", key=ex): st.session_state.example_q = ex; st.rerun()

    if "example_q" in st.session_state: question = st.session_state.example_q; del st.session_state.example_q

    if question:
        if st.session_state.vector_db is None: st.warning("Please upload and process documents first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                result = ask_rag(question, selected_lang)
                st.session_state.messages.append({"role": "assistant", "content": result["result"]})
            st.rerun()

st.markdown("---")
st.markdown("<center>© 2026 Department of Computer Science • University of Okara</center>", unsafe_allow_html=True)
