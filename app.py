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

# --- HEADER WITH LOGO ---
col1, col2 = st.columns([4,1])
with col1:
    st.markdown("🏛️ **University of Okara | MS/CS Research Project | Version 2.4**")
with col2:
    # Yahan apna logo ka URL daal do. Agar nahi hai to ye line delete kar do
    st.image("https://uo.edu.pk/wp-content/uploads/2021/05/UO-Logo.png", width=100)

st.title("🧠 Smart Multilingual RAG Assistant")

# --- FUNCTIONS ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_documents(uploaded_files):
    docs = []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path, encoding="utf-8")
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(temp_path, encoding="utf-8")
        elif uploaded_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(temp_path)
        else:
            continue
        
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = uploaded_file.name
        docs.extend(loaded_docs)
    shutil.rmtree(temp_dir)
    return docs

def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def build_vector_database(chunks):
    embeddings = get_embeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    return embeddings, vector_db

def build_rag(vector_db):
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.75})
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1400, groq_api_key=os.getenv("GROQ_API_KEY"))
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
    prompt = f"You are Smart Multilingual AI RAG Assistant.\nAnswer strictly from the uploaded document context.\nRULES:\n1. Grounding score must be 100%. No outside knowledge.\n2. {lang_instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    response = llm.invoke(prompt)
    return {"result": response.content, "source_documents": source_documents}

# --- SESSION STATE ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "available_files" not in st.session_state:
    st.session_state.available_files = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📂 Document Management")
    st.caption("Upload one or more documents and then click Process Documents to build the knowledge base.")

    uploaded_files = st.file_uploader("📁 Select Documents", type=["pdf", "docx", "txt", "csv", "pptx"], accept_multiple_files=True, key="file_uploader")

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
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
        else:
            st.warning("Please upload files first.")
    
    st.divider()
    st.markdown("### ℹ️ System Information")
    st.markdown("📚 **Multiple Document Support**\n🔎 **Semantic Search (FAISS)**\n🧠 **Context-Aware Retrieval**\n🤖 **AI Response Generation**")

# --- MAIN CONTENT ---
if st.session_state.vector_db is None:
    st.info("📋 **System Status:** Awaiting document upload & processing...")
else:
    st.success(f"📋 **System Status:** Knowledge Base Ready - {len(st.session_state.available_files)} files loaded")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### ☑️ Project Statistics")
    stats_data = {
        "Metric": ["Uploaded Files", "Processed Documents", "Generated Chunks", "Model"],
        "Value": [len(st.session_state.available_files), len(st.session_state.available_files), len(st.session_state.chunks), "openai/gpt-oss-120b"]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 💡 Example Questions")
    c1, c2, c3 = st.columns(1)
    if c1.button("📍 Summarize this document.", use_container_width=True):
        st.session_state.example_q = "Summarize this document."
    if c2.button("🎯 What are the main objectives?", use_container_width=True):
        st.session_state.example_q = "What are the main objectives?"
    if c3.button("📄 List the key findings.", use_container_width=True):
        st.session_state.example_q = "List the key findings."
    
    selected_lang = st.selectbox("Answer Language", ["Auto / Detect", "English", "Urdu", "Roman Urdu", "Arabic"])

st.divider()
st.markdown("### 💬 AI Conversation")

if "example_q" in st.session_state:
    question = st.session_state.example_q
    del st.session_state.example_q
else:
    question = st.text_input("Type your question and press Enter to submit...", key="question_input")

if st.button("Get Answer", type="primary"):
    if question:
        if st.session_state.vector_db is None:
            st.warning("Please upload and process documents first.")
        else:
            with st.spinner("Thinking..."):
                result = ask_rag(question, selected_lang)
                st.markdown("#### Answer")
                st.write(result["result"])
                
                with st.expander("📄 Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:500] + "...")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("<center>© 2026 Smart Multilingual AI RAG Assistant. All rights reserved.<br>Department of Computer Science • University of Okara</center>", unsafe_allow_html=True)
