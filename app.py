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

LOGO_PATH = (
    "https://raw.githubusercontent.com/"
    "seekhalidiqbal/rag-assets/main/"
    "logo%20University%20of%20Okara.png"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #2563eb 100%);
        padding: 30px 35px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }

    .hero-content {
        flex: 1;
    }

    .hero-title {
        color: white;
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 18px;
        font-weight: 600;
        margin-top: 8px;
    }

    .hero-small {
        color: #bfdbfe;
        font-size: 15px;
        margin-top: 7px;
    }

    .hero-logo {
        max-height: 100px;
        width: auto;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 6px;
    }

    .status-ready {
        background: #ecfdf5;
        border: 1px solid #10b981;
        color: #047857;
        padding: 13px 17px;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .status-waiting {
        background: #fff7ed;
        border: 1px solid #f59e0b;
        color: #b45309;
        padding: 13px 17px;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 20px;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin-bottom: 15px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 7px;
    }

    .card-text {
        color: #64748b;
        font-size: 14px;
        line-height: 1.6;
    }

    .stat-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px 12px;
        text-align: center;
        min-height: 115px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .stat-icon {
        font-size: 25px;
    }

    .stat-number {
        font-size: 28px;
        font-weight: 800;
        color: #2563eb;
        margin-top: 4px;
    }

    .stat-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }

    .source-box {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 8px;
        color: #0f172a;
    }

    .capability-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 18px;
        min-height: 190px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .capability-title {
        color: #0f172a;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .capability-text {
        color: #64748b;
        font-size: 14px;
        line-height: 1.8;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        padding: 20px;
        margin-top: 30px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

defaults = {
    "documents": [],
    "chunks": [],
    "embedding_model": None,
    "vector_db": None,
    "retriever": None,
    "llm": None,
    "available_files": [],
    "chat_history": [],
    "processing_time": 0.0,
    "last_retrieved": 0,
    "last_response_time": 0.0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# CONFIGURATION
# ==========================================================

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ==========================================================
# GROQ API KEY
# ==========================================================

try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================================
# HERO
# ==========================================================

st.markdown(
    f"""<div class="hero">
        <div class="hero-content">
            <div class="hero-title">🤖 Smart Multilingual AI RAG Assistant</div>
            <div class="hero-subtitle">Intelligent Multi-Document Question Answering System</div>
            <div class="hero-small">Department of Computer Science • University of Okara • MSCS Research Project</div>
        </div>
        <div>
            <img src="{LOGO_PATH}" class="hero-logo" alt="University of Okara Logo" />
        </div>
    </div>""",
    unsafe_allow_html=True,
)


# ==========================================================
# SYSTEM STATUS
# ==========================================================

if st.session_state.vector_db is not None:
    st.markdown(
        """<div class="status-ready">
            🟢 System Ready — Knowledge Base is active and ready for questions.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<div class="status-waiting">
            🟠 Waiting for Documents — Upload documents from the sidebar and click <b>Process Documents</b>.
        </div>""",
        unsafe_allow_html=True,
    )


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.image(LOGO_PATH, use_column_width=True)
    st.markdown("## 📂 Document Management")
    st.caption("Upload multiple documents to create your RAG knowledge base.")

    uploaded_files = st.file_uploader(
        "📁 Select Documents",
        type=["pdf", "docx", "txt", "csv", "pptx"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.markdown("### 📊 Knowledge Base")
    st.metric("📂 Uploaded Files", len(st.session_state.available_files))
    st.metric("📄 Pages / Documents", len(st.session_state.documents))
    st.metric("🧩 Generated Chunks", len(st.session_state.chunks))

    vector_count = 0
    if st.session_state.vector_db is not None:
        try:
            vector_count = st.session_state.vector_db.index.ntotal
        except Exception:
            vector_count = 0

    st.metric("🔍 FAISS Vectors", vector_count)

    st.divider()

    st.markdown("### 🧠 RAG Components")
    st.write("✅ Multi-Document Support")
    st.write("✅ Smart Chunking")
    st.write("✅ Multilingual Embeddings")
    st.write("✅ FAISS Vector Search")
    st.write("✅ MMR Retrieval")
    st.write("✅ File-Specific Retrieval")
    st.write("✅ Document Comparison")
    st.write("✅ Groq LLM")
    st.write("✅ Source/Page Citation")

    st.divider()

    st.markdown(
        """
        **LLM Model**
        `openai/gpt-oss-120b`

        **Embedding Model**
        `intfloat/multilingual-e5-base`

        **Chunk Size**
        `800`

        **Chunk Overlap**
        `150`
        """
    )

    st.divider()
    st.caption("Smart Multilingual AI RAG Assistant")
    st.caption("Version 7.0")


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(files):
    documents = []

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for uploaded_file in files:
        try:
            filename = uploaded_file.name
            save_path = os.path.join(UPLOAD_DIR, filename)

            with open(save_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            extension = Path(filename).suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(save_path)
            elif extension == ".docx":
                loader = UnstructuredWordDocumentLoader(save_path)
            elif extension == ".txt":
                loader = TextLoader(save_path, encoding="utf-8")
            elif extension == ".csv":
                loader = CSVLoader(save_path)
            elif extension == ".pptx":
                loader = UnstructuredPowerPointLoader(save_path)
            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = filename

            documents.extend(docs)

        except Exception as error:
            st.error(f"Error loading {uploaded_file.name}: {error}")

    return documents


# ==========================================================
# SMART CHUNKING
# ==========================================================

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_documents(documents)


# ==========================================================
# BUILD VECTOR DATABASE
# ==========================================================

def build_vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_db = FAISS.from_documents(chunks, embedding_model)
    return embedding_model, vector_db


# ==========================================================
# BUILD RAG
# ==========================================================

def build_rag(vector_db):
    if not GROQ_API_KEY:
        raise Exception(
            "GROQ_API_KEY is not configured. Please add GROQ_API_KEY to Streamlit Secrets."
        )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30,
            "lambda_mult": 0.70,
        },
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1400,
        groq_api_key=GROQ_API_KEY,
    )

    return retriever, llm


# ==========================================================
# FILE LIST
# ==========================================================

def prepare_available_files(documents):
    files = sorted(
        list(
            set(
                os.path.basename(str(doc.metadata.get("source", "")))
                for doc in documents
                if doc.metadata.get("source", "")
            )
        )
    )
    return files


# ==========================================================
# NORMALIZE FILE NAME
# ==========================================================

def normalize_file_name(filename):
    if not filename:
        return ""
    return os.path.basename(str(filename)).strip().lower()


# ==========================================================
# GET FILE CHUNKS
# ==========================================================

def get_file_chunks(file_name):
    target = normalize_file_name(file_name)
    return [
        doc
        for doc in st.session_state.chunks
        if normalize_file_name(doc.metadata.get("source", "")) == target
    ]


# ==========================================================
# DETECT FILE
# ==========================================================

def detect_file(question):
    if not question:
        return None

    question_lower = question.lower()

    for filename in st.session_state.available_files:
        if filename.lower() in question_lower:
            return filename

    for filename in st.session_state.available_files:
        base = os.path.splitext(filename)[0].lower()
        if len(base) >= 4 and base in question_lower:
            return filename

    return None


# ==========================================================
# COMPARISON QUESTION
# ==========================================================

def is_comparison_question(question):
    question_lower = question.lower()
    keywords = [
        "compare",
        "comparison",
        "compare documents",
        "compare files",
        "difference between",
        "differences between",
        "similarity between",
        "similarities between",
        "all uploaded documents",
        "all uploaded files",
    ]

    return any(keyword in question_lower for keyword in keywords)


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(docs):
    result = []
    seen = set()

    for doc in docs:
        source = normalize_file_name(doc.metadata.get("source", ""))
        page = str(doc.metadata.get("page", ""))
        content = doc.page_content or ""
        key = (source, page, content[:250])

        if key not in seen:
            seen.add(key)
            result.append(doc)

    return result


# ==========================================================
# RETRIEVE FROM FILE
# ==========================================================

def retrieve_from_file(question, file_name, max_results=10):
    file_chunks = get_file_chunks(file_name)

    if not file_chunks:
        return []

    temporary_db = FAISS.from_documents(
        file_chunks, st.session_state.embedding_model
    )

    temp_retriever = temporary_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": min(max_results, len(file_chunks)),
            "fetch_k": min(30, len(file_chunks)),
            "lambda_mult": 0.65,
        },
    )

    return temp_retriever.invoke(question)


# ==========================================================
# RETRIEVE FROM ALL FILES
# ==========================================================

def retrieve_from_all_files(question):
    results = []
    for filename in st.session_state.available_files:
        try:
            docs = retrieve_from_file(question, filename, max_results=8)
            results.extend(docs)
        except Exception as error:
            print(f"Retrieval error: {error}")

    return remove_duplicates(results)


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(docs):
    context_parts = []
    for index, doc in enumerate(docs, start=1):
        source = os.path.basename(str(doc.metadata.get("source", "Unknown")))
        page = doc.metadata.get("page")

        if page is not None:
            try:
                page = int(page) + 1
            except Exception:
                pass

        header = f"[Retrieved Chunk {index}] Source: {source}"
        if page is not None:
            header += f" | Page: {page}"

        content = doc.page_content or ""
        context_parts.append(header + "\n" + content)

    return "\n\n".join(context_parts)


# ==========================================================
# FILE SUMMARY
# ==========================================================

def build_file_summary():
    lines = []
    for filename in st.session_state.available_files:
        file_chunks = get_file_chunks(filename)
        lines.append(f"- {filename}: {len(file_chunks)} text chunk(s)")

    return "\n".join(lines)


# ==========================================================
# ASK RAG
# ==========================================================

def ask_rag(question):
    vector_db = st.session_state.vector_db
    retriever = st.session_state.retriever
    llm = st.session_state.llm

    if vector_db is None or retriever is None or llm is None:
        raise Exception("RAG system is not ready. Please process documents first.")

    question = question.strip()
    question_lower = question.lower()

    if any(
        phrase in question_lower
        for phrase in ["how many files", "number of files", "total files", "uploaded files"]
    ):
        return {
            "result": f"There are {len(st.session_state.available_files)} uploaded file(s).",
            "source_documents": [],
        }

    if any(
        phrase in question_lower
        for phrase in ["how many chunks", "number of chunks", "total chunks", "generated chunks"]
    ):
        return {
            "result": f"{len(st.session_state.chunks)} chunk(s) were generated.",
            "source_documents": [],
        }

    if any(
        phrase in question_lower
        for phrase in ["how many pages", "number of pages", "total pages", "page count"]
    ):
        return {
            "result": f"There are {len(st.session_state.documents)} processed page/document object(s).",
            "source_documents": [],
        }

    matched_file = detect_file(question)

    if matched_file:
        source_documents = retrieve_from_file(question, matched_file, max_results=10)
    elif is_comparison_question(question):
        source_documents = retrieve_from_all_files(question)
    else:
        source_documents = retriever.invoke(question)

    source_documents = remove_duplicates(source_documents)[:30]

    if not source_documents:
        return {
            "result": "I couldn't find the answer in the uploaded documents.",
            "source_documents": [],
        }

    context = build_context(source_documents)
    file_summary = build_file_summary()

    prompt = f"""
You are Smart Multilingual AI RAG Assistant.

Answer ONLY from the uploaded document context.

IMPORTANT RULES:

1. Never use outside knowledge.
2. Never guess.
3. Never hallucinate.
4. Never invent facts.
5. Preserve exact values from documents.
6. Pay attention to filenames and page numbers.
7. Keep information associated with the correct file.
8. Answer in the same language as the user's question.

If the requested information is genuinely absent, reply exactly:
I couldn't find the answer in the uploaded documents.

EXACT INFORMATION:
For email addresses, phone numbers, CNIC, names, dates, IDs, URLs, addresses, qualifications and other exact values:
- Copy the exact value from the document.
- Do not change spelling.
- Do not change digits.
- Do not invent values.

COMPARISON:
If the user asks to compare documents:
- Compare every uploaded file separately.
- Group pages belonging to the same filename.
- Explain major similarities.
- Explain major differences.
- Do not treat chunks as separate documents.
- Do not confuse processed pages with uploaded files.

UPLOADED FILE INFORMATION:
{file_summary}

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

ANSWER:
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "result": answer,
        "source_documents": source_documents,
    }


# ==========================================================
# PROCESS DOCUMENTS
# ==========================================================

def process_documents(files):
    if not files:
        st.warning("Please upload at least one document.")
        return

    start_time = time.time()

    try:
        with st.spinner("📚 Loading documents..."):
            documents = load_documents(files)

        if not documents:
            st.error("No supported documents were loaded.")
            return

        with st.spinner("🧩 Creating smart chunks..."):
            chunks = create_chunks(documents)

        if not chunks:
            st.error("No text chunks were created.")
            return

        with st.spinner("🔍 Creating multilingual embeddings and FAISS database..."):
            embedding_model, vector_db = build_vector_database(chunks)

        with st.spinner("🤖 Building RAG system..."):
            retriever, llm = build_rag(vector_db)

        available_files = prepare_available_files(documents)

        st.session_state.documents = documents
        st.session_state.chunks = chunks
        st.session_state.embedding_model = embedding_model
        st.session_state.vector_db = vector_db
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.available_files = available_files
        st.session_state.chat_history = []

        elapsed = round(time.time() - start_time, 2)
        st.session_state.processing_time = elapsed
        st.session_state.last_retrieved = 0
        st.session_state.last_response_time = 0.0

        st.success(f"✅ Knowledge Base Ready! Processed in {elapsed} seconds.")
        st.rerun()

    except Exception as error:
        st.error(f"❌ Processing Error: {error}")
        print(f"PROCESSING ERROR: {error}")


# ==========================================================
# PROCESS BUTTON
# ==========================================================

if process_button:
    process_documents(uploaded_files)


# ==========================================================
# INTRO CARD
# ==========================================================

st.markdown(
    """<div class="info-card">
        <div class="card-title">🧠 AI Research Assistant</div>
        <div class="card-text">
            Upload multiple documents, build a multilingual semantic knowledge base, and ask questions using Retrieval-Augmented Generation (RAG). Answers are grounded in your uploaded documents with source and page references.
        </div>
    </div>""",
    unsafe_allow_html=True,
)


# ==========================================================
# MAIN STATISTICS
# ==========================================================

col1, col2, col3, col4, col5 = st.columns(5)

vector_count = 0
if st.session_state.vector_db is not None:
    try:
        vector_count = st.session_state.vector_db.index.ntotal
    except Exception:
        vector_count = 0

stats = [
    (col1, "📂", len(st.session_state.available_files), "Files"),
    (col2, "📄", len(st.session_state.documents), "Pages / Docs"),
    (col3, "🧩", len(st.session_state.chunks), "Chunks"),
    (col4, "🔍", vector_count, "FAISS Vectors"),
    (col5, "⚡", f"{st.session_state.processing_time:.2f}s", "Processing Time"),
]

for column, icon, number, label in stats:
    with column:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ==========================================================
# KNOWLEDGE BASE
# ==========================================================

if st.session_state.available_files:
    st.divider()
    st.subheader("📚 Knowledge Base")

    kb1, kb2 = st.columns([2, 1])

    with kb1:
        for filename in st.session_state.available_files:
            file_chunks = get_file_chunks(filename)
            st.markdown(
                f"""<div class="source-box">
                    📄 <b>{filename}</b><br>
                    <span style="color:#64748b;">{len(file_chunks)} text chunks</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with kb2:
        st.markdown(
            f"""<div class="info-card">
                <div class="card-title">📊 Knowledge Base Summary</div>
                <div class="card-text">
                    📂 Files: <b>{len(st.session_state.available_files)}</b><br><br>
                    🧩 Chunks: <b>{len(st.session_state.chunks)}</b><br><br>
                    🔍 FAISS Vectors: <b>{vector_count}</b>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ==========================================================
# CHAT HISTORY
# ==========================================================

if st.session_state.chat_history:
    st.divider()
    st.subheader("💬 Conversation")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ==========================================================
# QUESTION INPUT
# ==========================================================

question = st.chat_input("💬 Ask a question about your uploaded documents...")


# ==========================================================
# QUESTION PROCESSING
# ==========================================================

if question:
    if st.session_state.vector_db is None:
        st.warning("⚠️ Please upload and process documents first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        start_time = time.time()

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching knowledge base..."):
                try:
                    result = ask_rag(question)
                    answer = result["result"]
                    source_documents = result["source_documents"]

                    st.session_state.last_retrieved = len(source_documents)
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state.last_response_time = elapsed

                    st.markdown(answer)

                    source_groups = {}
                    for doc in source_documents:
                        source = os.path.basename(
                            str(doc.metadata.get("source", "Unknown"))
                        )
                        page = doc.metadata.get("page")

                        if page is not None:
                            try:
                                page = int(page) + 1
                            except Exception:
                                pass

                        if source not in source_groups:
                            source_groups[source] = []

                        if page is not None and page not in source_groups[source]:
                            source_groups[source].append(page)

                    if source_groups:
                        st.markdown("### 📄 Sources")
                        for source, pages in source_groups.items():
                            pages.sort()
                            if pages:
                                st.markdown(
                                    f"**📄 {source}** — Page(s): {', '.join(map(str, pages))}"
                                )
                            else:
                                st.markdown(f"**📄 {source}**")

                    final_answer = answer
                    if source_groups:
                        final_answer += "\n\n### 📄 Sources\n"
                        for source, pages in source_groups.items():
                            pages.sort()
                            if pages:
                                final_answer += f"\n- **{source}** (Pages: {', '.join(map(str, pages))})"
                            else:
                                final_answer += f"\n- **{source}**"

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": final_answer}
                    )

                except Exception as error:
                    error_message = f"❌ Error: {error}"
                    st.error(error_message)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_message}
                    )


# ==========================================================
# RAG STATISTICS
# ==========================================================

st.divider()
st.subheader("📊 RAG Statistics")

s1, s2, s3, s4, s5 = st.columns(5)

questions_count = sum(
    1 for message in st.session_state.chat_history if message["role"] == "user"
)
answers_count = sum(
    1 for message in st.session_state.chat_history if message["role"] == "assistant"
)

with s1:
    st.metric("💬 Questions", questions_count)

with s2:
    st.metric("🤖 Answers", answers_count)

with s3:
    st.metric("🔎 Retrieved Chunks", st.session_state.last_retrieved)

with s4:
    st.metric("⚡ Response Time", f"{st.session_state.last_response_time:.2f}s")

with s5:
    st.metric("🧠 LLM", "GPT-OSS-120B")


# ==========================================================
# CHAT CONTROLS
# ==========================================================

if st.session_state.chat_history:
    st.markdown("")
    clear_col, info_col = st.columns([1, 4])

    with clear_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_retrieved = 0
            st.session_state.last_response_time = 0.0
            st.rerun()

    with info_col:
        st.caption(
            "Conversation history is maintained during the current Streamlit session."
        )


# ==========================================================
# SYSTEM CAPABILITIES
# ==========================================================

st.divider()
st.subheader("🚀 System Capabilities")

cap1, cap2, cap3 = st.columns(3)

with cap1:
    st.markdown(
        """<div class="capability-card">
            <div class="capability-title">📚 Multi-Document RAG</div>
            <div class="capability-text">
                • PDF support<br>
                • DOCX support<br>
                • TXT support<br>
                • CSV support<br>
                • PPTX support<br>
                • File-specific retrieval
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with cap2:
    st.markdown(
        """<div class="capability-card">
            <div class="capability-title">🧠 AI Retrieval</div>
            <div class="capability-text">
                • Multilingual embeddings<br>
                • FAISS vector database<br>
                • MMR retrieval<br>
                • Context-aware answers<br>
                • Document comparison<br>
                • Semantic search
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

with cap3:
    st.markdown(
        """<div class="capability-card">
            <div class="capability-title">⚡ AI Generation</div>
            <div class="capability-text">
                • Groq inference<br>
                • GPT-OSS-120B<br>
                • Grounded responses<br>
                • Source/page citation<br>
                • Multilingual answering<br>
                • Hallucination-aware prompting
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    f"""<div class="footer">
        <hr>
        <img src="{LOGO_PATH}" style="height: 40px; margin-bottom: 10px;" alt="University of Okara Logo"><br>
        <b>🤖 Smart Multilingual AI RAG Assistant</b><br>
        Department of Computer Science • University of Okara<br>
        MSCS Research Project • Version 7.0<br><br>
        Built with Streamlit • LangChain • FAISS • Hugging Face • Groq
    </div>""",
    unsafe_allow_html=True,
)
