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
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Multilingual AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f3f6fb;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .department {
        font-size: 21px;
        font-weight: 600;
        color: #e8f0ff;
    }

    .university {
        font-size: 17px;
        color: #d8e4ff;
    }

    .header-card {
        background: linear-gradient(
            90deg,
            #0f172a,
            #2563eb
        );
        padding: 25px;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .stat-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 10px;
    }

    .stat-number {
        font-size: 27px;
        font-weight: 800;
        color: #2563eb;
    }

    .stat-label {
        font-size: 14px;
        color: #475569;
    }

    .source-box {
        background: #f8fafc;
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #2563eb;
        margin-top: 10px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: #ecfdf5;
        border: 1px solid #10b981;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "available_files" not in st.session_state:
    st.session_state.available_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0.0

if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = 0

if "last_response_time" not in st.session_state:
    st.session_state.last_response_time = 0.0


# ==========================================================
# CONFIGURATION
# ==========================================================

UPLOAD_DIR = "uploaded_documents"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ==========================================================
# GROQ API KEY
# ==========================================================

try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="header-card">

        <div class="main-title">
            🤖 Smart Multilingual AI RAG Assistant
        </div>

        <div class="department">
            Department of Computer Science
        </div>

        <div class="university">
            University of Okara • MSCS Research Project
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Select Documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "pptx",
        ],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    st.subheader("📊 Knowledge Base")

    st.metric(
        "Uploaded Files",
        len(
            st.session_state.available_files
        )
    )

    st.metric(
        "Processed Pages",
        len(
            st.session_state.documents
        )
    )

    st.metric(
        "Generated Chunks",
        len(
            st.session_state.chunks
        )
    )

    vector_count = 0

    if st.session_state.vector_db is not None:

        try:
            vector_count = (
                st.session_state.vector_db
                .index
                .ntotal
            )
        except Exception:
            vector_count = 0

    st.metric(
        "FAISS Vectors",
        vector_count
    )

    st.divider()

    st.subheader("ℹ️ System Information")

    st.write("✅ Multiple Document Support")
    st.write("✅ Semantic Search")
    st.write("✅ FAISS Vector Database")
    st.write("✅ Multilingual Embeddings")
    st.write("✅ Context-Aware Retrieval")
    st.write("✅ Groq LLM")
    st.write("✅ Source Citation")

    st.divider()

    st.caption(
        "Smart Multilingual AI RAG Assistant"
    )

    st.caption(
        "Version 6.02"
    )


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(files):

    documents = []

    if os.path.exists(UPLOAD_DIR):

        shutil.rmtree(
            UPLOAD_DIR
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    for uploaded_file in files:

        try:

            filename = uploaded_file.name

            save_path = os.path.join(
                UPLOAD_DIR,
                filename
            )

            with open(
                save_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            extension = (
                Path(filename)
                .suffix
                .lower()
            )

            if extension == ".pdf":

                loader = PyPDFLoader(
                    save_path
                )

            elif extension == ".docx":

                loader = (
                    UnstructuredWordDocumentLoader(
                        save_path
                    )
                )

            elif extension == ".txt":

                loader = TextLoader(
                    save_path,
                    encoding="utf-8"
                )

            elif extension == ".csv":

                loader = CSVLoader(
                    save_path
                )

            elif extension == ".pptx":

                loader = (
                    UnstructuredPowerPointLoader(
                        save_path
                    )
                )

            else:

                continue

            docs = loader.load()

            for doc in docs:

                doc.metadata["source"] = (
                    filename
                )

            documents.extend(
                docs
            )

        except Exception as e:

            st.error(
                f"Error loading {uploaded_file.name}: {e}"
            )

    return documents


# ==========================================================
# CHUNKING
# ==========================================================

def create_chunks(documents):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            keep_separator=True,
        )
    )

    return splitter.split_documents(
        documents
    )


# ==========================================================
# BUILD VECTOR DATABASE
# ==========================================================

def build_vector_database(chunks):

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=(
                "intfloat/"
                "multilingual-e5-base"
            ),
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )
    )

    vector_db = (
        FAISS.from_documents(
            chunks,
            embedding_model
        )
    )

    return (
        embedding_model,
        vector_db
    )


# ==========================================================
# BUILD RAG
# ==========================================================

def build_rag(vector_db):

    if not GROQ_API_KEY:

        raise Exception(
            "GROQ_API_KEY is not configured. "
            "Please add it in Streamlit Secrets."
        )

    retriever = (
        vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 12,
                "fetch_k": 30,
                "lambda_mult": 0.70,
            },
        )
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1400,
        groq_api_key=GROQ_API_KEY,
    )

    return (
        retriever,
        llm
    )


# ==========================================================
# PREPARE FILE LIST
# ==========================================================

def prepare_available_files(
    documents
):

    files = sorted(
        list(
            set(
                os.path.basename(
                    str(
                        doc.metadata.get(
                            "source",
                            ""
                        )
                    )
                )
                for doc in documents
                if doc.metadata.get(
                    "source",
                    ""
                )
            )
        )
    )

    return files


# ==========================================================
# NORMALIZE FILE NAME
# ==========================================================

def normalize_file_name(
    filename
):

    if not filename:

        return ""

    return (
        os.path.basename(
            str(filename)
        )
        .strip()
        .lower()
    )


# ==========================================================
# GET FILE CHUNKS
# ==========================================================

def get_file_chunks(
    file_name
):

    target = (
        normalize_file_name(
            file_name
        )
    )

    return [
        doc
        for doc in st.session_state.chunks
        if normalize_file_name(
            doc.metadata.get(
                "source",
                ""
            )
        ) == target
    ]


# ==========================================================
# DETECT FILE
# ==========================================================

def detect_file(
    question
):

    if not question:

        return None

    q = question.lower()

    for filename in (
        st.session_state.available_files
    ):

        if filename.lower() in q:

            return filename

    for filename in (
        st.session_state.available_files
    ):

        base = (
            os.path.splitext(
                filename
            )[0]
            .lower()
        )

        if (
            len(base) >= 4
            and base in q
        ):

            return filename

    return None


# ==========================================================
# COMPARISON QUESTION
# ==========================================================

def is_comparison_question(
    question
):

    q = question.lower()

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

    return any(
        keyword in q
        for keyword in keywords
    )


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(
    docs
):

    result = []

    seen = set()

    for doc in docs:

        source = normalize_file_name(
            doc.metadata.get(
                "source",
                ""
            )
        )

        page = str(
            doc.metadata.get(
                "page",
                ""
            )
        )

        content = (
            doc.page_content
            or ""
        )

        key = (
            source,
            page,
            content[:250]
        )

        if key not in seen:

            seen.add(key)

            result.append(
                doc
            )

    return result


# ==========================================================
# RETRIEVE FROM SPECIFIC FILE
# ==========================================================

def retrieve_from_file(
    question,
    file_name,
    max_results=10
):

    file_chunks = (
        get_file_chunks(
            file_name
        )
    )

    if not file_chunks:

        return []

    temporary_db = (
        FAISS.from_documents(
            file_chunks,
            st.session_state.embedding_model,
        )
    )

    temp_retriever = (
        temporary_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": min(
                    max_results,
                    len(file_chunks)
                ),
                "fetch_k": min(
                    30,
                    len(file_chunks)
                ),
                "lambda_mult": 0.65,
            },
        )
    )

    return temp_retriever.invoke(
        question
    )


# ==========================================================
# RETRIEVE FROM ALL FILES
# ==========================================================

def retrieve_from_all_files(
    question
):

    results = []

    for filename in (
        st.session_state.available_files
    ):

        try:

            docs = retrieve_from_file(
                question,
                filename,
                max_results=8,
            )

            results.extend(
                docs
            )

        except Exception as e:

            print(
                f"Retrieval error: {e}"
            )

    return remove_duplicates(
        results
    )


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(
    docs
):

    context_parts = []

    for index, doc in enumerate(
        docs,
        start=1
    ):

        source = os.path.basename(
            str(
                doc.metadata.get(
                    "source",
                    "Unknown"
                )
            )
        )

        page = doc.metadata.get(
            "page"
        )

        if page is not None:

            try:

                page = (
                    int(page) + 1
                )

            except Exception:

                pass

        header = (
            f"[Retrieved Chunk {index}] "
            f"Source: {source}"
        )

        if page is not None:

            header += (
                f" | Page: {page}"
            )

        content = (
            doc.page_content
            or ""
        )

        context_parts.append(
            header
            + "\n"
            + content
        )

    return "\n\n".join(
        context_parts
    )


# ==========================================================
# FILE SUMMARY
# ==========================================================

def build_file_summary():

    lines = []

    for filename in (
        st.session_state.available_files
    ):

        file_chunks = (
            get_file_chunks(
                filename
            )
        )

        lines.append(
            f"- {filename}: "
            f"{len(file_chunks)} text chunk(s)"
        )

    return "\n".join(
        lines
    )


# ==========================================================
# ASK RAG
# ==========================================================

def ask_rag(
    question
):

    vector_db = (
        st.session_state.vector_db
    )

    retriever = (
        st.session_state.retriever
    )

    llm = (
        st.session_state.llm
    )

    if (
        vector_db is None
        or retriever is None
        or llm is None
    ):

        raise Exception(
            "RAG system is not ready. "
            "Please process documents first."
        )

    question = question.strip()

    q = question.lower()

    # ------------------------------------------------------
    # FILE COUNT
    # ------------------------------------------------------

    if any(
        x in q
        for x in [
            "how many files",
            "number of files",
            "total files",
            "uploaded files",
        ]
    ):

        return {
            "result":
                f"There are "
                f"{len(st.session_state.available_files)} "
                f"uploaded file(s).",
            "source_documents": [],
        }

    # ------------------------------------------------------
    # CHUNK COUNT
    # ------------------------------------------------------

    if any(
        x in q
        for x in [
            "how many chunks",
            "number of chunks",
            "total chunks",
            "generated chunks",
        ]
    ):

        return {
            "result":
                f"{len(st.session_state.chunks)} "
                f"chunk(s) were generated.",
            "source_documents": [],
        }

    # ------------------------------------------------------
    # PAGE COUNT
    # ------------------------------------------------------

    if any(
        x in q
        for x in [
            "how many pages",
            "number of pages",
            "total pages",
            "page count",
        ]
    ):

        return {
            "result":
                f"There are "
                f"{len(st.session_state.documents)} "
                f"processed page/document object(s).",
            "source_documents": [],
        }

    # ------------------------------------------------------
    # FILE DETECTION
    # ------------------------------------------------------

    matched_file = detect_file(
        question
    )

    # ------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------

    if matched_file:

        source_documents = (
            retrieve_from_file(
                question,
                matched_file,
                max_results=10,
            )
        )

    elif is_comparison_question(
        question
    ):

        source_documents = (
            retrieve_from_all_files(
                question
            )
        )

    else:

        source_documents = (
            retriever.invoke(
                question
            )
        )

    source_documents = (
        remove_duplicates(
            source_documents
        )
    )

    source_documents = (
        source_documents[:30]
    )

    if not source_documents:

        return {
            "result":
                "I couldn't find the answer "
                "in the uploaded documents.",
            "source_documents": [],
        }

    context = build_context(
        source_documents
    )

    file_summary = (
        build_file_summary()
    )

    # ------------------------------------------------------
    # RAG PROMPT
    # ------------------------------------------------------

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

If the requested information is genuinely absent,
reply exactly:

I couldn't find the answer in the uploaded documents.

EXACT INFORMATION:

For email addresses, phone numbers, CNIC,
names, dates, IDs, URLs, addresses,
qualifications and other exact values:

- Copy the exact value from the document.
- Do not change spelling.
- Do not change digits.
- Do not invent values.

COMPARISON:

If the user asks to compare documents:

- Compare every uploaded file separately.
- Group pages belonging to the same filename.
- Explain the purpose of each file.
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

    response = llm.invoke(
        prompt
    )

    answer = (
        response.content
        if hasattr(
            response,
            "content"
        )
        else str(response)
    )

    return {
        "result": answer,
        "source_documents":
            source_documents,
    }


# ==========================================================
# PROCESS DOCUMENTS
# ==========================================================

def process_documents():

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

        return

    start = time.time()

    try:

        with st.spinner(
            "📚 Loading documents..."
        ):

            documents = load_documents(
                uploaded_files
            )

        if not documents:

            st.error(
                "No supported documents were loaded."
            )

            return

        st.info(
            f"📄 Loaded {len(documents)} "
            f"document/page objects."
        )

        with st.spinner(
            "🧩 Creating smart chunks..."
        ):

            chunks = create_chunks(
                documents
            )

        st.info(
            f"🧩 Created {len(chunks)} chunks."
        )

        with st.spinner(
            "🔍 Creating multilingual embeddings and FAISS database..."
        ):

            (
                embedding_model,
                vector_db
            ) = build_vector_database(
                chunks
            )

        with st.spinner(
            "🤖 Building RAG system..."
        ):

            (
                retriever,
                llm
            ) = build_rag(
                vector_db
            )

        available_files = (
            prepare_available_files(
                documents
            )
        )

        # --------------------------------------------------
        # SAVE TO SESSION
        # --------------------------------------------------

        st.session_state.documents = (
            documents
        )

        st.session_state.chunks = (
            chunks
        )

        st.session_state.embedding_model = (
            embedding_model
        )

        st.session_state.vector_db = (
            vector_db
        )

        st.session_state.retriever = (
            retriever
        )

        st.session_state.llm = (
            llm
        )

        st.session_state.available_files = (
            available_files
        )

        st.session_state.chat_history = []

        elapsed = round(
            time.time() - start,
            2
        )

        st.session_state.processing_time = (
            elapsed
        )

        st.session_state.last_retrieved = 0

        st.session_state.last_response_time = 0

        st.success(
            "✅ Knowledge Base Ready!"
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"❌ Processing Error: {str(e)}"
        )

        print(
            f"PROCESSING ERROR: {e}"
        )


# ==========================================================
# PROCESS BUTTON
# ==========================================================

if process_button:

    process_documents()


# ==========================================================
# MAIN DASHBOARD
# ==========================================================

st.title(
    "🤖 AI Research Assistant"
)

st.write(
    "Ask questions about your uploaded documents "
    "and receive context-aware answers with sources."
)


# ==========================================================
# STATISTICS
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📂 Files",
        len(
            st.session_state.available_files
        )
    )

with col2:

    st.metric(
        "📄 Pages",
        len(
            st.session_state.documents
        )
    )

with col3:

    st.metric(
        "🧩 Chunks",
        len(
            st.session_state.chunks
        )
    )

with col4:

    vector_count = 0

    if (
        st.session_state.vector_db
        is not None
    ):

        try:

            vector_count = (
                st.session_state.vector_db
                .index
                .ntotal
            )

        except Exception:

            vector_count = 0

    st.metric(
        "🔍 Vectors",
        vector_count
    )


st.divider()


# ==========================================================
# KNOWLEDGE BASE FILES
# ==========================================================

if st.session_state.available_files:

    with st.expander(
        "📚 Uploaded Documents"
    ):

        for filename in (
            st.session_state.available_files
        ):

            file_chunks = (
                get_file_chunks(
                    filename
                )
            )

            st.write(
                f"📄 **{filename}** — "
                f"{len(file_chunks)} chunks"
            )


# ==========================================================
# CHAT HISTORY
# ==========================================================

if st.session_state.chat_history:

    st.subheader(
        "💬 Conversation"
    )

    for message in (
        st.session_state.chat_history
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


# ==========================================================
# QUESTION INPUT
# ==========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# ==========================================================
# QUESTION PROCESSING
# ==========================================================

if question:

    if (
        st.session_state.vector_db
        is None
    ):

        st.warning(
            "⚠️ Please upload and process "
            "documents first."
        )

    else:

        # --------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        start = time.time()

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🤖 Searching documents and generating answer..."
            ):

                try:

                    result = ask_rag(
                        question
                    )

                    answer = result[
                        "result"
                    ]

                    source_documents = (
                        result[
                            "source_documents"
                        ]
                    )

                    st.session_state.last_retrieved = (
                        len(
                            source_documents
                        )
                    )

                    elapsed = round(
                        time.time()
                        - start,
                        2
                    )

                    st.session_state.last_response_time = (
                        elapsed
                    )

                    # --------------------------------------
                    # DISPLAY ANSWER
                    # --------------------------------------

                    st.markdown(
                        answer
                    )

                    # --------------------------------------
                    # SOURCES
                    # --------------------------------------

                    source_groups = {}

                    for doc in (
                        source_documents
                    ):

                        source = os.path.basename(
                            str(
                                doc.metadata.get(
                                    "source",
                                    "Unknown"
                                )
                            )
                        )

                        page = doc.metadata.get(
                            "page"
                        )

                        if page is not None:

                            try:

                                page = (
                                    int(page) + 1
                                )

                            except Exception:

                                pass

                        if source not in (
                            source_groups
                        ):

                            source_groups[
                                source
                            ] = []

                        if (
                            page is not None
                            and page not in (
                                source_groups[
                                    source
                                ]
                            )
                        ):

                            source_groups[
                                source
                            ].append(
                                page
                            )

                    if source_groups:

                        st.markdown(
                            "### 📄 Sources"
                        )

                        for (
                            source,
                            pages
                        ) in source_groups.items():

                            pages.sort()

                            if pages:

                                page_text = (
                                    ", ".join(
                                        str(p)
                                        for p in pages
                                    )
                                )

                                st.markdown(
                                    f"**📄 {source}** "
                                    f"— Page(s): "
                                    f"{page_text}"
                                )

                            else:

                                st.markdown(
                                    f"**📄 {source}**"
                                )

                    # --------------------------------------
                    # SAVE ASSISTANT MESSAGE
                    # --------------------------------------

                    final_answer = answer

                    if source_groups:

                        final_answer += (
                            "\n\n### 📄 Sources\n"
                        )

                        for (
                            source,
                            pages
                        ) in source_groups.items():

                            pages.sort()

                            if pages:

                                final_answer += (
                                    f"\n- **{source}** "
                                    f"(Pages: "
                                    f"{', '.join(map(str, pages))})"
                                )

                            else:

                                final_answer += (
                                    f"\n- **{source}**"
                                )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content":
                                final_answer,
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"❌ Error: {str(e)}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content":
                                error_message,
                        }
                    )


# ==========================================================
# RESPONSE STATISTICS
# ==========================================================

st.divider()

st.subheader(
    "📊 RAG Statistics"
)

s1, s2, s3, s4, s5 = st.columns(5)

with s1:

    st.metric(
        "Questions",
        sum(
            1
            for message
            in st.session_state.chat_history
            if message["role"] == "user"
        )
    )

with s2:

    st.metric(
        "Answers",
        sum(
            1
            for message
            in st.session_state.chat_history
            if message["role"] == "assistant"
        )
    )

with s3:

    st.metric(
        "Retrieved Chunks",
        st.session_state.last_retrieved
    )

with s4:

    st.metric(
        "Response Time",
        f"{st.session_state.last_response_time:.2f}s"
    )

with s5:

    st.metric(
        "Model",
        "GPT-OSS-120B"
    )


# ==========================================================
# CLEAR CHAT
# ==========================================================

if st.session_state.chat_history:

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.chat_history = []

        st.session_state.last_retrieved = 0

        st.session_state.last_response_time = 0

        st.rerun()


# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
    """
    <br>
    <hr>

    <div style="text-align:center;color:#64748b;">

    <b>Smart Multilingual AI RAG Assistant</b><br>

    Department of Computer Science • University of Okara<br>

    MSCS Research Project • Version 6.02

    </div>
    """,
    unsafe_allow_html=True,
)
