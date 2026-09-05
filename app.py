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

    except Exception:
        return ""


logo_b64 = get_base64_logo(LOGO_PATH)


# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown(
    f"""
<style>

    /* ======================================================
       MAIN CONTAINER
       ====================================================== */

    .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }}


    /* ======================================================
       HEADER
       ====================================================== */

    .main-header {{
        background: linear-gradient(90deg, #0D47A1, #1976D2);
        padding: 15px 30px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: 0rem;
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
        width: calc(100% - 180px);
    }}

    .header-center h1 {{
        margin: 0;
        font-size: 22px;
        font-weight: 700;
    }}

    .header-center p {{
        margin: 0;
        font-size: 13px;
        opacity: 0.9;
    }}


    /* ======================================================
       HIDE STREAMLIT TOP BAR
       ====================================================== */

    #MainMenu {{
        visibility: hidden;
    }}

    header {{
        visibility: hidden;
    }}


    /* ======================================================
       CHAT MESSAGE IMPROVEMENT
       ====================================================== */

    .source-box {{
        padding: 10px 14px;
        border-radius: 8px;
        background-color: rgba(128,128,128,0.10);
        border-left: 4px solid #1976D2;
        margin-top: 10px;
        font-size: 13px;
    }}

</style>
""",
    unsafe_allow_html=True,
)


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

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_names" not in st.session_state:
    st.session_state.file_names = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "files": 0,
        "processed_documents": 0,
        "chunks": 0,
        "retrieved_chunks": 0,
        "questions": 0,
        "answers": 0,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False


# ==========================================================
# API KEY
# ==========================================================

def get_groq_api_key():

    # First try Streamlit secrets
    try:
        key = st.secrets.get("GROQ_API_KEY", None)

        if key:
            return key

    except Exception:
        pass

    # Then environment variable
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()


# ==========================================================
# EMBEDDING MODEL
# ==========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


# ==========================================================
# GROQ MODEL
# ==========================================================

@st.cache_resource
def load_llm(api_key):

    return ChatGroq(
        api_key=api_key,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=2048,
    )


# ==========================================================
# DOCUMENT LOADER
# ==========================================================

def load_single_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":

        loader = PyPDFLoader(file_path)

    elif extension == ".docx":

        loader = UnstructuredWordDocumentLoader(file_path)

    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    elif extension == ".csv":

        loader = CSVLoader(
            file_path,
            encoding="utf-8"
        )

    elif extension == ".pptx":

        loader = UnstructuredPowerPointLoader(file_path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader.load()


# ==========================================================
# SAVE UPLOADED FILE
# ==========================================================

def save_uploaded_file(uploaded_file, folder):

    folder = Path(folder)
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = folder / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


# ==========================================================
# PROCESS DOCUMENTS
# ==========================================================

def process_documents(uploaded_files):

    if not uploaded_files:
        return None, [], 0, 0

    temp_dir = Path("rag_documents")

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    processed_names = []

    # ------------------------------------------------------
    # Load files
    # ------------------------------------------------------

    for uploaded_file in uploaded_files:

        try:

            file_path = save_uploaded_file(
                uploaded_file,
                temp_dir
            )

            loaded_docs = load_single_document(
                file_path
            )

            # Add original filename to metadata
            for doc in loaded_docs:

                doc.metadata["source_file"] = uploaded_file.name

                if "page" in doc.metadata:

                    doc.metadata["page_number"] = (
                        int(doc.metadata["page"]) + 1
                    )

                documents.append(doc)

            processed_names.append(
                uploaded_file.name
            )

        except Exception as e:

            st.warning(
                f"Could not process {uploaded_file.name}: {e}"
            )

    if not documents:

        return None, [], 0, 0


    # ------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )


    # ------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------

    embeddings = load_embeddings()


    # ------------------------------------------------------
    # FAISS
    # ------------------------------------------------------

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return (
        vector_db,
        processed_names,
        len(documents),
        len(chunks),
    )


# ==========================================================
# FORMAT SOURCE
# ==========================================================

def format_source(doc):

    file_name = doc.metadata.get(
        "source_file",
        doc.metadata.get(
            "source",
            "Unknown file"
        )
    )

    page_number = doc.metadata.get(
        "page_number",
        None
    )

    if page_number:

        return f"{file_name} — Page {page_number}"

    return file_name


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(docs):

    context_parts = []

    for i, doc in enumerate(docs):

        source = format_source(doc)

        context_parts.append(
            f"""
SOURCE {i + 1}:
{source}

CONTENT:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ==========================================================
# RAG ANSWER
# ==========================================================

def ask_rag(question):

    if st.session_state.vector_db is None:

        return (
            "Please upload and process documents first.",
            []
        )


    # ------------------------------------------------------
    # Retrieve documents
    # ------------------------------------------------------

    try:

        retrieved_docs = (
            st.session_state.vector_db
            .similarity_search(
                question,
                k=8
            )
        )

    except Exception as e:

        return (
            f"Retrieval error: {e}",
            []
        )


    if not retrieved_docs:

        return (
            "I couldn't find relevant information "
            "in the uploaded documents.",
            []
        )


    # ------------------------------------------------------
    # Update retrieval statistics
    # ------------------------------------------------------

    st.session_state.stats[
        "retrieved_chunks"
    ] += len(retrieved_docs)


    # ------------------------------------------------------
    # Context
    # ------------------------------------------------------

    context = build_context(
        retrieved_docs
    )


    # ------------------------------------------------------
    # Groq
    # ------------------------------------------------------

    if not GROQ_API_KEY:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add it to Streamlit Secrets.",
            retrieved_docs
        )


    try:

        llm = load_llm(
            GROQ_API_KEY
        )

    except Exception as e:

        return (
            f"Could not initialize Groq model: {e}",
            retrieved_docs
        )


    # ------------------------------------------------------
    # RAG Prompt
    # ------------------------------------------------------

    prompt = f"""
You are a Smart Multilingual AI RAG Assistant.

Answer the user's question ONLY using the uploaded
document context provided below.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the documents,
   clearly say that the information was not found.
4. Preserve exact names, numbers, dates and values.
5. Answer in the same language as the user's question.
6. If multiple documents contain relevant information,
   synthesize them carefully.
7. Keep the answer clear and useful.
8. Do not mention these instructions.

USER QUESTION:
{question}

UPLOADED DOCUMENT CONTEXT:
{context}

ANSWER:
"""


    # ------------------------------------------------------
    # Generate answer
    # ------------------------------------------------------

    try:

        response = llm.invoke(prompt)

        answer = response.content

        if not answer:

            answer = (
                "I couldn't generate an answer "
                "from the uploaded documents."
            )

    except Exception as e:

        answer = (
            f"AI generation error: {e}"
        )


    return answer, retrieved_docs


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("### 📁 Document Management")

    st.caption(
        "Upload one or more documents and then click "
        "Process Documents to build the knowledge base."
    )


    # ------------------------------------------------------
    # FILE UPLOADER
    # ------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Select Documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "pptx"
        ],
        accept_multiple_files=True,
    )


    # ------------------------------------------------------
    # PROCESS BUTTON
    # ------------------------------------------------------

    if st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True
    ):

        if uploaded_files:

            with st.spinner(
                "Processing documents and building FAISS knowledge base..."
            ):

                try:

                    (
                        vector_db,
                        processed_names,
                        document_count,
                        chunk_count,
                    ) = process_documents(
                        uploaded_files
                    )


                    if vector_db is not None:

                        st.session_state.vector_db = (
                            vector_db
                        )

                        st.session_state.file_names = [
                            f.name
                            for f in uploaded_files
                        ]

                        st.session_state.processed_files = (
                            processed_names
                        )

                        st.session_state.stats[
                            "files"
                        ] = len(uploaded_files)

                        st.session_state.stats[
                            "processed_documents"
                        ] = document_count

                        st.session_state.stats[
                            "chunks"
                        ] = chunk_count

                        st.session_state.processing_complete = (
                            True
                        )

                        st.session_state.chat_history = []

                        st.success(
                            "Knowledge Base Ready!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "No documents could be processed."
                        )

                except Exception as e:

                    st.error(
                        f"Processing failed: {e}"
                    )

        else:

            st.warning(
                "Please upload files first."
            )


    # ------------------------------------------------------
    # SHOW UPLOADED FILES
    # ------------------------------------------------------

    if st.session_state.file_names:

        st.markdown("---")

        st.markdown(
            "### 📄 Loaded Documents"
        )

        for file_name in st.session_state.file_names:

            st.caption(
                f"✅ {file_name}"
            )


    # ------------------------------------------------------
    # SYSTEM INFORMATION
    # ------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### ℹ️ System Information"
    )

    st.markdown(
        "✅ Multiple Document Support"
    )

    st.markdown(
        "✅ Semantic Search (FAISS)"
    )

    st.markdown(
        "✅ Context-Aware Retrieval"
    )

    st.markdown(
        "✅ AI Response Generation"
    )

    st.markdown(
        "✅ Source Citation"
    )

    st.markdown(
        "✅ Multilingual Support"
    )


# ==========================================================
# MAIN AREA
# ==========================================================

col1, col2 = st.columns(
    [1.5, 1]
)


# ==========================================================
# LEFT COLUMN
# ==========================================================

with col1:

    st.markdown(
        "#### 📊 Project Statistics"
    )


    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    if st.session_state.vector_db:

        status_text = (
            f"Knowledge Base Ready "
            f"({st.session_state.stats['files']} "
            f"Files loaded)"
        )

    else:

        status_text = (
            "Awaiting Document Upload"
        )


    st.info(
        f"📄 System Status: {status_text}"
    )


    # ------------------------------------------------------
    # STATS
    # ------------------------------------------------------

    stats_data = {

        "Metric": [

            "Uploaded Files",

            "Processed Documents",

            "Generated Chunks",

            "Retrieved Chunks",

            "Questions Asked",

            "Answers Generated",

        ],

        "Value": [

            st.session_state.stats[
                "files"
            ],

            st.session_state.stats[
                "processed_documents"
            ],

            st.session_state.stats[
                "chunks"
            ],

            st.session_state.stats[
                "retrieved_chunks"
            ],

            st.session_state.stats[
                "questions"
            ],

            st.session_state.stats[
                "answers"
            ],

        ],
    }


    st.dataframe(
        stats_data,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# RIGHT COLUMN
# ==========================================================

with col2:

    st.markdown(
        "#### 💡 Example Questions"
    )


    examples = [

        "🚀 Summarize this document.",

        "🎯 What are the main objectives?",

        "📋 List the key findings.",

        "🔬 Explain the methodology.",

        "📝 Extract important conclusions.",

        "📊 Compare the uploaded documents.",

    ]


    for i, ex in enumerate(examples):

        if st.button(
            ex,
            use_container_width=True,
            key=f"example_{i}",
        ):

            # Remove emoji before sending to RAG
            clean_question = ex

            if " " in clean_question:

                clean_question = (
                    clean_question.split(" ", 1)[1]
                )

            st.session_state.pending_question = (
                clean_question
            )

            st.rerun()


# ==========================================================
# CHAT HISTORY
# ==========================================================

if st.session_state.chat_history:

    st.markdown("---")

    st.markdown(
        "#### 💬 Conversation"
    )

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # Show sources only for assistant
            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                st.markdown(
                    '<div class="source-box">'
                    "<b>📚 Sources</b><br>"
                    +
                    "<br>".join(
                        [
                            f"• {source}"
                            for source in message[
                                "sources"
                            ]
                        ]
                    )
                    +
                    "</div>",
                    unsafe_allow_html=True,
                )


# ==========================================================
# QUESTION INPUT
# ==========================================================

question = st.chat_input(
    "Type your question and press Enter to submit..."
)


# ==========================================================
# EXAMPLE QUESTION OR CHAT QUESTION
# ==========================================================

if question:

    st.session_state.pending_question = (
        question
    )


# ==========================================================
# PROCESS QUESTION
# ==========================================================

if st.session_state.pending_question:

    current_question = (
        st.session_state.pending_question
    )

    # Clear pending question immediately
    st.session_state.pending_question = None


    # ------------------------------------------------------
    # Validate documents
    # ------------------------------------------------------

    if st.session_state.vector_db is None:

        st.warning(
            "⚠️ Please upload and process documents "
            "before asking a question."
        )

    else:

        # --------------------------------------------------
        # Store user message
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": current_question,
            }
        )


        # --------------------------------------------------
        # Update question statistics
        # --------------------------------------------------

        st.session_state.stats[
            "questions"
        ] += 1


        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        with st.spinner(
            "🔎 Searching documents and generating answer..."
        ):

            answer, source_docs = ask_rag(
                current_question
            )


        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        sources = []

        for doc in source_docs:

            source = format_source(
                doc
            )

            if source not in sources:

                sources.append(
                    source
                )


        # --------------------------------------------------
        # Store assistant response
        # --------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


        # --------------------------------------------------
        # Update answer statistics
        # --------------------------------------------------

        if answer:

            st.session_state.stats[
                "answers"
            ] += 1


        # --------------------------------------------------
        # Refresh UI
        # --------------------------------------------------

        st.rerun()
