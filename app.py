import os
import shutil
import time
from pathlib import Path

import gradio as gr

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
# CONFIGURATION
# ==========================================================

UPLOAD_DIR = "uploaded_documents"

os.makedirs(UPLOAD_DIR, exist_ok=True)

documents = []
chunks = []

embedding_model = None
vector_db = None
retriever = None
llm = None

available_files = []


# ==========================================================
# GROQ API KEY
# ==========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not configured.")


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(files):

    global documents

    documents = []

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not files:
        return []

    for file in files:

        try:

            if isinstance(file, str):
                file_path = file
            else:
                file_path = getattr(file, "name", str(file))

            filename = os.path.basename(file_path)

            save_path = os.path.join(
                UPLOAD_DIR,
                filename
            )

            shutil.copy(file_path, save_path)

            ext = Path(filename).suffix.lower()

            if ext == ".pdf":

                loader = PyPDFLoader(save_path)

            elif ext == ".docx":

                loader = UnstructuredWordDocumentLoader(
                    save_path
                )

            elif ext == ".txt":

                loader = TextLoader(
                    save_path,
                    encoding="utf-8"
                )

            elif ext == ".csv":

                loader = CSVLoader(save_path)

            elif ext == ".pptx":

                loader = UnstructuredPowerPointLoader(
                    save_path
                )

            else:

                print(
                    f"Unsupported file type: {filename}"
                )
                continue

            docs = loader.load()

            # Make source filename consistent
            for doc in docs:
                doc.metadata["source"] = filename

            documents.extend(docs)

            print(
                f"Loaded: {filename} "
                f"({len(docs)} document/page objects)"
            )

        except Exception as e:

            print(
                f"Error loading file: {file_path}"
            )
            print(str(e))

    return documents


# ==========================================================
# SMART CHUNKING
# ==========================================================

def create_chunks():

    global chunks

    if not documents:
        raise Exception(
            "No documents loaded."
        )

    splitter = RecursiveCharacterTextSplitter(

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
            ""
        ],

        keep_separator=True
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    return chunks


# ==========================================================
# BUILD FAISS DATABASE
# ==========================================================

def build_vector_database():

    global embedding_model
    global vector_db

    if not chunks:
        raise Exception(
            "No chunks found."
        )

    print(
        "Loading multilingual embedding model..."
    )

    embedding_model = HuggingFaceEmbeddings(

        model_name="intfloat/multilingual-e5-base",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(
        "Embedding model loaded."
    )

    vector_db = FAISS.from_documents(

        documents=chunks,

        embedding=embedding_model
    )

    print(
        f"FAISS vectors: {vector_db.index.ntotal}"
    )

    return vector_db


# ==========================================================
# BUILD RETRIEVER + LLM
# ==========================================================

def build_rag():

    global retriever
    global llm

    if vector_db is None:
        raise Exception(
            "Vector database is not ready."
        )

    if not GROQ_API_KEY:
        raise Exception(
            "GROQ_API_KEY is not configured."
        )

    retriever = vector_db.as_retriever(

        search_type="mmr",

        search_kwargs={
            "k": 12,
            "fetch_k": 30,
            "lambda_mult": 0.70
        }
    )

    llm = ChatGroq(

        model="llama-3.3-70b-versatile",

        temperature=0,

        max_tokens=1400,

        groq_api_key=GROQ_API_KEY
    )

    print("RAG system ready.")


# ==========================================================
# FILE INFORMATION
# ==========================================================

def prepare_available_files():

    global available_files

    available_files = sorted(
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

    return available_files


# ==========================================================
# NORMALIZE FILE NAME
# ==========================================================

def normalize_file_name(file_name):

    if not file_name:
        return ""

    return os.path.basename(
        str(file_name)
    ).strip().lower()


# ==========================================================
# GET FILE DOCUMENTS
# ==========================================================

def get_file_documents(file_name):

    target = normalize_file_name(
        file_name
    )

    return [
        doc
        for doc in documents
        if normalize_file_name(
            doc.metadata.get(
                "source",
                ""
            )
        ) == target
    ]


# ==========================================================
# GET TRUE PAGE COUNT
# ==========================================================

def get_true_page_count(file_name):

    file_docs = get_file_documents(
        file_name
    )

    if not file_docs:
        return 0

    pages = set()

    for doc in file_docs:

        page = doc.metadata.get(
            "page"
        )

        if page is not None:

            try:
                pages.add(
                    int(page)
                )

            except Exception:
                pass

    if pages:
        return len(pages)

    return len(file_docs)


# ==========================================================
# GET FILE CHUNKS
# ==========================================================

def get_file_chunks(file_name):

    target = normalize_file_name(
        file_name
    )

    return [
        doc
        for doc in chunks
        if normalize_file_name(
            doc.metadata.get(
                "source",
                ""
            )
        ) == target
    ]


# ==========================================================
# DETECT FILE FROM QUESTION
# ==========================================================

def detect_file(question):

    if not question:
        return None

    q = question.lower()

    # Exact filename
    for file_name in available_files:

        if file_name.lower() in q:
            return file_name

    # Partial filename
    for file_name in available_files:

        base = os.path.splitext(
            file_name
        )[0].lower()

        if (
            len(base) >= 4
            and base in q
        ):
            return file_name

    return None


# ==========================================================
# COMPARISON QUESTION
# ==========================================================

def is_comparison_question(question):

    q = question.lower()

    keywords = [
        "compare",
        "comparison",
        "compare documents",
        "compare files",
        "compare uploaded documents",
        "compare uploaded files",
        "all uploaded documents",
        "all uploaded files",
        "difference between",
        "differences between",
        "similarity between",
        "similarities between"
    ]

    return any(
        keyword in q
        for keyword in keywords
    )


# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

def remove_duplicates(docs):

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
            if doc.page_content
            else ""
        )

        key = (
            source,
            page,
            content[:250]
        )

        if key not in seen:

            seen.add(key)
            result.append(doc)

    return result


# ==========================================================
# RETRIEVE FROM SPECIFIC FILE
# ==========================================================

def retrieve_from_file(
    question,
    file_name,
    max_results=10
):

    file_chunks = get_file_chunks(
        file_name
    )

    if not file_chunks:
        return []

    temporary_db = FAISS.from_documents(

        documents=file_chunks,

        embedding=embedding_model
    )

    temp_retriever = temporary_db.as_retriever(

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
            "lambda_mult": 0.65
        }
    )

    return temp_retriever.invoke(
        question
    )


# ==========================================================
# RETRIEVE FROM ALL FILES
# ==========================================================

def retrieve_from_all_files(question):

    results = []

    for file_name in available_files:

        try:

            docs = retrieve_from_file(
                question,
                file_name,
                max_results=8
            )

            results.extend(docs)

        except Exception as e:

            print(
                f"Retrieval error for {file_name}: {e}"
            )

    return remove_duplicates(
        results
    )


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(docs):

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
                page = int(page) + 1
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
            if doc.page_content
            else ""
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

    for file_name in available_files:

        pages = get_true_page_count(
            file_name
        )

        file_chunks = get_file_chunks(
            file_name
        )

        lines.append(
            f"- {file_name}: "
            f"{pages} page(s), "
            f"{len(file_chunks)} text chunk(s)"
        )

    return "\n".join(lines)


# ==========================================================
# ASK RAG
# ==========================================================

def ask_rag(question):

    if not vector_db or not retriever or not llm:

        raise Exception(
            "RAG system is not ready. "
            "Please process documents first."
        )

    if not question or not question.strip():

        return {
            "result": "Please enter a question.",
            "source_documents": []
        }

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
            "uploaded files"
        ]
    ):

        return {
            "result": (
                f"There are {len(available_files)} "
                f"uploaded file(s)."
            ),
            "source_documents": []
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
            "generated chunks"
        ]
    ):

        return {
            "result": (
                f"{len(chunks)} "
                f"chunk(s) were generated."
            ),
            "source_documents": []
        }

    # ------------------------------------------------------
    # TOTAL PAGE COUNT
    # ------------------------------------------------------

    if any(
        x in q
        for x in [
            "how many pages",
            "number of pages",
            "total pages",
            "page count"
        ]
    ):

        total_pages = len(
            documents
        )

        return {
            "result": (
                f"There are {len(available_files)} "
                f"uploaded file(s) containing "
                f"{total_pages} "
                f"processed page/document object(s)."
            ),
            "source_documents": []
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

        source_documents = retrieve_from_file(
            question,
            matched_file,
            max_results=10
        )

    elif is_comparison_question(
        question
    ):

        source_documents = retrieve_from_all_files(
            question
        )

    else:

        source_documents = retriever.invoke(
            question
        )

    source_documents = remove_duplicates(
        source_documents
    )

    source_documents = source_documents[:30]

    if not source_documents:

        return {
            "result":
                "I couldn't find the answer "
                "in the uploaded documents.",
            "source_documents": []
        }

    # ------------------------------------------------------
    # CONTEXT
    # ------------------------------------------------------

    context = build_context(
        source_documents
    )

    file_summary = build_file_summary()

    # ------------------------------------------------------
    # FINAL PROMPT
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
        "source_documents": source_documents
    }


# ==========================================================
# PROCESS DOCUMENTS
# ==========================================================

def process_documents(files):

    global documents
    global chunks
    global vector_db
    global retriever
    global llm
    global available_files

    if not files:

        return (
            "Please upload at least one document.",
            default_stats()
        )

    start = time.time()

    try:

        load_documents(files)

        if not documents:

            return (
                "No supported documents were loaded.",
                default_stats()
            )

        create_chunks()

        build_vector_database()

        build_rag()

        prepare_available_files()

        elapsed = round(
            time.time() - start,
            2
        )

        status = f"""
Knowledge Base Ready

Uploaded Files: {len(available_files)}
Processed Pages/Documents: {len(documents)}
Generated Chunks: {len(chunks)}
Vectors: {vector_db.index.ntotal}
Processing Time: {elapsed} sec
"""

        stats = f"""
| Metric | Status |
|:--|:--:|
| 📂 Uploaded Files | **{len(available_files)}** |
| 📄 Processed Pages | **{len(documents)}** |
| 🧩 Generated Chunks | **{len(chunks)}** |
| 🔍 Retrieved Chunks | **0** |
| ⚡ Processing Time | **{elapsed} sec** |
| 🤖 Model | **Groq Llama** |
"""

        return status, stats

    except Exception as e:

        print(
            "PROCESSING ERROR:",
            str(e)
        )

        return (
            f"Error: {str(e)}",
            default_stats()
        )


# ==========================================================
# DEFAULT STATS
# ==========================================================

def default_stats():

    return """
| Metric | Status |
|:--|:--:|
| 📂 Uploaded Files | **0** |
| 📄 Processed Pages | **0** |
| 🧩 Generated Chunks | **0** |
| 🔍 Retrieved Chunks | **0** |
| ⚡ Response Time | **0.00 sec** |
| 🤖 Model | **Groq Llama** |
"""


# ==========================================================
# CHAT FUNCTION
# ==========================================================

def rag_chat(
    question,
    chat_history
):

    if chat_history is None:
        chat_history = []

    if not vector_db:

        return (
            chat_history,
            "",
            "Please process documents first.",
            default_stats()
        )

    if not question or not question.strip():

        return (
            chat_history,
            "",
            "Please enter a question.",
            default_stats()
        )

    start = time.time()

    try:

        result = ask_rag(
            question.strip()
        )

        answer = result.get(
            "result",
            "I couldn't find the answer in the uploaded documents."
        )

        source_documents = result.get(
            "source_documents",
            []
        )

    except Exception as e:

        return (
            chat_history,
            "",
            f"Error: {str(e)}",
            default_stats()
        )

    elapsed = round(
        time.time() - start,
        2
    )

    # ------------------------------------------------------
    # SOURCES
    # ------------------------------------------------------

    source_groups = {}

    for doc in source_documents:

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
                page = int(page) + 1
            except Exception:
                pass

        if source not in source_groups:
            source_groups[source] = []

        if (
            page is not None
            and page not in source_groups[source]
        ):
            source_groups[source].append(
                page
            )

    sources = []

    for source, pages in source_groups.items():

        pages.sort()

        if pages:

            if len(pages) == 1:

                sources.append(
                    f"• {source} "
                    f"(Page {pages[0]})"
                )

            else:

                page_text = ", ".join(
                    str(p)
                    for p in pages
                )

                sources.append(
                    f"• {source} "
                    f"(Pages {page_text})"
                )

        else:

            sources.append(
                f"• {source}"
            )

    if sources:

        final_answer = (
            str(answer).strip()
            + "\n\n## 📄 Sources\n\n"
            + "\n".join(sources)
        )

    else:

        final_answer = str(
            answer
        ).strip()

    # ------------------------------------------------------
    # CHAT HISTORY
    # ------------------------------------------------------

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )

    retrieved_count = len(
        source_documents
    )

    stats = f"""
| Metric | Status |
|:--|:--:|
| 📂 Uploaded Files | **{len(available_files)}** |
| 📄 Processed Pages | **{len(documents)}** |
| 🧩 Generated Chunks | **{len(chunks)}** |
| 🔍 Retrieved Chunks | **{retrieved_count}** |
| ⚡ Response Time | **{elapsed} sec** |
| 🤖 Model | **Groq Llama** |
"""

    return (
        chat_history,
        "",
        "Response Generated",
        stats
    )


# ==========================================================
# CLEAR CHAT
# ==========================================================

def clear_chat():

    return (
        [],
        "",
        "Chat Cleared",
        default_stats()
    )


# ==========================================================
# CSS
# ==========================================================

CUSTOM_CSS = """

.gradio-container {
    max-width: 1700px !important;
    margin: auto !important;
    padding: 20px !important;
    background: #F3F6FB;
}

footer {
    display: none !important;
}

#header-card {
    background: linear-gradient(
        90deg,
        #0F172A,
        #2563EB
    );
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
}

#main-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

#department {
    font-size: 22px;
    font-weight: 600;
    color: #E8F0FF;
}

#university {
    font-size: 18px;
    color: #D8E4FF;
    margin-top: 6px;
}

.version-box {
    background: white;
    border-radius: 15px;
    text-align: center;
    padding: 15px;
    color: #0F172A;
    font-weight: 700;
}

button {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

textarea {
    border-radius: 12px !important;
}

.gr-column {
    border-radius: 18px !important;
}
"""


# ==========================================================
# GRADIO INTERFACE
# ==========================================================

with gr.Blocks(
    title="Smart Multilingual AI RAG Assistant",
    css=CUSTOM_CSS
) as demo:

    # ------------------------------------------------------
    # HEADER
    # ------------------------------------------------------

    with gr.Row(
        elem_id="header-card"
    ):

        with gr.Column(
            scale=1,
            min_width=120
        ):

            gr.Markdown(
                "🤖",
                elem_id="logo"
            )

        with gr.Column(
            scale=8
        ):

            gr.HTML(
                """
                <div id="main-title">
                🤖 Smart Multilingual AI RAG Assistant
                </div>

                <div id="department">
                Department of Computer Science
                </div>

                <div id="university">
                University of Okara • MSCS Research Project
                </div>
                """
            )

        with gr.Column(
            scale=1,
            min_width=120
        ):

            gr.HTML(
                """
                <div class="version-box">
                Version<br>
                <b>6.02</b>
                </div>
                """
            )

    # ------------------------------------------------------
    # MAIN AREA
    # ------------------------------------------------------

    with gr.Row():

        # --------------------------------------------------
        # LEFT PANEL
        # --------------------------------------------------

        with gr.Column(
            scale=1,
            min_width=350
        ):

            gr.Markdown(
                """
                ## 📂 Document Management

                Upload one or more documents and process
                them to build the knowledge base.
                """
            )

            file_upload = gr.Files(
                label="📁 Select Documents",
                file_count="multiple",
                file_types=[
                    ".pdf",
                    ".docx",
                    ".txt",
                    ".csv",
                    ".pptx"
                ],
                height=220
            )

            process_btn = gr.Button(
                "⚙️ Process Documents",
                variant="primary",
                size="lg"
            )

            gr.Markdown(
                "## 📊 System Status"
            )

            status = gr.Textbox(
                show_label=False,
                lines=7,
                interactive=False,
                placeholder=(
                    "Ready to process documents..."
                )
            )

            gr.Markdown(
                "## 📈 Project Statistics"
            )

            stats_card = gr.Markdown(
                default_stats()
            )

            gr.Markdown(
                """
                ---

                ### ℹ️ System Information

                ✅ Multiple Document Support

                ✅ Semantic Search (FAISS)

                ✅ Multilingual Embeddings

                ✅ Context-Aware Retrieval

                ✅ AI Response Generation

                ✅ Source Citation

                ---
                """
            )

        # --------------------------------------------------
        # RIGHT PANEL
        # --------------------------------------------------

        with gr.Column(
            scale=3,
            min_width=700
        ):

            gr.Markdown(
                """
                ## 🤖 AI Research Assistant

                Ask questions about your uploaded documents
                and receive context-aware answers with sources.
                """
            )

            chatbot = gr.Chatbot(
                label="💬 AI Conversation",
                height=650
            )

            question = gr.Textbox(
                label="Ask Your Question",
                placeholder=(
                    "Example: Summarize the uploaded document..."
                ),
                lines=2
            )

            with gr.Row():

                ask_btn = gr.Button(
                    "🚀 Get Answer",
                    variant="primary",
                    scale=3
                )

                clear_btn = gr.Button(
                    "🗑️ Clear Chat",
                    variant="secondary",
                    scale=1
                )

            gr.Markdown(
                """
                **💡 Example Questions**

                • Summarize this document.

                • What are the main objectives?

                • List the key findings.

                • Explain the methodology.

                • Extract important conclusions.

                • Compare the uploaded documents.
                """
            )

    # ------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------

    process_btn.click(
        fn=process_documents,
        inputs=file_upload,
        outputs=[
            status,
            stats_card
        ]
    )

    ask_btn.click(
        fn=rag_chat,
        inputs=[
            question,
            chatbot
        ],
        outputs=[
            chatbot,
            question,
            status,
            stats_card
        ]
    )

    question.submit(
        fn=rag_chat,
        inputs=[
            question,
            chatbot
        ],
        outputs=[
            chatbot,
            question,
            status,
            stats_card
        ]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=[
            chatbot,
            question,
            status,
            stats_card
        ]
    )


# ==========================================================
# LAUNCH
# ==========================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",    
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )
