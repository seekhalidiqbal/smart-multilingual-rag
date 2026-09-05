import streamlit as st
import os
import fitz  # PyMuPDF
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Smart Multilingual RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS - HEADER BIG + SIDEBAR FIX
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
        padding: 25px 30px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 25px;
        margin-top: 0rem;
        margin-bottom: 25px;
        color: white;
        position: relative;
        min-height: 110px;
    }}
    .main-header img {{
        width: 85px;
        height: 85px;
        border-radius: 10px;
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
    .header-center h1 {{margin: 0; font-size: 24px; font-weight: 700;}}
    .header-center p {{margin: 5px 0 0 0; font-size: 14px; opacity: 0.95;}}
    [data-testid="stSidebar"] {{
        display: block !important;
        min-width: 350px !important;
        max-width: 350px !important;
    }}
    #MainMenu {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = 0
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ==========================================================
# HEADER
# ==========================================================
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <img src="https://www.uok.edu.pk/logo.png" alt="Logo">
        <div class="header-center">
            <h1>Smart_Multilingual_Multi_Document_AI_RAG_Assistant</h1>
            <p>🏛️ Department of Computer Science | 🎓 University of Okara • MSCS Research Project | ⚡ Version 6.24</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# SIDEBAR - DOCUMENT MANAGEMENT
# ==========================================================
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.write("Upload one or more documents and then click Process Documents to build the knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Select Documents", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                process_documents(uploaded_files)
                st.success("Documents Processed Successfully!")
                st.rerun()
        else:
            st.warning("Please upload at least one document.")
    
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    st.checkbox("✅ Multiple Document Support", value=True, disabled=True)
    st.checkbox("✅ Semantic Search (FAISS)", value=True, disabled=True)
    st.checkbox("✅ Context-Aware Retrieval", value=True, disabled=True)
    st.checkbox("✅ AI Response Generation", value=True, disabled=True)
    st.checkbox("✅ Source Citation", value=True, disabled=True)
    st.checkbox("✅ Multilingual Support", value=True, disabled=True)

# ==========================================================
# MAIN COLUMNS
# ==========================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### 📊 Project Statistics")
    
    # Status
    if st.session_state.vectorstore:
        st.info("✅ System Status: Ready - Ask any question")
    else:
        st.info("📄 System Status: Awaiting Document Upload")
    
    # Stats Table
    stats_data = {
        "Metric": ["Uploaded Files", "Processed Documents", "Generated Chunks", "Retrieved Chunks", "Questions Asked", "Answers Generated"],
        "Value": [
            len(st.session_state.uploaded_files),
            st.session_state.processed_docs,
            st.session_state.get("chunks", 0),
            st.session_state.get("retrieved", 0),
            st.session_state.get("questions", 0),
            len([m for m in st.session_state.messages if m["role"] == "assistant"])
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

with col_right:
    st.markdown("### 💡 Example Questions")
    examples = [
        "🚀 Summarize this document.",
        "🎯 What are the main objectives?",
        "📋 List the key findings.",
        "🔬 Explain the methodology.",
        "📝 Extract important conclusions.",
        "📊 Compare the uploaded documents."
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex.replace("🚀 ", "").replace("🎯 ", "").replace("📋 ", "").replace("🔬 ", "").replace("📝 ", "").replace("📊 ", "")})
            st.rerun()

# ==========================================================
# FUNCTIONS
# ==========================================================
def process_documents(files):
    all_text = []
    file_names = []
    for file in files:
        file_names.append(file.name)
        if file.name.endswith(".pdf"):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in pdf:
                text += page.get_text()
            all_text.append(text)
    
    st.session_state.uploaded_files = file_names
    st.session_state.processed_docs = len(files)
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(all_text)
    st.session_state.chunks = len(chunks)
    
    # Embeddings + Vectorstore
    # IMPORTANT: Yahan apni Google API Key daalo
    os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_GOOGLE_API_KEY_HERE"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # QA Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt_template = """Use the following context to answer the question. If you don't know, say so.
    Context: {context}
    Question: {question}
    Answer in the same language as the question.
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

def get_answer(query):
    if st.session_state.qa_chain is None:
        return "Please upload and process documents first.", []
    
    result = st.session_state.qa_chain({"query": query})
    answer = result["result"]
    sources = [doc.page_content[:200] + "..." for doc in result["source_documents"]]
    
    st.session_state.retrieved = len(result["source_documents"])
    st.session_state.questions = st.session_state.get("questions", 0) + 1
    
    return answer, sources

# ==========================================================
# CHAT INTERFACE
# ==========================================================
st.markdown("---")
# Purani chat show karo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for i, src in enumerate(message["sources"]):
                    st.write(f"**Source {i+1}:** {src}")

# Naya question input
if prompt := st.chat_input("Type your question and press Enter to submit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, sources = get_answer(prompt)
            st.markdown(response)
            if sources:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources):
                        st.write(f"**Source {i+1}:** {src}")
    
    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
    st.rerun()
