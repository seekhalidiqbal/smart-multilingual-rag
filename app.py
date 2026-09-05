import streamlit as st
import os
import pandas as pd
import fitz # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

st.set_page_config(page_title="Smart Multilingual RAG Assistant", page_icon="🧠", layout="wide")

# CSS FOR EXACT UI
st.markdown("""
<style>
   .main-header {background: linear-gradient(90deg, #0D47A1, #1976D2); padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px;}
   .main-header h1 {margin: 0; font-size: 22px;}
   .stButton>button {width: 100%;}
    [data-testid="stSidebar"] {min-width: 350px;}
</style>
""", unsafe_allow_html=True)

# SESSION
for key in ["messages","vectorstore","qa_chain","processed_docs","uploaded_files","questions","chunks","retrieved"]:
    if key not in st.session_state: st.session_state[key] = 0 if key in ["questions","chunks","retrieved","processed_docs"] else [] if key in ["messages","uploaded_files"] else None

# HEADER
st.markdown("""
<div class="main-header">
    <h1>Smart_Multilingual_Multi_Document_AI_RAG_Assistant</h1>
    <p>🏛️ Department of Computer Science | 🎓 University of Okara • MSCS Research Project | ⚡ Version 6.24</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### 📁 Document Management")
    st.write("Upload one or more documents and then click Process Documents to build the knowledge base.")
    uploaded_files = st.file_uploader("Select Documents", type=["pdf"], accept_multiple_files=True)
    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files: process_documents(uploaded_files)
        else: st.warning("Please upload at least one document.")
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    for item in ["Multiple Document Support","Semantic Search (FAISS)","Context-Aware Retrieval","AI Response Generation","Source Citation","Multilingual Support"]:
        st.checkbox(f"✅ {item}", value=True, disabled=True)

# MAIN
col1, col2 = st.columns([1.2,1])
with col1:
    st.markdown("### 📊 Project Statistics")
    if st.session_state.vectorstore: st.success("✅ System Status: Ready - Ask any question")
    else: st.info("📄 System Status: Awaiting Document Upload")
    
    stats = pd.DataFrame({
        "Metric": ["Uploaded Files","Processed Documents","Generated Chunks","Retrieved Chunks","Questions Asked","Answers Generated"],
        "Value": [len(st.session_state.uploaded_files),st.session_state.processed_docs,st.session_state.chunks,st.session_state.retrieved,st.session_state.questions,len([m for m in st.session_state.messages if m["role"]=="assistant"])]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 💡 Example Questions")
    for ex in ["🚀 Summarize this document.","🎯 What are the main objectives?","📋 List the key findings."]:
        if st.button(ex, key=ex): st.session_state.messages.append({"role":"user","content":ex[2:]}); st.rerun()

def process_documents(files):
    try:
        docs=[]; names=[]
        for file in files:
            names.append(file.name)
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in pdf])
            docs.append(Document(page_content=text, metadata={"source":file.name}))
        
        st.session_state.uploaded_files=names; st.session_state.processed_docs=len(files)
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=splitter.split_documents(docs); st.session_state.chunks=len(chunks)
        
        # KEY: API KEY from Secrets
        api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        if not api_key: st.error("GOOGLE_API_KEY not found. Add it in Settings > Secrets"); return
        os.environ["GOOGLE_API_KEY"]=api_key
        
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectorstore=FAISS.from_documents(chunks, embeddings)
        
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt=PromptTemplate(template="Context: {context}\nQuestion: {question}\nAnswer in same language.", input_variables=["context","question"])
        st.session_state.qa_chain=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=st.session_state.vectorstore.as_retriever(k=4),return_source_documents=True,chain_type_kwargs={"prompt":prompt})
        st.success("Documents Processed Successfully!")
    except Exception as e: st.error(f"Error: {e}")

def get_answer(query):
    if not st.session_state.qa_chain: return "Please process documents first.",[]
    result=st.session_state.qa_chain({"query":query})
    st.session_state.retrieved=len(result["source_documents"]); st.session_state.questions+=1
    sources=[f"{d.metadata['source']}: {d.page_content[:150]}..." for d in result["source_documents"]]
    return result["result"], sources

# CHAT
for m in st.session_state.messages:
    with st.chat_message(m["role"]): 
        st.markdown(m["content"])
        if "sources" in m: 
            with st.expander("📚 Sources"): 
                for s in m["sources"]: st.write(s)

if prompt:=st.chat_input("Type your question and press Enter to submit..."):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans,src=get_answer(prompt); st.markdown(ans)
            if src: 
                with st.expander("📚 Sources"): 
                    for s in src: st.write(s)
    st.session_state.messages.append({"role":"assistant","content":ans,"sources":src}); st.rerun()
