with st.sidebar:
    st.markdown("### 📂 Document Management")
    st.caption("Upload one or more documents and then click Process Documents to build the knowledge base.")

    uploaded_files = st.file_uploader(
        "📁 Select Documents", 
        type=["pdf", "docx", "txt", "csv", "pptx"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
        if uploaded_files:  # <- ye line bilkul 4 space andar honi chahiye
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
