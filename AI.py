import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub  # Optional LLM (you can skip QA if no model)
from dotenv import load_dotenv

# Load environment variables (for Hugging Face token if needed)
load_dotenv()

# ✅ Optional: Hugging Face Hub token (for inference if using HuggingFaceHub LLM)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Set Streamlit page config
st.set_page_config(page_title="AI STEM Tutor", layout="centered")
st.title(" AI Tutor for STEM Subjects")
st.markdown("Powered by Agentic AI, RAG, and Granite Principles")

# ✅ Upload PDF
uploaded_file = st.file_uploader("Upload a STEM textbook or PDF", type="pdf")

if uploaded_file:
    with open("chapter.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ✅ Load PDF
    loader = PyPDFLoader("chapter.pdf")
    docs = loader.load()

    # ✅ Use HuggingFace Embeddings (free + local)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Embed into FAISS DB
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # ✅ Optional: LLM using HuggingFaceHub (or comment this out)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # lightweight model good for Q&A
        model_kwargs={"temperature": 0.1, "max_new_tokens": 256},
        huggingfacehub_api_token=hf_token,
    )

    # ✅ Build Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    st.success("PDF processed! Ask your question below ")
    question = st.text_input("Ask a STEM Question (e.g., Explain Ohm’s Law)")

    if question:
        ethical_prompt = (
            "As an ethical AI STEM tutor, explain this using only the uploaded content. "
            "Be step-by-step, simple, and avoid hallucination.\n"
            f"Question: {question}"
        )
        response = qa_chain.run(ethical_prompt)
        st.write("### Answer:")
        st.success(response)


