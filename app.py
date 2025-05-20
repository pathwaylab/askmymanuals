import os
import sys
import boto3
import difflib
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Try importing Streamlit only if needed
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# --- Prompt Template ---
template = """
You are a helpful assistant answering questions using appliance manuals.

Only use the provided context to answer the question.
If the answer isn't clear or relevant, say "I couldn’t find that in the manual."

Context:
{context}

Question: {question}
Answer:
"""
QA_PROMPT = PromptTemplate.from_template(template)

# --- Load environment variables ---
load_dotenv(dotenv_path=Path("AskMyManualsS3.env"))

# --- Load vector store from FAISS ---
def load_vector_store():
    ask_mode = os.getenv("ASK_MODE", "cloud").lower()

    if ask_mode == "local":
        print("🖥️ Running in LOCAL mode (loading vector store from ../vector_store)")
        persist_path = "../vector_store"
    else:
        print("☁️ Running in CLOUD mode (loading vector store from /tmp/vector_store)")
        persist_path = "/tmp/vector_store"

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embeddings=embedder, allow_dangerous_deserialization=True)

# --- Load components ---
def load_components():
    vector_store = load_vector_store()

    product_names = list(set([
        doc.metadata.get("product_name", "")
        for doc in vector_store.docstore._dict.values()
        if doc.metadata.get("product_name")
    ]))

    generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M", max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=generator)

    return vector_store, llm, product_names

# --- Fuzzy product matching ---
def detect_product_fuzzy(user_input: str, known_products: list) -> str:
    lowered = user_input.lower()
    matches = difflib.get_close_matches(lowered, known_products, n=1, cutoff=0.6)
    return matches[0] if matches else None

# --- App mode selection ---
def run_streamlit_ui():
    st.set_page_config(page_title="Ask My Manuals", page_icon="📘")
    vector_store, llm, known_products = load_components()

    st.title("📘 Ask My Manuals")
    st.write("Ask a question about your appliances and devices.")

    user_input = st.text_input("Your question:")

    if user_input:
        with st.spinner("Thinking..."):
            product = detect_product_fuzzy(user_input, known_products)

            if product:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2, "filter": {"product_name": product}})
            else:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2})

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_PROMPT}
            )

            result = qa_chain.invoke(user_input)
            st.markdown(f"**🧠 Answer:** {result['result'].strip()}")

            st.markdown("### 🔍 Sources used:")
            for doc in result["source_documents"]:
                meta = doc.metadata
                manual = meta.get("product_name", "Unknown")
                model = meta.get("model", "-")
                page = meta.get("page_number", "Unknown")
                st.markdown(f"- **{manual}** (model {model}), page {page}")

def run_cli_mode():
    vector_store, llm, known_products = load_components()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    print("\n📘 Ask My Manuals (Command Line Mode)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        product = detect_product_fuzzy(user_input, known_products)
        if product:
            retriever = vector_store.as_retriever(search_kwargs={"k": 2, "filter": {"product_name": product}})
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        qa_chain.retriever = retriever
        result = qa_chain.invoke(user_input)

        print("\n🧠 Answer:", result["result"].strip())
        print("\n🔍 Sources used:")
        for doc in result["source_documents"]:
            meta = doc.metadata
            manual = meta.get("product_name", "Unknown")
            model = meta.get("model", "-")
            page = meta.get("page_number", "Unknown")
            print(f"- {manual} (model {model}), page {page}")
        print("\n")

# --- Entry point ---
if __name__ == "__main__":
    ask_mode = os.getenv("ASK_MODE", "cloud").lower()
    if ask_mode == "local":
        run_cli_mode()
    else:
        if STREAMLIT_AVAILABLE:
            run_streamlit_ui()
        else:
            print("❌ Streamlit is not available, and ASK_MODE is not set to local.")
