import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# --- Prompt ---
QA_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant answering questions using appliance manuals.
Only use the provided context to answer the question.
If the answer isn't clear or relevant, say "I couldn’t find that in the manual."
Context:
{context}
Question: {question}
Answer:
""")

# --- Environment Loading ---
ask_mode = os.getenv("ASK_MODE", "streamlit")
if ask_mode == "local" or ask_mode == "st_local":
    dotenv_path = Path("AskMyManualsLocal.env")
else:
    dotenv_path = Path("AskMyManualsS3.env")
load_dotenv(dotenv_path=dotenv_path)

# --- Streamlit Cloud: Ensure Vector Store Download ---
def ensure_vector_store_cloud():
    persist_path = "/tmp/vector_store"
    expected_files = ["index.faiss", "index.pkl"]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(persist_path, f))]
    if missing:
        import boto3
        s3 = boto3.client("s3")
        bucket = os.getenv("VECTOR_STORE_BUCKET")
        prefix = os.getenv("VECTOR_STORE_PREFIX", "vector_store")

        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
        for fname in expected_files:
            key = f"{prefix}/{fname}"
            local_path = os.path.join(persist_path, fname)
            print(f"Downloading {fname} from S3...")
            try:
                s3.download_file(bucket, key, local_path)
            except Exception as e:
                print(f"Error downloading {fname}: {e}")
    # After download, check again
    for fname in expected_files:
        if not os.path.exists(os.path.join(persist_path, fname)):
            raise FileNotFoundError(f"Vector store file missing: {fname}")

# --- Load Vector Store ---
def load_vector_store():
    """
    Loads the FAISS vector store using the embedding model specified by the EMBEDDER_MODEL environment variable.
    Supports both local (CLI) and cloud (Streamlit) modes.
    """
    mode = os.getenv("ASK_MODE", "streamlit")
    embedder_model_short_name = os.getenv("EMBEDDER_MODEL")

    # Mapping short names to full model names
    model_map = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5": "intfloat/e5-large-v2",
        "bge": "BAAI/bge-base-en-v1.5",
        "jina": "jinaai/jina-embeddings-v2-base-en"
    }

    embedder_model = model_map.get(embedder_model_short_name)
    if not embedder_model:
        print("❌ Embedder model not found or not set. Please set the EMBEDDER_MODEL environment variable.")
        return
    else:
        print(f"Using embedder {embedder_model}")

    if mode == "local":
        print("🖥️ Running in LOCAL mode (loading vector store from ../vector_store)")
        persist_path = str(Path(__file__).parent.parent / "vector_store")
        embedder = HuggingFaceEmbeddings(model_name=embedder_model)
    elif mode == "st_local":
        print("🖥️ Running in STREAMLIT LOCALHOST mode (loading vector store from /tmp/vector_store)")
        persist_path = "/tmp/vector_store"
        embedder = HuggingFaceEmbeddings(model_name=embedder_model)        
    else:
        print("☁️ Running in CLOUD mode (loading vector store from /tmp/vector_store)")
        persist_path = "/tmp/vector_store"
        ensure_vector_store_cloud()
        st.write("after ensure_vector_store_cloud")
        hf_token = st.secrets.get("HF_TOKEN")
        st.write("got HFtoken")
        if not hf_token:
            raise ValueError("HF_TOKEN not set in Streamlit secrets. Please add your Hugging Face token.")
        try:
            embedder = HuggingFaceEmbeddings(
                model_name=embedder_model,
                model_kwargs={"use_auth_token": hf_token}
            )
            st.write("Embedder loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load HuggingFaceEmbeddings: {e}")
            st.stop()

    return FAISS.load_local(persist_path, embedder, allow_dangerous_deserialization=True)

# --- Load Components ---
def load_components():
    mode = os.getenv("ASK_MODE", "streamlit")

    vector_store = load_vector_store()
    if mode == "streamlit" or mode == "st_local":
        st.write("load_vector_store completed successfully")
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3})
    if mode == "streamlit" or mode == "st_local":
        st.write("before LLM/generator")
    generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT)
    if mode == "streamlit" or mode == "st_local":
        st.write("after QA chain")
    # Get list of known product names
    known_products = set()
    for doc in vector_store.docstore._dict.values():
        name = doc.metadata.get("product_name", "").lower()
        if name:
            known_products.add(name)
    return vector_store, retriever, qa_chain, llm, known_products

# --- CLI Mode ---
def run_cli_mode():
    print("\n📘 Ask My Manuals (Command Line Mode)")
    print("Type 'exit' to quit.\n")
    vector_store, retriever, qa_chain, llm, known_products = load_components()
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        enriched_query = user_input.lower()
        docs = retriever.invoke(enriched_query)
        # 🔍 Product filtering
        product = next((name for name in known_products if name in enriched_query), None)
        if product:
            docs = [doc for doc in docs if doc.metadata.get("product_name", "").lower() == product]
        # 🔧 Truncate and limit docs
        truncated_docs = [
            Document(page_content=doc.page_content[:400], metadata=doc.metadata)
            for doc in docs
        ][:3]
        result = qa_chain.invoke({"context": truncated_docs, "question": user_input})
        print("\n🧠 Answer:", result.strip())
        print("\n🔍 Sources used:")
        for doc in truncated_docs:
            meta = doc.metadata
            print(f"- {meta.get('product_name', 'Unknown')} (model {meta.get('model', '-')})")
            print(" Preview:", doc.page_content[:200].replace("\n", " "), "\n")

# --- Streamlit UI ---
def run_streamlit_mode():
    st.set_page_config(page_title="Ask My Manuals", page_icon="📘")
    st.title("📘 Ask My Manuals")
    st.write("Ask a question about your appliances and devices.")
    try:
        vector_store, retriever, qa_chain, llm, known_products = load_components()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Thinking..."):
            enriched_query = user_input.lower()
            docs = retriever.invoke(enriched_query)
            product = next((name for name in known_products if name in enriched_query), None)
            if product:
                docs = [doc for doc in docs if doc.metadata.get("product_name", "").lower() == product]
            truncated_docs = [
                Document(page_content=doc.page_content[:400], metadata=doc.metadata)
                for doc in docs
            ][:3]
            result = qa_chain.invoke({"context": truncated_docs, "question": user_input})
            st.markdown(f"**🧠 Answer:** {result.strip()}")
            st.markdown("### 🔍 Sources used:")
            for doc in truncated_docs:
                meta = doc.metadata
                st.markdown(f"- **{meta.get('product_name', 'Unknown')}** (model {meta.get('model', '-')})")
                st.caption(doc.page_content[:200].replace("\n", " "))

def run_streamlit_mode_localhost():
    st.set_page_config(page_title="Ask My Manuals (Localhost)", page_icon="📘")
    st.title("📘 Ask My Manuals (Localhost)")
    st.write("Ask a question about your appliances and devices. (Localhost mode, no secrets required)")
    try:
        os.environ["ASK_MODE"] = "st_local"  # Ensure correct mode for this run
        vector_store, retriever, qa_chain, llm, known_products = load_components()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Thinking..."):
            enriched_query = user_input.lower()
            docs = retriever.invoke(enriched_query)
            product = next((name for name in known_products if name in enriched_query), None)
            if product:
                docs = [doc for doc in docs if doc.metadata.get("product_name", "").lower() == product]
            truncated_docs = [
                Document(page_content=doc.page_content[:400], metadata=doc.metadata)
                for doc in docs
            ][:3]
            result = qa_chain.invoke({"context": truncated_docs, "question": user_input})
            st.markdown(f"**🧠 Answer:** {result.strip()}")
            st.markdown("### 🔍 Sources used:")
            for doc in truncated_docs:
                meta = doc.metadata
                st.markdown(f"- **{meta.get('product_name', 'Unknown')}** (model {meta.get('model', '-')})")
                st.caption(doc.page_content[:200].replace("\n", " "))

# --- Entry Point ---
if __name__ == "__main__":
    ask_mode = os.getenv("ASK_MODE", "streamlit")
    if ask_mode == "local":
        run_cli_mode()
    elif ask_mode == "st_local":
        run_streamlit_mode_localhost()
    else:
        run_streamlit_mode()
