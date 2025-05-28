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
if os.getenv("ASK_MODE", "streamlit") == "local":
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
        #print(f"VECTOR_STORE_BUCKET: {bucket}")
        #print(f"VECTOR_STORE_PREFIX: {prefix}")

        st.write(f"VECTOR_STORE_BUCKET: {bucket}")
        st.write(f"VECTOR_STORE_PREFIX: {prefix}")

        if not os.path.exists(persist_path):
            os.makedirs(persist_path)
        for fname in expected_files:
            key = f"{prefix}/{fname}"
            local_path = os.path.join(persist_path, fname)
            print(f"Downloading {fname} from S3...")
            st.write(f"Downloading {key} from S3...")
            try:
                s3.download_file(bucket, key, local_path)
            except Exception as e:
                print(f"Error downloading {fname}: {e}")
                st.write(f"Error downloading {fname}: {e}")
    # After download, check again
    for fname in expected_files:
        if not os.path.exists(os.path.join(persist_path, fname)):
            raise FileNotFoundError(f"Vector store file missing: {fname}")

# --- Load Vector Store ---
def load_vector_store():
    mode = os.getenv("ASK_MODE", "streamlit")
    if mode == "local":
        print("🖥️ Running in LOCAL mode (loading vector store from ../vector_store)")
        persist_path = str(Path(__file__).parent.parent / "vector_store")
    else:
        print("☁️ Running in CLOUD mode (loading vector store from /tmp/vector_store)")
        persist_path = "/tmp/vector_store"
        ensure_vector_store_cloud()  # Only does anything in cloud mode
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embedder, allow_dangerous_deserialization=True)

# --- Load Components ---
def load_components():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3})
    generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=QA_PROMPT)
    # Get list of known product names
    known_products = set()
    for doc in vector_store.docstore._dict.values():
        name = doc.metadata.get("product_name", "").lower()
        if name:
            known_products.add(name)
    return vector_store, retriever, qa_chain, llm, known_products

# --- CLI Mode (UNTOUCHED) ---
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
    import streamlit as st
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
            result = qa_chain.invoke({"input_documents": truncated_docs, "question": user_input})
            st.markdown(f"**🧠 Answer:** {result.strip()}")
            st.markdown("### 🔍 Sources used:")
            for doc in truncated_docs:
                meta = doc.metadata
                st.markdown(f"- **{meta.get('product_name', 'Unknown')}** (model {meta.get('model', '-')})")
                st.caption(doc.page_content[:200].replace("\n", " "))

# --- Entry Point ---
if __name__ == "__main__":
    if os.getenv("ASK_MODE", "streamlit") == "local":
        run_cli_mode()
    else:
        run_streamlit_mode()
