import os
import sys
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
import boto3

# --- Load Env Vars ---
load_dotenv(dotenv_path=Path("AskMyManualsS3.env"))

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


def download_vector_store_from_s3():
    bucket = os.getenv("S3_BUCKET_NAME")
    s3_prefix = "vector_store/"
    local_path = "/tmp/vector_store"
    os.makedirs(local_path, exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    for file_name in ["index.faiss", "index.pkl"]:
        s3.download_file(bucket, f"{s3_prefix}{file_name}", f"{local_path}/{file_name}")

    print("✅ Vector store downloaded from S3.")
    return local_path


def load_vector_store():
    mode = os.getenv("ASK_MODE", "streamlit")
    if mode == "local":
        print("🖥️ Running in LOCAL mode (loading vector store from ../vector_store)")
        persist_path = str(Path(__file__).parent.parent / "vector_store")
    else:
        print("☁️ Running in CLOUD mode (downloading vector store from S3)")
        persist_path = download_vector_store_from_s3()

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embedder, allow_dangerous_deserialization=True)


def load_components():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)

    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    qa_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    # collect known product names
    docs = vector_store.similarity_search("*")
    known_products = list(set([doc.metadata.get("product_name", "").lower() for doc in docs]))

    return vector_store, retriever, qa_chain, llm, known_products


def run_cli_mode():
    print("\n\U0001F4D8 Ask My Manuals (Command Line Mode)")
    print("Type 'exit' to quit.\n")

    vector_store, retriever, qa_chain, llm, known_products = load_components()

    while True:
        user_input = input("Your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        enriched_query = user_input + f". Possible products: {', '.join(known_products)}"
        docs = retriever.get_relevant_documents(enriched_query)
        result = qa_chain.invoke({"input_documents": docs, "question": user_input})

        print(f"\n\U0001F9E0 Answer: {result.strip()}\n")
        print("\U0001F50D Sources used:")
        for doc in docs:
            meta = doc.metadata
            snippet = doc.page_content[:300].strip().replace("\n", " ")
            print(f"- {meta.get('product_name', 'unknown')} (model {meta.get('model', '-')}, page {meta.get('page_number', '-')})\n  Preview: {snippet}\n")


def run_streamlit_mode():
    st.set_page_config(page_title="Ask My Manuals", page_icon="📘")
    st.title("📘 Ask My Manuals")
    st.write("Ask a question about your appliances and devices.")

    vector_store, retriever, qa_chain, llm, known_products = load_components()

    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Thinking..."):
            enriched_query = user_input + f". Possible products: {', '.join(known_products)}"
            docs = retriever.get_relevant_documents(enriched_query)
            result = qa_chain.invoke({"input_documents": docs, "question": user_input})

            st.markdown(f"**🧠 Answer:** {result.strip()}")

            st.markdown("### 🔍 Sources used:")
            for doc in docs:
                meta = doc.metadata
                snippet = doc.page_content[:300].strip().replace("\n", " ")
                st.markdown(f"- **{meta.get('product_name', 'unknown')}** (model {meta.get('model', '-')}, page {meta.get('page_number', '-')})\n  Preview: {snippet}")


# --- Main Execution ---
if __name__ == "__main__":
    mode = os.getenv("ASK_MODE", "streamlit")
    if mode == "cli":
        run_cli_mode()
    else:
        run_streamlit_mode()
