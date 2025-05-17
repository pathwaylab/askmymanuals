import os
import streamlit as st
import boto3
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from dotenv import load_dotenv
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- Load Components ---
st.set_page_config(page_title="Ask My Manuals", page_icon="📘")

# Prompt template
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

# Load AWS credentials
load_dotenv(dotenv_path=Path("AskMyManualsS3.env"))
print("AWS Access key : ", os.getenv("AWS_ACCESS_KEY_ID"))

def download_vector_store_from_s3():
    bucket = os.getenv("S3_BUCKET_NAME")
    s3_prefix = "vector_store/"
    local_path = Path("/tmp/vector_store")
    local_path.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    for file_name in ["index.faiss", "index.pkl"]:
        s3.download_file(bucket, f"{s3_prefix}{file_name}", str(local_path / file_name))

    print("✅ Vector store downloaded from S3.")
    return local_path

@st.cache_resource
def load_components():
    # Load vector store
    vector_store_path = download_vector_store_from_s3()
    vector_store = FAISS.load_local(
        str(vector_store_path),
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Load model
    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)

    # Build RetrievalQA chain with prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    return qa_chain

# Load chain
qa_chain = load_components()

# --- Streamlit UI ---
st.title("📘 Ask My Manuals")
st.write("Ask a question about your appliances and devices.")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke(user_input)
        st.markdown(f"**🧠 Answer:** {result['result'].strip()}")

        st.markdown("### 🔍 Sources used:")
        for doc in result["source_documents"]:
            meta = doc.metadata
            manual = meta.get("product_name", "Unknown")
            model = meta.get("model", "-")
            page = meta.get("page_number", "Unknown")
            st.markdown(f"- **{manual}** (model {model}), page {page}")
