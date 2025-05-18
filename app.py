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

# --- Streamlit config ---
st.set_page_config(page_title="Ask My Manuals", page_icon="📘")

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

# --- Download vector store from S3 ---
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

    return local_path

# --- Load components ---
@st.cache_resource
def load_components():
    vector_store_path = download_vector_store_from_s3()
    vector_store = FAISS.load_local(
        str(vector_store_path),
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )

    # Extract searchable metadata
    search_terms_to_products = {}
    for doc in vector_store.docstore._dict.values():
        meta = doc.metadata
        product_name = meta.get("product_name", "").lower()
        brand = meta.get("brand", "").lower()
        model = meta.get("model", "").lower()

        if product_name:
            search_terms_to_products[product_name] = product_name
        if brand:
            search_terms_to_products[brand] = product_name
        if model:
            search_terms_to_products[model] = product_name

    # Load LLM
    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=generator)

    return vector_store, llm, search_terms_to_products

# --- Product detection ---
def detect_product_name_dynamic(user_input: str, search_terms_map: dict) -> str:
    lowered = user_input.lower()
    for keyword, product in search_terms_map.items():
        if keyword in lowered:
            return product
    return None

# --- App logic ---
vector_store, llm, search_terms_to_products = load_components()

st.title("📘 Ask My Manuals")
st.write("Ask a question about your appliances and devices.")

user_input = st.text_input("Your question:")

if user_input:
    with st.spinner("Thinking..."):
        product = detect_product_name_dynamic(user_input, search_terms_to_products)

        if product:
            filtered_retriever = vector_store.as_retriever(
                search_kwargs={"k": 5, "filter": {"product_name": product}}
            )
        else:
            filtered_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=filtered_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        st.write("Detected product:", product)

        # Quick check: how many chunks exist for that product?
        matches = [
            doc for doc in vector_store.docstore._dict.values()
            if product and doc.metadata.get("product_name", "").lower() == product.lower()
            #if doc.metadata.get("product_name", "").lower() == product.lower()
        ]
        st.write(f"Chunks available for '{product}':", len(matches))

        result = qa_chain.invoke(user_input)

        st.markdown(f"**🧠 Answer:** {result['result'].strip()}")
        st.markdown("### 🔍 Sources used:")
        for doc in result["source_documents"]:
            meta = doc.metadata
            manual = meta.get("product_name", "Unknown")
            model = meta.get("model", "-")
            page = meta.get("page_number", "Unknown")
            st.markdown(f"- **{manual}** (model {model}), page {page}")
