import argparse
import json
import os
import re
import time
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from sentencex import segment
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Configurable paths
DATA_DIR = "../data"
METADATA_PATH = "../metadata/manuals.json"
VECTOR_STORE_DIR = "../vector_store"
SENTENCES_PER_CHUNK = 5
BATCH_SIZE = 4  # Lower if you still get OOM

def extract_text_from_pdf(pdf_path):
    elements = partition_pdf(filename=pdf_path)
    text = "\n".join([el.text for el in elements if el.text])
    return text

def clean_text(text):
    text = re.sub(r'-\n', '', text)  # Remove hyphenation at line ends
    text = re.sub(r'\n+', '\n', text)  # Collapse multiple newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text

def sentencex_chunk_text(text, sentences_per_chunk=SENTENCES_PER_CHUNK):
    sentences = list(segment("en", text))
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk_text = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk_text)
    return chunks

def process_manual(pdf_path, sentences_per_chunk=SENTENCES_PER_CHUNK):
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    chunks = sentencex_chunk_text(cleaned_text, sentences_per_chunk)
    documents = []
    for idx, chunk in enumerate(chunks):
        metadata = {
            "file_path": pdf_path,
            "chunk_id": idx
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    return documents

def get_model_and_preprocessing(model_name):
    # Map short names to full model names
    model_map = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "e5": "intfloat/e5-large-v2",
        "bge": "BAAI/bge-base-en-v1.5",
        "jina": "jinaai/jina-embeddings-v2-base-en"
    }
    model_full_name = model_map[model_name]
    # Some models need a prefix for retrieval tasks
    if model_name in ["e5", "bge"]:
        preprocess = lambda x: f"passage: {x}"
    else:
        preprocess = lambda x: x
    model = SentenceTransformer(model_full_name, trust_remote_code=True)
    return model, preprocess

def main():
    start_time = time.time()
    print(f"Script started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    parser = argparse.ArgumentParser(description="ETL script for embedding manuals with selectable embedding model.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["minilm", "e5", "bge", "jina"],
                        help="Embedding model to use.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for embedding.")
    args = parser.parse_args()

    print(f"📄 Starting chunking | Time elapsed: {time.time() - start_time:.2f}s")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        manual_metadata = json.load(f)

    all_documents = []
    for entry in manual_metadata:
        file_path = os.path.normpath(entry["file_path"])
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        print(f"📄 Processing {file_path} | Time elapsed: {time.time() - start_time:.2f}s")
        try:
            docs = process_manual(file_path, SENTENCES_PER_CHUNK)
            for doc in docs:
                doc.metadata["product_name"] = entry["product_name"].lower()
                doc.metadata["model"] = entry["model"].lower()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    print(f"Chunking complete. Total documents to embed: {len(all_documents)} | Time elapsed: {time.time() - start_time:.2f}s")
    print(f"Starting embedding preprocess with {args.model}")
    model, preprocess = get_model_and_preprocessing(args.model)
    texts = [preprocess(doc.page_content) for doc in all_documents]
    metadatas = [doc.metadata for doc in all_documents]

    print(f"Starting embedding with {args.model} | Time elapsed: {time.time() - start_time:.2f}s")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Dummy embedding class for FAISS
    class DummyEmbeddings(Embeddings):
        def embed_documents(self, docs):
            raise NotImplementedError("Precomputed embeddings used; not callable.")
        def embed_query(self, query):
            raise NotImplementedError("Precomputed embeddings used; not callable.")

    # Build FAISS vector store from precomputed embeddings
    text_embeddings = list(zip(texts, embeddings))
    faiss_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=DummyEmbeddings(),
        metadatas=metadatas
    )
    faiss_store.save_local(VECTOR_STORE_DIR)

    print(f"✅ Vector store saved to {VECTOR_STORE_DIR}")
    print(f"Time elapsed: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
