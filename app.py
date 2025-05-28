import os
import boto3

if "streamlit" in os.getenv("ASK_MODE", ""):
    download_vector_store_from_s3()
    
def download_vector_store_from_s3():
    """
    Downloads index.faiss and index.pkl from S3 to /tmp/vector_store.
    Expects the following environment variables or Streamlit secrets to be set:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_DEFAULT_REGION
      - VECTOR_STORE_BUCKET
      - VECTOR_STORE_PREFIX
    """
    bucket = os.getenv("VECTOR_STORE_BUCKET")
    prefix = os.getenv("VECTOR_STORE_PREFIX", "vector_store")
    if not bucket or not prefix:
        raise ValueError("VECTOR_STORE_BUCKET and VECTOR_STORE_PREFIX must be set as environment variables or secrets.")

    s3 = boto3.client("s3")
    local_dir = "/tmp/vector_store"
    os.makedirs(local_dir, exist_ok=True)
    expected_files = ["index.faiss", "index.pkl"]

    for fname in expected_files:
        key = f"{prefix}/{fname}"
        local_path = os.path.join(local_dir, fname)
        try:
            s3.download_file(bucket, key, local_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading {fname} from s3://{bucket}/{key}: {e}")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"File {local_path} not found after download.")
