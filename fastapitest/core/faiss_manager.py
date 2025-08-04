import os
import json
import hashlib
import logging
import boto3
import faiss
import pickle
import time
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_CACHE_DIR = "article_faiss_cache"
S3_BUCKET_NAME = "factseeker-faiss-db"
S3_PREFIX = "article_faiss_cache/"
os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)

s3 = boto3.client("s3")

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def download_from_s3_if_exists(s3_key: str, local_path: str) -> bool:
    try:
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logging.info(f"🔽 S3에서 다운로드 완료: {s3_key}")
        return True
    except Exception as e:
        logging.warning(f"⚠️ S3 다운로드 실패: {s3_key} → {e}")
        return False

def upload_to_s3(local_path: str, s3_key: str):
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logging.info(f"🆙 S3 업로드 완료: {s3_key}")
    except Exception as e:
        logging.error(f"❌ S3 업로드 실패: {s3_key} → {e}")

def get_or_build_faiss(url: str, article_text: str, embed_model) -> FAISS:
    hashed = sha256_of(url)
    folder_path = os.path.join(CHUNK_CACHE_DIR, hashed)
    os.makedirs(folder_path, exist_ok=True)
    
    faiss_path = os.path.join(folder_path, "index.faiss")
    pkl_path = os.path.join(folder_path, "index.pkl")

    s3_faiss_key = f"{S3_PREFIX}{hashed}/index.faiss"
    s3_pkl_key = f"{S3_PREFIX}{hashed}/index.pkl"

    # ✅ 로컬에 없으면 S3에서 다운로드 시도
    if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
        logging.info("📦 로컬 캐시 없음 → S3에서 로딩 시도")
        start = time.time()
        download_from_s3_if_exists(s3_faiss_key, faiss_path)
        download_from_s3_if_exists(s3_pkl_key, pkl_path)
        elapsed = time.time() - start
        logging.info(f"⏱️ [S3 다운로드] 소요 시간: {elapsed:.2f}초")

    # ✅ 다운로드되었거나 원래부터 로컬에 있으면 로드
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        logging.info("✅ FAISS 캐시 로드 완료")
        with open(pkl_path, "rb") as f:
            stored_texts = pickle.load(f)
        return FAISS.load_local(folder_path, embed_model, stored_texts)

    # ❌ 둘 다 없으면 새로 생성
    logging.info("⚙️ FAISS 인덱스 새로 생성 중...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(article_text)
    docs = [Document(page_content=chunk, metadata={"url": url}) for chunk in chunks]
    db = FAISS.from_documents(docs, embed_model)
    db.save_local(folder_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)

    # 업로드
    upload_to_s3(faiss_path, s3_faiss_key)
    upload_to_s3(pkl_path, s3_pkl_key)

    return db
