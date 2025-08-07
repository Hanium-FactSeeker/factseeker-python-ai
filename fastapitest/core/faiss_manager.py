import os
import hashlib
import logging
import boto3
import pickle
import time
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# lambdas에서 get_article_text 함수를 가져옵니다.
# 실제 환경에서는 의존성 주입 등을 고려할 수 있습니다.
from core.lambdas import get_article_text

CHUNK_CACHE_DIR = "article_faiss_cache"
S3_BUCKET_NAME = "factseeker-faiss-db"
S3_PREFIX = "article_faiss_cache/"
os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)

s3 = boto3.client("s3")

def sha256_of(text: str) -> str:
    """텍스트의 SHA256 해시를 계산합니다."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def download_from_s3_if_exists(s3_key, local_path):
    """S3에서 파일이 존재하면 다운로드합니다."""
    try:
        s3.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        logging.info(f"📦 S3에서 캐시 파일 확인: {s3_key}")
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        if os.path.exists(local_path):
            logging.info(f"✅ S3 캐시 다운로드 성공: {local_path} ({os.path.getsize(local_path)} bytes)")
            return True
    except Exception:
        logging.info(f"💨 S3에 캐시 파일 없음: {s3_key}")
        return False
    return False

def upload_to_s3(local_path: str, s3_key: str):
    """파일을 S3에 업로드합니다."""
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_key)
        logging.info(f"🆙 S3 업로드 완료: {s3_key}")
    except Exception as e:
        logging.error(f"❌ S3 업로드 실패: {s3_key} -> {e}")

async def get_documents_with_caching(url: str, embed_model) -> list[Document]:
    """
    URL을 기반으로 문서를 가져옵니다.
    캐시(로컬/S3)를 확인하고, 없으면 문서를 크롤링, 처리 후 캐시에 저장합니다.
    """
    hashed = sha256_of(url)
    folder_path = os.path.join(CHUNK_CACHE_DIR, hashed)
    os.makedirs(folder_path, exist_ok=True)

    faiss_path = os.path.join(folder_path, "index.faiss")
    pkl_path = os.path.join(folder_path, "index.pkl")

    # 1. 로컬 캐시 확인
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        logging.info(f"✅ 로컬 캐시에서 문서 로드: {url}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # 2. S3 캐시 확인
    s3_faiss_key = f"{S3_PREFIX}{hashed}/index.faiss"
    s3_pkl_key = f"{S3_PREFIX}{hashed}/index.pkl"

    if download_from_s3_if_exists(s3_pkl_key, pkl_path) and download_from_s3_if_exists(s3_faiss_key, faiss_path):
        logging.info(f"✅ S3 캐시에서 문서 로드: {url}")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # 3. 캐시가 없으면 크롤링 및 신규 생성
    logging.info(f"⚙️ 캐시 없음. 기사 크롤링 및 FAISS 인덱스 신규 생성 시작: {url}")
    
    article_text = await get_article_text(url)
    if not article_text or len(article_text) < 200:
        logging.warning(f"크롤링 실패 또는 내용이 너무 짧아 처리를 중단합니다: {url}")
        return []

    # 텍스트를 청크로 분할하여 Document 생성
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(article_text)
    docs = [Document(page_content=chunk, metadata={"url": url, "original_title": ""}) for chunk in chunks]
    
    if not docs:
        logging.warning(f"문서 청크 생성 실패: {url}")
        return []

    # FAISS 인덱스 생성 및 저장
    try:
        db = FAISS.from_documents(docs, embed_model)
        db.save_local(folder_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(docs, f)
        logging.info(f"💾 로컬에 캐시 저장 완료: {folder_path}")

        # S3에 업로드
        upload_to_s3(faiss_path, s3_faiss_key)
        upload_to_s3(pkl_path, s3_pkl_key)

        return docs
    except Exception as e:
        logging.error(f"FAISS 인덱스 생성 또는 저장 실패: {url}, 에러: {e}")
        return []