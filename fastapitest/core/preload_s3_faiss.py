import os
import time
import logging
import boto3
from core.faiss_manager import download_from_s3_if_exists, CHUNK_CACHE_DIR

# S3 설정
S3_BUCKET_NAME = "factseeker-faiss-db"
S3_PREFIX = "article_faiss_cache/"
s3 = boto3.client("s3")

def list_faiss_keys_from_s3():
    """S3에서 .faiss 파일 키만 리스트로 반환"""
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".faiss"):
                keys.append(obj["Key"])
    return keys


async def preload_faiss_from_existing_s3():
    """S3에 이미 존재하는 인덱스들만 로컬로 미리 다운로드"""
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    logging.info("🚀 S3에 존재하는 FAISS 인덱스 프리로드 시작")

    faiss_keys = list_faiss_keys_from_s3()
    for faiss_key in faiss_keys:
        base = os.path.splitext(os.path.basename(faiss_key))[0]
        pkl_key = f"{S3_PREFIX}{base}.pkl"

        faiss_path = os.path.join(CHUNK_CACHE_DIR, f"{base}.faiss")
        pkl_path = os.path.join(CHUNK_CACHE_DIR, f"{base}.pkl")

        start = time.time()
        faiss_ok = download_from_s3_if_exists(faiss_key, faiss_path)
        pkl_ok = download_from_s3_if_exists(pkl_key, pkl_path)
        elapsed = time.time() - start

        if faiss_ok and pkl_ok:
            logging.info(f"✅ 프리로드 완료: {base} ⏱️ {elapsed:.2f}초")
        else:
            logging.warning(f"⚠️ 프리로드 실패: {base}")