import os
import time
import logging
import boto3
from core.faiss_manager import download_from_s3_if_exists, CHUNK_CACHE_DIR

# S3 설정
S3_BUCKET_NAME = "factseeker-faiss-db"
s3 = boto3.client("s3")

def list_faiss_keys_from_s3(s3_prefix):
    """지정된 prefix에서 .faiss 파일 키만 리스트로 반환"""
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix)
    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".faiss"):
                keys.append(obj["Key"])
    return keys

def map_query_to_partition(query_vector, num_partitions=10):
    """임베딩 벡터 기반 파티션 결정 (단순 해시 기반)"""
    return int(sum(query_vector) * 1000) % num_partitions

def get_partition_prefix_from_query(query_vector):
    partition_num = map_query_to_partition(query_vector)
    return f"feature_faiss_db_openai_partition/partition_{partition_num}/"

async def preload_faiss_from_existing_s3(s3_prefix):
    """지정된 S3 prefix 하위에 존재하는 인덱스들만 로컬로 다운로드"""
    if s3_prefix.startswith("article_faiss_cache"):
        logging.info("🛑 본문 인덱스는 프리로드하지 않음")
        return

    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    logging.info(f"🚀 S3 FAISS 인덱스 프리로드 시작 (prefix={s3_prefix})")

    faiss_keys = list_faiss_keys_from_s3(s3_prefix)
    logging.info(f"🔢 프리로드 대상 인덱스 개수: {len(faiss_keys)}개")

    for faiss_key in faiss_keys:
        if not faiss_key.endswith(".faiss"):
            continue

        # 경로에서 폴더 이름 추출 (해시값 or partition 이름)
        dir_name = os.path.dirname(faiss_key).replace(s3_prefix, "").strip("/")
        pkl_key = f"{s3_prefix}{dir_name}/index.pkl"

        faiss_path = os.path.join(CHUNK_CACHE_DIR, f"{dir_name}.faiss")
        pkl_path = os.path.join(CHUNK_CACHE_DIR, f"{dir_name}.pkl")

        start = time.time()
        faiss_ok = download_from_s3_if_exists(faiss_key, faiss_path)
        pkl_ok = download_from_s3_if_exists(pkl_key, pkl_path)
        elapsed = time.time() - start

        if faiss_ok and pkl_ok:
            logging.info(f"✅ 프리로드 완료: {dir_name} ⏱️ {elapsed:.2f}초")
        else:
            logging.warning(f"⚠️ 프리로드 실패: {dir_name}")