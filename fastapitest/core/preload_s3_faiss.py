import os
import time
import logging
import boto3
from botocore.exceptions import ClientError

# --- 설정 ---
# CHUNK_CACHE_DIR는 main.py에서도 사용하므로 여기서 내보냅니다.
CHUNK_CACHE_DIR = "article_faiss_cache" 
S3_BUCKET_NAME = "factseeker-faiss-db"
s3 = boto3.client("s3")
# --- 설정 끝 ---

def _download_s3_file(s3_key, local_path):
    """S3에서 단일 파일을 다운로드하는 내부 헬퍼 함수"""
    try:
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.warning(f"S3에 파일 없음: {s3_key}")
        else:
            logging.error(f"S3 다운로드 실패: {s3_key} -> {local_path} / error: {e}")
        return False

def _list_faiss_keys_from_s3(s3_prefix):
    """지정된 prefix에서 .faiss 파일 키만 리스트로 반환"""
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=s3_prefix)
    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".faiss"):
                keys.append(obj["Key"])
    return keys

def preload_faiss_from_existing_s3(s3_prefix):
    """
    지정된 S3 prefix 하위의 모든 FAISS 파티션을 로컬 캐시 디렉토리로 미리 다운로드합니다.
    """
    # 본문 캐시는 URL 해시 기반이라 프리로드 대상이 아니므로 건너뜁니다.
    if "article_faiss_cache" in s3_prefix:
        logging.info("본문 인덱스(article_faiss_cache)는 프리로드하지 않습니다.")
        return

    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    logging.info(f"🚀 S3 FAISS 인덱스 프리로드 시작 (prefix={s3_prefix})")

    faiss_keys = _list_faiss_keys_from_s3(s3_prefix)
    logging.info(f"🔢 프리로드 대상 FAISS 파티션 개수: {len(faiss_keys)}개")

    for faiss_key in faiss_keys:
        # S3 키 예시: 'feature_faiss_db_openai_partition/partition_0/index.faiss'
        # dir_name은 'partition_0'과 같은 파티션 폴더 이름이 됩니다.
        dir_name = os.path.basename(os.path.dirname(faiss_key))
        pkl_key = os.path.join(os.path.dirname(faiss_key), "index.pkl")
        
        local_dir = os.path.join(CHUNK_CACHE_DIR, dir_name)
        
        # 이미 로컬에 있으면 건너뜁니다.
        if os.path.exists(local_dir):
            logging.info(f"이미 존재함, 건너뛰기: {local_dir}")
            continue

        os.makedirs(local_dir, exist_ok=True)
        faiss_path = os.path.join(local_dir, "index.faiss")
        pkl_path = os.path.join(local_dir, "index.pkl")

        start = time.time()
        faiss_ok = _download_s3_file(faiss_key, faiss_path)
        pkl_ok = _download_s3_file(pkl_key, pkl_path)
        elapsed = time.time() - start

        if faiss_ok and pkl_ok:
            logging.info(f"✅ 프리로드 완료: {dir_name} ⏱️ {elapsed:.2f}초")
        else:
            logging.warning(f"⚠️ 프리로드 실패: {dir_name}")

# 이 파일을 직접 실행할 경우를 위한 테스트 코드
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 실제 제목 FAISS 파티션이 저장된 S3 경로를 지정합니다.
    TITLE_FAISS_S3_PREFIX = "feature_faiss_db_openai_partition/"
    preload_faiss_from_existing_s3(TITLE_FAISS_S3_PREFIX)