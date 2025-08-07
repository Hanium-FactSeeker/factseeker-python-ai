import os
import logging
import boto3
from botocore.exceptions import ClientError
from core.faiss_manager import S3_BUCKET_NAME, S3_PREFIX, CHUNK_CACHE_DIR

logging.basicConfig(level=logging.INFO)
s3 = boto3.client("s3")

def preload_faiss_from_existing_s3():
    """
    S3 bucket에서 모든 FAISS 파티션을 로컬 캐시 디렉토리로 미리 다운로드합니다.
    """
    if not os.path.exists(CHUNK_CACHE_DIR):
        os.makedirs(CHUNK_CACHE_DIR)

    logging.info(f"S3 버킷 '{S3_BUCKET_NAME}'의 '{S3_PREFIX}'에서 FAISS 파티션 다운로드를 시작합니다.")
    
    try:
        # S3_PREFIX 아래의 모든 '폴더'(파티션)를 찾습니다.
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX, Delimiter='/')
        
        partition_prefixes = []
        for page in pages:
            if "CommonPrefixes" in page:
                for obj in page.get('CommonPrefixes', []):
                    partition_prefixes.append(obj.get('Prefix'))

        if not partition_prefixes:
            logging.warning("다운로드할 FAISS 파티션을 찾을 수 없습니다.")
            return

        logging.info(f"{len(partition_prefixes)}개의 파티션을 다운로드합니다.")

        for s3_key_prefix in partition_prefixes:
            # S3 경로에서 로컬 디렉토리 이름을 추출합니다 (예: 'partition_0/')
            partition_name = os.path.basename(os.path.normpath(s3_key_prefix))
            local_partition_dir = os.path.join(CHUNK_CACHE_DIR, partition_name)

            if os.path.exists(local_partition_dir):
                logging.info(f"'{local_partition_dir}'은 이미 존재하므로 건너뜁니다.")
                continue

            os.makedirs(local_partition_dir, exist_ok=True)
            
            # 해당 파티션 폴더 내의 모든 파일을 다운로드합니다.
            response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key_prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_file_key = obj['Key']
                    local_file_path = os.path.join(local_partition_dir, os.path.basename(s3_file_key))
                    try:
                        logging.info(f"다운로드 중: {s3_file_key} -> {local_file_path}")
                        s3.download_file(S3_BUCKET_NAME, s3_file_key, local_file_path)
                    except ClientError as e:
                        logging.error(f"'{s3_file_key}' 파일 다운로드 실패: {e}")

    except Exception as e:
        logging.error(f"FAISS 파티션 프리로딩 중 오류 발생: {e}", exc_info=True)

