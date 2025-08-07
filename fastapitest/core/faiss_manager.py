import os
import shutil
import hashlib
import logging
import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 설정 ---
CHUNK_CACHE_DIR = "article_faiss_cache"
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
S3_PREFIX = "article_faiss_cache/"
os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)

try:
    s3 = boto3.client("s3")
except Exception as e:
    s3 = None
    logging.critical(f"S3 클라이언트 초기화 실패! S3 기능을 사용할 수 없습니다. 에러: {e}")

# --- 내부 헬퍼 함수 ---
def _normalize_url(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.replace('www.', '')
    params = parse_qs(parsed.query)
    filtered_params = {k: v for k, v in params.items() if not k.startswith('utm_') and k != 'fbclid'}
    query = urlencode(filtered_params, doseq=True)
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, parsed.fragment)).rstrip('/')

def _url_to_cache_key(url):
    return hashlib.md5(_normalize_url(url).encode()).hexdigest()

def _download_from_s3(local_dir_path, s3_key_prefix):
    if not s3: return False
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key_prefix)
        if 'Contents' not in response: return False
        os.makedirs(local_dir_path, exist_ok=True)
        for obj in response['Contents']:
            s3_path = obj['Key']
            local_file_path = os.path.join(local_dir_path, os.path.basename(s3_path))
            s3.download_file(S3_BUCKET_NAME, s3_path, local_file_path)
        logging.info(f"✅ S3 캐시 다운로드 성공: {s3_key_prefix}")
        return True
    except ClientError as e:
        logging.error(f"S3 다운로드 실패: s3://{S3_BUCKET_NAME}/{s3_key_prefix} - {e}")
        return False

def _upload_to_s3(local_dir_path: str, s3_key_prefix: str):
    if not s3: return
    try:
        for file_name in os.listdir(local_dir_path):
            local_file_path = os.path.join(local_dir_path, file_name)
            if os.path.isfile(local_file_path):
                s3_key = os.path.join(s3_key_prefix, file_name)
                s3.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        logging.info(f"✅ S3 업로드 성공: {s3_key_prefix}")
    except Exception as e:
        logging.error(f"❌ S3 업로드 실패: {s3_key_prefix} -> {e}")

# --- 외부 호출 함수 ---
def get_or_build_faiss(url: str, article_text: str, embed_model) -> FAISS | None:
    """기사 URL을 기준으로 FAISS 인덱스를 생성하고 S3에 캐싱합니다."""
    if not article_text or len(article_text.strip()) < 200:
        logging.warning(f"기사 내용이 너무 짧아 FAISS 인덱스를 생성하지 않습니다: {url}")
        return None

    cache_key = _url_to_cache_key(url)
    folder_path = os.path.join(CHUNK_CACHE_DIR, cache_key)
    s3_key_prefix = f"{S3_PREFIX}{cache_key}/"

    logging.info(f"⚙️ FAISS 인덱스 신규 생성 시도: {url}")
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = [Document(page_content=article_text, metadata={"url": url})]
        chunks = splitter.split_documents(docs)
        
        db = FAISS.from_documents(chunks, embed_model)
        os.makedirs(folder_path, exist_ok=True)
        db.save_local(folder_path)
        
        _upload_to_s3(folder_path, s3_key_prefix)
        return db
    except Exception as e:
        logging.error(f"❌ FAISS 인덱스 생성 및 저장 실패: {url} - {e}")
        return None

def load_faiss_from_cache(url: str, embed_model) -> FAISS | None:
    """URL을 기준으로 로컬/S3 캐시에서 FAISS 인덱스를 로드합니다."""
    cache_key = _url_to_cache_key(url)
    folder_path = os.path.join(CHUNK_CACHE_DIR, cache_key)
    s3_key_prefix = f"{S3_PREFIX}{cache_key}/"

    # 1. 로컬 캐시 확인
    if os.path.exists(folder_path):
        try:
            logging.info(f"로컬 캐시에서 로드: {folder_path}")
            return FAISS.load_local(folder_path, embed_model, allow_dangerous_deserialization=True)
        except Exception as e:
            shutil.rmtree(folder_path, ignore_errors=True)
            logging.warning(f"로컬 캐시 손상, S3 확인 시도: {e}")

    # 2. S3 캐시 확인
    if _download_from_s3(folder_path, s3_key_prefix):
        try:
            logging.info(f"S3 캐시에서 로드: {s3_key_prefix}")
            return FAISS.load_local(folder_path, embed_model, allow_dangerous_deserialization=True)
        except Exception as e:
            shutil.rmtree(folder_path, ignore_errors=True)
            logging.warning(f"S3 캐시 손상, 캐시 없음으로 처리: {e}")
    
    return None