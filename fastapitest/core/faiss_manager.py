import os
import hashlib
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# FAISS 벡터 DB 및 캐시 경로
# 로컬 테스트용 경로, 배포 시 S3 경로로 변경 예정
FAISS_DB_PATH   = os.getenv("LOCAL_FAISS_DB_PATH", "feature_faiss_db_openai")
CHUNK_CACHE_DIR = os.getenv("LOCAL_CHUNK_CACHE_DIR", "article_faiss_cache")

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """텍스트를 지정된 크기로 분할합니다."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

async def get_or_build_faiss(url, article_text=None, embed_model=None, cache_dir=CHUNK_CACHE_DIR):     
    """
    URL 기반으로 FAISS DB를 로드하거나 새로 구축하여 저장하고,
    FAISS DB가 로드될 경우 해당 기사의 원본 텍스트를 반환합니다.
    """
    idx = hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(cache_dir, idx)
    
    if os.path.isdir(path):
        logging.info(f"FAISS DB 캐시 로드: {url}")
        db = FAISS.load_local(path, embed_model, allow_dangerous_deserialization=True)
        
        all_docs = list(db.docstore._dict.values())
        retrieved_text = "\n".join([doc.page_content for doc in all_docs])
                
        logging.info(f"캐시에서 본문 텍스트 복원 완료 (길이: {len(retrieved_text)}자).")
        return db, retrieved_text # DB와 복원된 텍스트 반환
    
    if article_text is None:
        raise ValueError("FAISS DB 캐시가 없으면 article_text가 반드시 제공되어야 합니다.")

    os.makedirs(path, exist_ok=True)
    chunks = split_text(article_text)
    docs = [Document(page_content=c, metadata={"url": url}) for c in chunks]
    db = FAISS.from_documents(docs, embed_model)
    db.save_local(path)
    logging.info(f"FAISS DB 캐시 생성 및 저장: {url}")
    return db, article_text # 새로 생성된 DB와 원본 텍스트 반환