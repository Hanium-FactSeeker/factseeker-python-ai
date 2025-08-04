import os
import asyncio
import json
import faiss
from langchain.docstore.document import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 서비스 로직 임포트
from services.fact_checker import run_fact_check
from core.faiss_manager import CHUNK_CACHE_DIR
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from core.lambdas import clean_news_title, search_news_google_cs # 필요한 유틸리티 추가
import logging
from core.faiss_manager import get_or_build_faiss
import time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(
    title="YouTube Fact-Checker API",
    description="유튜브 영상의 주장을 팩트체크하는 API",
    version="1.0.0"
)

embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    request_timeout=60,
    max_retries=5,
    chunk_size=500
)

# FAISS DB 및 캐시 디렉토리 생성 (서버 시작 시)
@app.on_event("startup")
async def startup_event():
    
    logging.info("--- FastAPI 애플리케이션 시작 ---")
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)

    PRELOAD_URLS = [
    "https://www.chosun.com/politics/2025/08/01/article1",
    "https://www.khan.co.kr/article/202507111106001",
    # 추가 가능
    ]

    for url in PRELOAD_URLS:
        try:
            dummy_text = "프리로드용 텍스트"  # 실제 사용되지 않음 (S3에만 있으면 됨)
            get_or_build_faiss(url, dummy_text, embed_model)
        except Exception as e:
            logging.warning(f"⚠️ 프리로드 실패: {url} → {e}")
    logging.info("✅ FAISS 프리로드 완료")

class FactCheckRequest(BaseModel):
    youtube_url: str

class FactCheckResponse(BaseModel):
    video_id: str
    video_total_confidence_score: int
    claims: list


@app.post("/fact-check")
async def perform_fact_check(request: FactCheckRequest):
    """
    제공된 유튜브 URL에 대해 팩트체크를 수행합니다.
    """
    logging.info(f"--- 팩트체크 요청 수신: {request.youtube_url} ---")
    try:
        logging.info("팩트체크 시작...")
        result = await run_fact_check(request.youtube_url)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result   # dict 전체 그대로 반환!
    except Exception as e:
        logging.exception(f"API 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"팩트체크 처리 중 내부 서버 오류가 발생했습니다: {e}")
