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
from core.preload_s3_faiss import preload_faiss_from_existing_s3



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

    # 본문 임베딩 캐시 프리로드
    await preload_faiss_from_existing_s3("article_faiss_cache/")

    # (선택 사항) 파티션 전체를 순회하면서 로딩
    for i in range(10):
        prefix = f"feature_faiss_db_openai_partition/partition_{i}/"
        await preload_faiss_from_existing_s3(prefix)

    logging.info("✅ 전체 FAISS 프리로드 완료")

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
