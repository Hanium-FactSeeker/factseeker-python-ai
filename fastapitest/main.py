import os
import asyncio
import logging
from shutil import rmtree
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.faiss_manager import CHUNK_CACHE_DIR
from core.preload_s3_faiss import preload_faiss_from_existing_s3
from services.fact_checker import run_fact_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(
    title="YouTube Fact-Checker API",
    description="유튜브 영상의 주장을 팩트체크하는 API",
    version="1.0.0"
)

def clean_local_cache_dir():
    logging.info(f"🧹 로컬 캐시 디렉토리 ({CHUNK_CACHE_DIR}) 정리 시작")
    if os.path.exists(CHUNK_CACHE_DIR):
        try:
            rmtree(CHUNK_CACHE_DIR)
            logging.info("✅ 캐시 디렉토리 정리 완료")
        except OSError as e:
            logging.error(f"❌ 캐시 디렉토리 삭제 실패: {e}")

# 전역 FAISS 파티션 경로 리스트 (폴더별로 저장!)
FAISS_PARTITION_DIRS = []

@app.on_event("startup")
async def startup_event():
    logging.info("--- FastAPI 애플리케이션 시작 ---")
    clean_local_cache_dir()
    await preload_faiss_from_existing_s3("article_faiss_cache/")
    # 파티션별 전체 프리로드
    for i in range(10):
        prefix = f"feature_faiss_db_openai_partition/partition_{i}/"
        await preload_faiss_from_existing_s3(prefix)
    # 파티션 경로 자동 수집 (폴더명만 저장, 실제 partition_0~9만!)
    global FAISS_PARTITION_DIRS
    FAISS_PARTITION_DIRS = []
    for i in range(10):
        part_dir = os.path.join(CHUNK_CACHE_DIR, f"partition_{i}")
        faiss_path = os.path.join(part_dir, "index.faiss")
        pkl_path = os.path.join(part_dir, "index.pkl")
        logging.info(f"[DEBUG] 체크: {faiss_path} / {os.path.exists(faiss_path)}")
        logging.info(f"[DEBUG] 체크: {pkl_path} / {os.path.exists(pkl_path)}")
        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            FAISS_PARTITION_DIRS.append(part_dir)
    logging.info(f"✅ 전체 FAISS 파티션 로드 경로: {FAISS_PARTITION_DIRS}")

class FactCheckRequest(BaseModel):
    youtube_url: str

@app.post("/fact-check")
async def perform_fact_check(request: FactCheckRequest):
    logging.info(f"--- 팩트체크 요청 수신: {request.youtube_url} ---")
    try:
        result = await run_fact_check(request.youtube_url, FAISS_PARTITION_DIRS)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logging.exception(f"API 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"팩트체크 처리 중 내부 서버 오류가 발생했습니다: {e}")
