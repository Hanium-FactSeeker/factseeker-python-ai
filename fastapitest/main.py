import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# core 폴더의 필요한 함수들을 가져옵니다.
from core.fact_checker import run_fact_check
from core.preload_s3_faiss import preload_faiss_from_existing_s3, CHUNK_CACHE_DIR

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- 데이터 모델 정의 ---
class FactCheckRequest(BaseModel):
    youtube_url: str

# --- FAISS 데이터 로드를 위한 변수 ---
faiss_partition_dirs = []

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행될 코드
    logging.info("애플리케이션 시작...")
    
    # 제목 FAISS가 저장된 정확한 S3 경로를 지정합니다.
    TITLE_FAISS_S3_PREFIX = "feature_faiss_db_openai_partition/"
    
    # ✨✨✨ 수정된 부분: 'await'와 인수를 모두 제거합니다. ✨✨✨
    preload_faiss_from_existing_s3(TITLE_FAISS_S3_PREFIX)
    
    # 로컬 캐시 폴더에서 로드된 파티션 디렉토리 목록을 전역 변수에 저장합니다.
    if os.path.exists(CHUNK_CACHE_DIR):
        faiss_partition_dirs.clear()
        for item in os.listdir(CHUNK_CACHE_DIR):
            item_path = os.path.join(CHUNK_CACHE_DIR, item)
            if os.path.isdir(item_path) and item.startswith("partition_"):
                faiss_partition_dirs.append(item_path)
    
    if faiss_partition_dirs:
        logging.info(f"✅ {len(faiss_partition_dirs)}개의 제목 FAISS 파티션을 성공적으로 로드했습니다.")
    else:
        logging.warning("⚠️ 로드된 제목 FAISS 파티션이 없습니다. 제목 기반 검색이 작동하지 않을 수 있습니다.")
        
    yield
    # 서버 종료 시 실행될 코드
    logging.info("애플리케이션 종료...")

# --- FastAPI 앱 생성 ---
app = FastAPI(lifespan=lifespan)

# --- API 엔드포인트 ---
@app.post("/fact-check")
async def fact_check_endpoint(request: FactCheckRequest):
    """
    유튜브 URL을 받아 팩트체크를 수행하고 결과를 반환합니다.
    """
    if not faiss_partition_dirs:
        raise HTTPException(
            status_code=503, 
            detail="서버가 아직 준비되지 않았습니다. FAISS 파티션이 로드되지 않았습니다."
        )
        
    try:
        result = await run_fact_check(
            youtube_url=request.youtube_url,
            faiss_partition_dirs=faiss_partition_dirs
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logging.error(f"팩트체크 처리 중 심각한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.")

@app.get("/")
def read_root():
    return {"message": "FactSeeker AI 서버가 실행 중입니다."}