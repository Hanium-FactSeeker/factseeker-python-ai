# main.py
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import asyncio

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 경로 설정 ---
# 프로젝트의 루트 디렉토리를 sys.path에 추가합니다.
# 이 파일(main.py)이 있는 위치를 기준으로 core 폴더를 찾을 수 있도록 설정합니다.
# 예: /home/ubuntu/factseeker-python-ai/fastapitest/main.py -> /home/ubuntu/factseeker-python-ai
# 실제 환경에 맞게 경로를 조정해야 할 수 있습니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fact_checker import run_fact_check

# --- FastAPI 앱 초기화 ---
app = FastAPI()

class FactCheckRequest(BaseModel):
    youtube_url: str

# --- 기존 코드에서 FAISS_PARTITION_DIRS는 더 이상 필요 없으므로 제거하거나 주석 처리합니다. ---
# # FAISS 파티션 디렉토리 (뉴스 데이터가 저장된 곳)
# FAISS_PARTITION_BASE_DIR = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "news_faiss_db_partitions"
# )
# FAISS_PARTITION_DIRS = [
#     os.path.join(FAISS_PARTITION_BASE_DIR, d)
#     for d in os.listdir(FAISS_PARTITION_BASE_DIR)
#     if os.path.isdir(os.path.join(FAISS_PARTITION_BASE_DIR, d))
# ]
# logging.info(f"FAISS 파티션 디렉토리 로드 완료: {len(FAISS_PARTITION_DIRS)}개")


@app.post("/fact-check")
async def perform_fact_check(request: FactCheckRequest):
    """
    유튜브 URL을 받아 팩트체크를 수행하는 API 엔드포인트
    """
    try:
        logging.info(f"요청 수신: {request.youtube_url}")
        
        # ✨✨✨ 수정된 부분 ✨✨✨
        # run_fact_check 함수에서 FAISS_PARTITION_DIRS 인자를 제거합니다.
        result = await run_fact_check(request.youtube_url)
        
        if "error" in result:
            logging.error(f"팩트체크 처리 중 오류 발생: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        logging.info(f"팩트체크 성공적으로 완료: {request.youtube_url}")
        return JSONResponse(content=result)

    except HTTPException as e:
        # HTTPException은 그대로 전달
        raise e
    except Exception as e:
        # 그 외 모든 예외 처리
        logging.exception(f"예상치 못한 오류 발생: {request.youtube_url} - {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- 서버 실행 ---
if __name__ == "__main__":
    # 개발 환경에서 직접 실행할 때 사용
    # 예: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)