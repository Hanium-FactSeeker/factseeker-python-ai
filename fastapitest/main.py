import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨
# --- 경로 문제 해결 (가장 중요한 부분) ---
# 이 파일(main.py)의 상위 폴더(factseeker-python-ai)를
# 파이썬이 모듈을 검색하는 경로에 추가합니다.
# 이렇게 하면 'core' 폴더를 찾을 수 있게 됩니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨ ✨✨✨


# 이제 파이썬이 core 폴더를 찾을 수 있으므로, 이 import가 성공합니다.
from core.fact_checker import run_fact_check

# --- FastAPI 앱 초기화 ---
app = FastAPI()

class FactCheckRequest(BaseModel):
    youtube_url: str

@app.post("/fact-check")
async def perform_fact_check(request: FactCheckRequest):
    """
    유튜브 URL을 받아 팩트체크를 수행하는 API 엔드포인트
    """
    try:
        logging.info(f"요청 수신: {request.youtube_url}")

        # 이전에 수정한 대로, 더 이상 필요 없는 인자 없이 호출합니다.
        result = await run_fact_check(request.youtube_url)

        if "error" in result:
            logging.error(f"팩트체크 처리 중 오류 발생: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])

        logging.info(f"팩트체크 성공적으로 완료: {request.youtube_url}")
        return JSONResponse(content=result)

    except HTTPException as e:
        # HTTPException은 그대로 다시 발생시킵니다.
        raise e
    except Exception as e:
        # 그 외 모든 예외는 서버 오류로 처리합니다.
        logging.exception(f"예상치 못한 오류 발생: {request.youtube_url} - {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- 서버 실행 ---
if __name__ == "__main__":
    # 개발 환경에서 직접 실행할 때 사용
    # 예: python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)