import os
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import contextlib
from dotenv import load_dotenv

# core 폴더의 필요한 함수들을 가져옵니다.
from services.fact_checker import run_fact_check
from core.preload_s3_faiss import preload_faiss_from_existing_s3, CHUNK_CACHE_DIR
from article_checker.router import create_router as create_article_router

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# --- 데이터 모델 정의 ---
class FactCheckRequest(BaseModel):
    youtube_url: str

# --- FAISS 데이터 로드를 위한 변수 ---
faiss_partition_dirs = []


def _kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def _compute_month_kst() -> str:
    return _kst_now().strftime("%Y%m")


def _refresh_faiss_partition_dirs():
    """Rescan local cache for partition_* folders and refresh the global list."""
    if not os.path.exists(CHUNK_CACHE_DIR):
        return
    faiss_partition_dirs.clear()

    def partition_num(name: str) -> int:
        try:
            base = os.path.basename(name)
            num = int(''.join(ch for ch in base if ch.isdigit()))
            return num
        except Exception:
            return -1

    items = [
        os.path.join(CHUNK_CACHE_DIR, item)
        for item in os.listdir(CHUNK_CACHE_DIR)
        if os.path.isdir(os.path.join(CHUNK_CACHE_DIR, item)) and item.startswith("partition_")
    ]
    for item_path in sorted(items, key=partition_num, reverse=True):
        faiss_partition_dirs.append(item_path)


def _remove_local_partition(prefix: str):
    import shutil
    part = os.path.basename(os.path.dirname(prefix.rstrip('/')))
    local_dir = os.path.join(CHUNK_CACHE_DIR, part)
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)
        logging.info(f"🧹 로컬 파티션 삭제: {local_dir}")


async def _watch_titles_preload_task(base_prefix: str, poll_interval_sec: float = 120.0, include_partition_10: bool = True):
    """Background watcher: monitors S3 title partitions and re-preloads on change.

    - Watches monthly partition (Asia/Seoul). Optionally also partition_10.
    - On index.faiss change (with index.pkl present), removes local partition and re-preloads just that prefix.
    - Refreshes global faiss_partition_dirs after each reload.
    """
    s3 = boto3.client("s3")
    bucket = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
    seen: dict[str, str] = {}

    def prefixes_now() -> list[str]:
        ym = _compute_month_kst()
        monthly = f"{base_prefix.rstrip('/')}/partition_{ym}/"
        res = [monthly]
        if include_partition_10:
            res.append(f"{base_prefix.rstrip('/')}/partition_10/")
        return res

    def head(key: str):
        try:
            return s3.head_object(Bucket=bucket, Key=key)
        except Exception:
            return None

    while True:
        try:
            for prefix in prefixes_now():
                faiss_key = f"{prefix}index.faiss"
                pkl_key = f"{prefix}index.pkl"
                faiss_head = head(faiss_key)
                pkl_head = head(pkl_key)
                if not faiss_head or not pkl_head:
                    continue
                tag = f"{faiss_head.get('ETag')}_{faiss_head.get('LastModified').timestamp()}"
                if seen.get(prefix) != tag:
                    logging.info(f"🔔 S3 변경 감지: {faiss_key} → 제목 프리로드 재실행")
                    _remove_local_partition(prefix)
                    # 강제 재다운로드를 통해 로컬 캐시가 남아 있어도 최신으로 교체
                    preload_faiss_from_existing_s3(prefix, force_reload=True)
                    _refresh_faiss_partition_dirs()
                    seen[prefix] = tag
        except Exception as e:
            logging.warning(f"제목 프리로드 워처 오류(계속 진행): {e}")

        await asyncio.sleep(poll_interval_sec)

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 실행될 코드
    logging.info("애플리케이션 시작...")
    
    # 제목 FAISS가 저장된 정확한 S3 경로를 지정합니다.
    TITLE_FAISS_S3_PREFIX = "feature_faiss_db_openai_partition/"

    # 최초 1회 프리로드
    preload_faiss_from_existing_s3(TITLE_FAISS_S3_PREFIX)
    _refresh_faiss_partition_dirs()
    
    if faiss_partition_dirs:
        logging.info(f"✅ {len(faiss_partition_dirs)}개의 제목 FAISS 파티션을 성공적으로 로드했습니다.")
    else:
        logging.warning("⚠️ 로드된 제목 FAISS 파티션이 없습니다. 제목 기반 검색이 작동하지 않을 수 있습니다.")
        
    # 백그라운드 워처 시작 (S3 변경 시 제목 프리로드 재실행)
    watch_enabled = os.environ.get("TITLE_PRELOAD_WATCH", "1") in ("1", "true", "TRUE", "yes", "YES")
    watch_interval = float(os.environ.get("TITLE_PRELOAD_WATCH_INTERVAL", "120"))
    include_p10 = os.environ.get("TITLE_PRELOAD_INCLUDE_P10", "1") in ("1", "true", "TRUE", "yes", "YES")
    watch_task = None
    if watch_enabled:
        logging.info(f"🕒 제목 프리로드 감시 시작(interval={watch_interval}s, include_p10={include_p10})")
        watch_task = asyncio.create_task(_watch_titles_preload_task(TITLE_FAISS_S3_PREFIX, watch_interval, include_p10))

    try:
        yield
    finally:
        if watch_task:
            watch_task.cancel()
            with contextlib.suppress(Exception):
                await watch_task
    # 서버 종료 시 실행될 코드
    logging.info("애플리케이션 종료...")

# --- FastAPI 앱 생성 ---
app = FastAPI(lifespan=lifespan)

# 기사 팩트체크 라우터 추가 (기존 로직은 그대로 유지)
app.include_router(create_article_router(lambda: faiss_partition_dirs))

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
