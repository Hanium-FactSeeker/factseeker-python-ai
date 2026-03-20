import os
import sys
import asyncio
import logging
import random 
import time 
from typing import List, Set
from pathlib import Path
import boto3

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS


def _resolve_imports():
    """환경에 따라 동적으로 모듈을 불러옵니다.
    - 우선 youtube_verification 패키지 경로로 시도
    - 실패 시 youtube_verification 디렉토리를 sys.path에 추가하고 패키지 내부 상대 경로로 시도
    """
    try:
        from services.fact_checker import ensure_article_faiss as _ensure, embed_model as _embed
        from core.preload_s3_faiss import preload_faiss_from_existing_s3 as _preload, CHUNK_CACHE_DIR as _cache
        return _ensure, _embed, _preload, _cache
    except ModuleNotFoundError:
        pkg_dir = Path(__file__).resolve().parents[1]  # youtube_verification 디렉토리
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        from services.fact_checker import ensure_article_faiss as _ensure, embed_model as _embed
        from core.preload_s3_faiss import preload_faiss_from_existing_s3 as _preload, CHUNK_CACHE_DIR as _cache
        return _ensure, _embed, _preload, _cache


ensure_article_faiss, embed_model, preload_faiss_from_existing_s3, CHUNK_CACHE_DIR = _resolve_imports()


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _find_partitions(target_partition: int | None = None) -> List[str]:
    parts: List[str] = []
    if not os.path.exists(CHUNK_CACHE_DIR):
        return parts

    def part_num(name: str) -> int:
        import re
        base = os.path.basename(name)
        m = re.search(r"(\d+)", base)
        return int(m.group(1)) if m else -1

    for item in os.listdir(CHUNK_CACHE_DIR):
        p = os.path.join(CHUNK_CACHE_DIR, item)
        if os.path.isdir(p) and item.startswith("partition_"):
            current_part_num = part_num(item)
            if target_partition is None or current_part_num == target_partition:
                parts.append(p)
    parts.sort(key=part_num, reverse=True)
    return parts


def _is_valid_url(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


def _urls_from_partitions(parts: List[str]) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        try:
            db = FAISS.load_local(part, embeddings=embed_model, allow_dangerous_deserialization=True)
            for doc in db.docstore._dict.values():
                u = (doc.metadata or {}).get("url")
                if _is_valid_url(u) and u not in seen:
                    seen.add(u)
                    urls.append(u)
        except Exception as e:
            logging.error(f"파티션 로드 실패: {part} -> {e}")
    return urls


def _barrier_path() -> str:
    return os.path.join(CHUNK_CACHE_DIR, ".preload.lock")


def _acquire_preload_barrier():
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    path = _barrier_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
        logging.info(f"🔒 프리로드 배리어 생성: {path}")
    except Exception as e:
        logging.warning(f"프리로드 배리어 생성 실패(무시): {e}")


def _release_preload_barrier():
    path = _barrier_path()
    try:
        if os.path.exists(path):
            os.remove(path)
            logging.info("🔓 프리로드 배리어 해제")
    except Exception as e:
        logging.warning(f"프리로드 배리어 해제 실패(무시): {e}")


def _expected_partitions_from_s3(prefix: str) -> Set[str]:
    """S3에서 기대되는 파티션 디렉터리 이름 집합을 계산(.faiss 기준)."""
    parts: Set[str] = set()
    try:
        s3 = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj.get('Key')
                if key and key.endswith('.faiss'):
                    parts.add(os.path.basename(os.path.dirname(key)))
    except Exception as e:
        logging.warning(f"S3 파티션 조회 실패(건너뜀): {e}")
    return parts


async def _wait_until_preload_complete(prefix: str, timeout_sec: float = 900.0, poll_interval_sec: float = 3.0) -> None:
    """
    프리로드 완료를 보장하기 위해, S3에 존재하는 모든 파티션(partition_*)의
    index.faiss/index.pkl이 로컬에 존재할 때까지 대기합니다.
    """
    expected = _expected_partitions_from_s3(prefix)
    if not expected:
        logging.info("S3에서 기대 파티션이 없어 대기 없이 진행합니다.")
        return

    logging.info(f"⏳ 프리로드 완료 대기 시작 (기대 파티션 {len(expected)}개)")
    start = time.time()

    def _has_all_locally() -> tuple[int, int]:
        ok = 0
        total = len(expected)
        for part in expected:
            local_dir = os.path.join(CHUNK_CACHE_DIR, part)
            if os.path.isdir(local_dir):
                faiss_ok = os.path.exists(os.path.join(local_dir, 'index.faiss'))
                pkl_ok = os.path.exists(os.path.join(local_dir, 'index.pkl'))
                if faiss_ok and pkl_ok:
                    ok += 1
        return ok, total

    while True:
        ok, total = _has_all_locally()
        if ok >= total:
            logging.info("✅ 프리로드된 파티션이 모두 확인되었습니다.")
            return
        if time.time() - start > timeout_sec:
            logging.warning(f"⚠️ 프리로드 대기 시간 초과: {ok}/{total}개 확인됨. 계속 진행합니다.")
            return
        logging.info(f"... 대기 중: {ok}/{total}개 확인됨 (폴링 {poll_interval_sec:.0f}s)")
        await asyncio.sleep(poll_interval_sec)


async def _bounded_prewarm(urls: List[str], concurrency: int, min_delay: float, max_delay: float) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(u: str):
        async with sem:
            # Add random delay here
            delay = random.uniform(min_delay, max_delay)
            logging.info(f"⏳ {delay:.2f}초 대기 후 {u} 처리 시작...")
            await asyncio.sleep(delay)

            try:
                db = await ensure_article_faiss(u)
                if db:
                    logging.info(f"✅ 프리워밍 완료: {u}")
                else:
                    logging.warning(f"⚠️ 프리워밍 실패/본문부족: {u}")
            except Exception as e:
                logging.error(f"❌ 프리워밍 중 오류: {u} -> {e}")

    tasks = [asyncio.create_task(one(u)) for u in urls]
    await asyncio.gather(*tasks)


async def main_async(prefix: str, source: str, url_file: str | None, limit: int, concurrency: int, partition_number: int | None, min_delay: float, max_delay: float, preload_wait_timeout: float, preload_poll_interval: float, force_reload: bool):
    if source == "partitions":
        logging.info(f"🚀 제목 FAISS S3 프리로드 시작 (prefix={prefix})")
        # 강제 리로드 요청 시: 대상 파티션의 로컬 캐시 삭제로 항상 최신 상태 재다운로드
        if force_reload:
            try:
                targets = _expected_partitions_from_s3(prefix)
                if partition_number is not None:
                    targets = {f"partition_{partition_number}"}
                for part in targets:
                    local_dir = os.path.join(CHUNK_CACHE_DIR, part)
                    if os.path.isdir(local_dir):
                        import shutil
                        shutil.rmtree(local_dir, ignore_errors=True)
                        logging.info(f"🧹 강제 리로드: 로컬 캐시 삭제 -> {local_dir}")
            except Exception as e:
                logging.warning(f"강제 리로드 처리 중 경고(계속 진행): {e}")
        _acquire_preload_barrier()
        try:
            preload_faiss_from_existing_s3(prefix)
            await _wait_until_preload_complete(prefix, timeout_sec=preload_wait_timeout, poll_interval_sec=preload_poll_interval)
        finally:
            _release_preload_barrier()
        parts = _find_partitions(target_partition=partition_number)
        logging.info(f"🔢 감지된 파티션 수: {len(parts)}")
        urls = _urls_from_partitions(parts)
        logging.info(f"🔎 파티션에서 수집된 URL 수: {len(urls)}")
    else:
        if not url_file or not os.path.exists(url_file):
            raise SystemExit("--file 경로가 유효하지 않습니다.")
        with open(url_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        logging.info(f"📄 파일에서 URL {len(urls)}건 로드")

    if limit > 0:
        urls = urls[:limit]
        logging.info(f"⏱️ 제한 적용: {limit}건 대상으로 프리워밍 실행")

    if not urls:
        logging.warning("프리워밍할 URL이 없습니다.")
        return

    await _bounded_prewarm(urls, concurrency=concurrency, min_delay=min_delay, max_delay=max_delay)


def main():
    load_dotenv()
    import argparse

    p = argparse.ArgumentParser(description="기사 본문/FAISS 캐시 사전 구축 스크립트")
    p.add_argument("--prefix", default="feature_faiss_db_openai_partition/", help="S3 제목 FAISS 프리픽스")
    p.add_argument("--source", choices=["partitions", "file"], default="partitions", help="URL 소스 선택")
    p.add_argument("--file", help="--source file일 때 사용할 URL 목록 파일 경로")
    p.add_argument("--limit", type=int, default=0, help="최대 처리 URL 수(0=무제한)")
    p.add_argument("--concurrency", type=int, default=3, help="동시 프리워밍 개수")
    p.add_argument("--partition", type=int, default=None, help="처리할 특정 파티션 번호 (예: 1)")
    p.add_argument("--min-delay", type=float, default=1.0, help="각 URL 처리 전 최소 대기 시간 (초)")
    p.add_argument("--max-delay", type=float, default=5.0, help="각 URL 처리 전 최대 대기 시간 (초)")
    p.add_argument("--preload-wait-timeout", type=float, default=900.0, help="프리로드 완료 대기 최대 시간(초)")
    p.add_argument("--preload-poll-interval", type=float, default=3.0, help="프리로드 상태 폴링 주기(초)")
    p.add_argument("--force-reload", action="store_true", help="로컬 파티션 캐시 삭제 후 S3에서 재프리로드")
    args = p.parse_args()

    asyncio.run(main_async(
        prefix=args.prefix,
        source=args.source,
        url_file=args.file,
        limit=args.limit,
        concurrency=args.concurrency,
        partition_number=args.partition,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        preload_wait_timeout=args.preload_wait_timeout,
        preload_poll_interval=args.preload_poll_interval,
        force_reload=args.force_reload,
    ))


if __name__ == "__main__":
    main()
