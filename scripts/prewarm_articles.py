import os
import asyncio
import logging
import random 
import time 
from typing import List, Set

from dotenv import load_dotenv

# Reuse existing app components without modifying them
from fastapitest.services.fact_checker import ensure_article_faiss, embed_model
from fastapitest.core.preload_s3_faiss import preload_faiss_from_existing_s3, CHUNK_CACHE_DIR
from langchain_community.vectorstores import FAISS


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


def _urls_from_partitions(parts: List[str]) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        try:
            db = FAISS.load_local(part, embeddings=embed_model, allow_dangerous_deserialization=True)
            for doc in db.docstore._dict.values():
                u = (doc.metadata or {}).get("url")
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)
        except Exception as e:
            logging.error(f"파티션 로드 실패: {part} -> {e}")
    return urls


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


async def main_async(prefix: str, source: str, url_file: str | None, limit: int, concurrency: int, partition_number: int | None, min_delay: float, max_delay: float):
    if source == "partitions":
        logging.info(f"🚀 제목 FAISS S3 프리로드 시작 (prefix={prefix})")
        preload_faiss_from_existing_s3(prefix)
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
    ))


if __name__ == "__main__":
    main()
