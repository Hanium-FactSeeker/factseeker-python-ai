import os
import sys
import json
import time
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError


def _kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def compute_month_kst() -> str:
    return _kst_now().strftime("%Y%m")


@dataclass
class WatchConfig:
    bucket: str
    base_prefix: str
    interval_sec: int
    concurrency: int
    limit: int
    state_path: str


def _target_prefix(cfg: WatchConfig) -> str:
    # 월별 파티션만 감시 (서울시간)
    ym = compute_month_kst()
    return f"{cfg.base_prefix.rstrip('/')}/partition_{ym}/"


def _head(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise


def _load_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def _run_prewarm(prefix: str, concurrency: int, limit: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "youtube_verification.scripts.prewarm_articles",
        "--source",
        "partitions",
        "--prefix",
        prefix,
        "--force-reload",
        "--concurrency",
        str(concurrency),
        "--limit",
        str(limit),
    ]
    logging.info(f"🚀 prewarm 실행: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        logging.error(f"prewarm 실패(rc={p.returncode})\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    else:
        logging.info(f"prewarm 완료\nSTDOUT:\n{p.stdout}")
    return p.returncode


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser(description="S3 index.faiss 변경 감지 시 prewarm 자동 실행 (월별 파티션 전용)")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db"))
    p.add_argument("--base-prefix", default=os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/"))
    p.add_argument("--interval", type=int, default=int(os.environ.get("WATCH_INTERVAL", "120")))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("PREWARM_CONCURRENCY", "3")))
    p.add_argument("--limit", type=int, default=int(os.environ.get("PREWARM_LIMIT", "0")))
    p.add_argument("--state-file", default=os.environ.get("WATCH_STATE_FILE", ".watch_state.json"))
    args = p.parse_args()

    cfg = WatchConfig(
        bucket=args.bucket,
        base_prefix=args.base_prefix,
        interval_sec=args.interval,
        concurrency=args.concurrency,
        limit=args.limit,
        state_path=args.state_file,
    )

    s3 = boto3.client("s3")
    state = _load_state(cfg.state_path)

    logging.info(f"🕒 감시 시작 (monthly, bucket={cfg.bucket}, base={cfg.base_prefix})")
    while True:
        try:
            prefix = _target_prefix(cfg)
            faiss_key = f"{prefix}index.faiss"
            pkl_key = f"{prefix}index.pkl"

            # 둘 다 존재하는지 확인
            faiss_head = _head(s3, cfg.bucket, faiss_key)
            pkl_head = _head(s3, cfg.bucket, pkl_key)
            if not faiss_head or not pkl_head:
                logging.info(f"대상 파일 미존재. 대기 (prefix={prefix})")
            else:
                # 변경 감지: etag 또는 lastmodified 비교
                faiss_tag = f"{faiss_head['ETag']}_{faiss_head['LastModified'].timestamp()}"
                last_seen = state.get(prefix)
                if last_seen != faiss_tag:
                    logging.info(f"변경 감지: {faiss_key} (etag/mtime 갱신)")
                    # 짧게 대기 후 prewarm 실행 (업로드 순서 안정화 보조)
                    time.sleep(5)
                    rc = _run_prewarm(prefix, cfg.concurrency, cfg.limit)
                    if rc == 0:
                        state[prefix] = faiss_tag
                        _save_state(cfg.state_path, state)
                else:
                    logging.debug("변경 없음")
        except Exception as e:
            logging.error(f"감시 루프 오류: {e}")

        time.sleep(cfg.interval_sec)


if __name__ == "__main__":
    main()
