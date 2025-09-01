import os
import json
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from botocore.exceptions import ClientError

from fastapitest.core.preload_s3_faiss import (
    preload_faiss_from_existing_s3,
    CHUNK_CACHE_DIR,
)


def _kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def compute_month_kst() -> str:
    return _kst_now().strftime("%Y%m")


@dataclass
class WatchConfig:
    bucket: str
    base_prefix: str
    interval_sec: int
    state_path: str


def _target_prefixes(cfg: WatchConfig) -> list[str]:
    ym = compute_month_kst()
    monthly = f"{cfg.base_prefix.rstrip('/')}/partition_{ym}/"
    p10 = f"{cfg.base_prefix.rstrip('/')}/partition_10/"
    return [monthly, p10]


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


def _remove_local_partition(prefix: str) -> None:
    import shutil
    part = os.path.basename(os.path.dirname(prefix.rstrip('/')))
    local_dir = os.path.join(CHUNK_CACHE_DIR, part)
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)
        logging.info(f"🧹 로컬 파티션 삭제: {local_dir}")


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser(description="S3 index.faiss 변경 감지 시 제목 FAISS 프리로드 자동 실행 (월별 파티션)")
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db"))
    p.add_argument("--base-prefix", default=os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/"))
    p.add_argument("--interval", type=int, default=int(os.environ.get("WATCH_INTERVAL", "120")))
    p.add_argument("--state-file", default=os.environ.get("WATCH_STATE_FILE", ".watch_titles_state.json"))
    args = p.parse_args()

    cfg = WatchConfig(
        bucket=args.bucket,
        base_prefix=args.base_prefix,
        interval_sec=args.interval,
        state_path=args.state_file,
    )

    s3 = boto3.client("s3")
    state = _load_state(cfg.state_path)
    logging.info(f"🕒 제목 프리로드 감시 시작 (monthly + partition_10, bucket={cfg.bucket}, base={cfg.base_prefix})")

    while True:
        try:
            for prefix in _target_prefixes(cfg):
                faiss_key = f"{prefix}index.faiss"
                pkl_key = f"{prefix}index.pkl"
                faiss_head = _head(s3, cfg.bucket, faiss_key)
                pkl_head = _head(s3, cfg.bucket, pkl_key)
                if not faiss_head or not pkl_head:
                    logging.debug(f"대상 파일 미존재: {prefix}")
                    continue
                tag = f"{faiss_head['ETag']}_{faiss_head['LastModified'].timestamp()}"
                last_seen = state.get(prefix)
                if last_seen != tag:
                    logging.info(f"변경 감지: {faiss_key} → 프리로드 재실행")
                    _remove_local_partition(prefix)
                    preload_faiss_from_existing_s3(prefix)
                    state[prefix] = tag
                    _save_state(cfg.state_path, state)
                else:
                    logging.debug(f"변경 없음: {prefix}")
        except Exception as e:
            logging.error(f"감시 루프 오류: {e}")

        time.sleep(cfg.interval_sec)


if __name__ == "__main__":
    main()
