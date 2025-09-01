# BigKinds → FAISS Pipeline (Ops Guide)

This document summarizes the end-to-end pipeline and changes implemented in this repo to collect BigKinds data, build monthly title FAISS partitions, prewarm article-body FAISS caches, and keep FastAPI online services in sync.

## S3 Layout
- Titles (monthly or fixed): `s3://factseeker-faiss-db/feature_faiss_db_openai_partition/partition_YYYYMM/{index.pkl,index.faiss}`
- Article body caches (per URL): `s3://factseeker-faiss-db/article_faiss_cache/<url_hash>/{index.pkl,index.faiss}`
- Upload order (important): always upload `index.pkl` first, then `index.faiss`.

## What We Added/Changed

### Scripts
- `fastapitest/scripts/run_bigkinds_collect_and_build.py`
  - Headless Chrome Selenium collection (1-day quick filter) → build/merge monthly title FAISS partition → upload to S3 (pkl → faiss).
  - Uses KST (Asia/Seoul) to compute the current `YYYYMM`.

- `fastapitest/scripts/test_partition10_build_and_prewarm.py`
  - For testing: log in, filter (전국일간지/정치), set custom date range(s) and download Excel, then incrementally merge into `partition_10` and upload, followed by immediate prewarm of article bodies (`--force-reload`).

- `fastapitest/scripts/prewarm_articles.py`
  - Added `--force-reload` flag to delete local partition folder(s) before preloading from S3 to ensure freshness.
  - Usage example: `python -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_YYYYMM/ --force-reload --limit 0 --concurrency 3`.

- Watchers (optional):
  - `fastapitest/scripts/watch_s3_and_prewarm.py`
    - Event-driven “prewarm” watcher (monthly only). Detects `index.faiss` changes and triggers `prewarm_articles --force-reload`.
  - `fastapitest/scripts/watch_s3_titles_preload.py`
    - Title preloader watcher (monthly + `partition_10`). Detects `index.faiss` change and re-preloads title FAISS only.

- Local collectors (Windows/macOS friendly):
  - `bigkinds_to_csv.py`
    - Selenium + webdriver-manager. Logs in, applies filters, date range, downloads Excel once, converts to CSV.
    - “언론사 탭 → 방송사 클릭 → 전국일간지(JS)”. “분석기사”는 `label[for="filter-tm-use"]`를 JS로 클릭(실패 시 `error_filter.png` 저장).
  - `bigkinds_to_csv_uc.py`
    - Undetected Chrome driver (UC). Manual login wait; iterates in 7-day windows; merges multiple Excel files into a single CSV.

### FastAPI (Online Service)
- `fastapitest/main.py`
  - On startup: preload titles from `feature_faiss_db_openai_partition/` (1-time) and scan local `article_faiss_cache/partition_*` to populate `faiss_partition_dirs`.
  - Background watcher (enabled by default) monitors S3 for `partition_YYYYMM` and `partition_10` title index changes. On change:
    - Delete the local target partition folder only.
    - Re-preload from S3.
    - Refresh the in-memory `faiss_partition_dirs` list.
  - Env flags:
    - `TITLE_PRELOAD_WATCH=1|0` (default 1)
    - `TITLE_PRELOAD_WATCH_INTERVAL=120` (seconds)
    - `TITLE_PRELOAD_INCLUDE_P10=1|0` (default 1)

### Partition Ordering for Fact Check
- The system sorts partitions by numeric suffix in descending order (latest first). With monthly names like `partition_202508`, the order becomes:
  - `partition_202508` > `partition_202507` > `partition_10` > `partition_9` …
- Controls:
  - `MAX_EVIDENCES_PER_CLAIM` (default 10): stops when enough validated evidence is collected.
  - `PARTITION_STOP_HITS` (default 1): if N URLs are found in the current (latest) partition, older partitions are not scanned. Set to 10 to “fill from latest first, then fallback to older”.

## Daily Ops (KST)
- Collection/Build (23:30 KST):
  - `python -m fastapitest.scripts.run_bigkinds_collect_and_build`
  - Upload to `feature_faiss_db_openai_partition/partition_YYYYMM/` (pkl → faiss)
- Prewarm (23:40 KST):
  - Either delete local folder then prewarm, or run prewarm with `--force-reload`.
  - Example (monthly):
    - `rm -rf article_faiss_cache/partition_$(date +%Y%m)`
    - `python -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_$(date +%Y%m)/ --limit 0 --concurrency 3`
  - Note: `partition_10` can be excluded per policy (no recrawl), but title preloading watcher may still reload titles.

## Cron (KST) Example
```
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
TZ=Asia/Seoul
BIGKINDS_USER_ID=...   # do not commit real creds
BIGKINDS_USER_PW=...
OPENAI_API_KEY=sk-...
S3_BUCKET_NAME=factseeker-faiss-db

# 23:30 collect/build/upload (monthly partition)
30 23 * * * cd /path/to/repo && /usr/bin/python3 -m fastapitest.scripts.run_bigkinds_collect_and_build >> logs/collect_$(date +\%Y\%m\%d).log 2>&1

# 23:40 prewarm (choose one)
# A) rm then prewarm
40 23 * * * cd /path/to/repo && rm -rf article_faiss_cache/partition_$(date +\%Y\%m) && /usr/bin/python3 -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_$(date +\%Y\%m)/ --limit 0 --concurrency 3 >> logs/prewarm_$(date +\%Y\%m\%d).log 2>&1
# B) force reload prewarm
#40 23 * * * cd /path/to/repo && /usr/bin/python3 -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_$(date +\%Y\%m)/ --force-reload --limit 0 --concurrency 3 >> logs/prewarm_$(date +\%Y\%m\%d).log 2>&1
```

## Environment Variables
- Required
  - `BIGKINDS_USER_ID`, `BIGKINDS_USER_PW`: BigKinds account (do not commit to Git).
  - `OPENAI_API_KEY`: for embeddings.
- Optional
  - `S3_BUCKET_NAME` (default: `factseeker-faiss-db`)
  - `S3_INDEX_PREFIX` (default: `feature_faiss_db_openai_partition/`)
  - `DOWNLOAD_DIR`, `HEADLESS` (collectors)
  - `PREWARM_CONCURRENCY`, `PREWARM_LIMIT`
  - `PARTITION_STOP_HITS`, `MAX_EVIDENCES_PER_CLAIM`
  - `TITLE_PRELOAD_*` flags (FastAPI watcher)

## Local Test (Windows/macOS)
- One-shot collector → CSV (webdriver-manager):
  - File: `bigkinds_to_csv.py`
  - Example:
    - `python bigkinds_to_csv.py --start 2025-08-26 --end 2025-08-31 --out out.csv --download-dir "C:\\Users\\you\\Downloads"`
  - Tips: On Windows, if chromedriver lookup fails, either update Chrome, use `webdriver-manager` (already wired), or switch to UC script below.

- UC + manual-login collector → merged CSV (7-day chunks):
  - File: `bigkinds_to_csv_uc.py`
  - Example:
    - `python bigkinds_to_csv_uc.py --start 2025-08-26 --end 2025-08-31 --out out.csv --download-dir "C:\\Users\\you\\Downloads" --login-wait 20`

## Testing Script (Partition 10)
- File: `fastapitest/scripts/test_partition10_build_and_prewarm.py`
  - Download BigKinds Excel for a custom range (e.g., 2025-08-26 to 2025-08-31), merge into `partition_10`, upload (pkl → faiss), and immediately prewarm with `--force-reload`.
  - Usage (Linux):
    - `export BIGKINDS_USER_ID='...' BIGKINDS_USER_PW='...' OPENAI_API_KEY='sk-...'`
    - `python3 -m fastapitest.scripts.test_partition10_build_and_prewarm`

## Known Issues & Troubleshooting
- Chrome/Driver resolution on Windows:
  - If Selenium Manager fails, either use `undetected-chromedriver`, or `webdriver-manager` (wired in `bigkinds_to_csv.py`), or pin a local driver path.
- BigKinds UI changes or bot detection:
  - Use UC script and manual login (`bigkinds_to_csv_uc.py`).
  - Prefer JS click for certain toggles (e.g., 전국일간지). “분석기사”는 기본적으로 `label[for="filter-tm-use"]` 클릭을 사용.
- Upload order must be `index.pkl` → `index.faiss` to avoid preload race.

### Open Issues To Fix (observed locally)
- Date alert from BigKinds when applying filter:
  - Alert text: `종료일자가 올바른 형식이 아닙니다. 올바른 형식: YYYYY-MM-DD`
  - Action: ensure date format is `YYYY-MM-DD` and that both `input` and `change` events are dispatched after setting values. In scripts we use the helper to fire both events, but the site can still raise an alert under load; add a small wait after setting dates and/or handle/accept unexpected alerts before proceeding.
- Inconsistent filename on download: `NewsResult_20250901-undefine (1).xlsx`
  - Likely tied to the alert/filters sequence; once the date inputs are accepted, the filename should include the proper end date. Track and normalize filenames after download if needed.

## Notes
- We do not recrawl partition_10 by policy; title preload watcher may still reload titles for it (no body prewarm).
- With monthly names, latest wins in scanning order; set `PARTITION_STOP_HITS=10` if you want to fill from the latest partition first before falling back.
- FastAPI watcher is enabled by default and refreshes titles live; disable with `TITLE_PRELOAD_WATCH=0` if not desired.
