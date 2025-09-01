# Monthly Partition + EC2/EBS Operations Guide

This document describes how to operate title FAISS partitions on a monthly basis and store article body FAISS caches on EC2 EBS, without offline/manual uploads beyond your existing daily pipeline.

- Keep existing `partition_10` as-is.
- From the next cycle, create a new monthly partition: `partition_YYYYMM`.
- Every day, incrementally update that month’s partition with new titles.
- Article body caches are keyed per-URL in `article_faiss_cache/<url_hash>/`; existing URLs are reused, new ones are created.

## Prerequisites
- Packages: `boto3`, `python-dotenv`, `langchain-community`, `faiss-cpu` installed on EC2.
- IAM permission to read/write the S3 bucket.
- Env: `S3_BUCKET_NAME` (default: `factseeker-faiss-db`) if you use a custom bucket.

## S3 Layout
- Title index (per month): `feature_faiss_db_openai_partition/partition_YYYYMM/{index.pkl,index.faiss}`
- Article body cache (per URL): `article_faiss_cache/<url_hash>/{index.pkl,index.faiss}`

Important: Always upload `index.pkl` first, then `index.faiss`. The preload flow detects `.faiss` and expects both files to be present.

## Monthly Rollover (first day of month)
1) Build initial title index for the new month (from your daily/ETL output of `{title, url}` pairs).
2) Upload to S3 under `feature_faiss_db_openai_partition/partition_YYYYMM/` in this order:
   - `index.pkl`
   - `index.faiss`
3) On EC2, preload and prewarm using only this month’s partition:
   - `python -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_YYYYMM/ --limit 0 --concurrency 3`

Notes
- The script preloads title FAISS, collects URLs from the partition, then ensures article body FAISS per-URL (reuses cache if present, crawls/builds otherwise).

## Daily Incremental Update (same month)
1) Update the monthly title index with new titles (skip already-present URLs) and save.
2) Upload to the SAME S3 path (overwrite) in this order:
   - `index.pkl`
   - `index.faiss`
3) On EC2, force-refresh the local copy of the month’s title partition, then prewarm:
   - Remove local partition directory to bypass “already exists, skip” logic:
     - `rm -rf article_faiss_cache/partition_YYYYMM`
   - Run prewarm to reload the updated partition and process URLs:
     - `python -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_YYYYMM/ --limit 0 --concurrency 3`

Why delete locally? The current preload function skips partitions that already exist locally. Removing the local folder guarantees the updated index is downloaded.

## Store Local Cache on EBS
To keep local cache on EBS (no code change required), mount EBS and symlink `article_faiss_cache` to it.

Example
- Format and mount (adjust device and mount path to your setup):
  - `sudo mkfs -t xfs /dev/nvme1n1`
  - `sudo mkdir -p /mnt/ebs && sudo mount /dev/nvme1n1 /mnt/ebs`
  - `sudo chown -R $USER:$USER /mnt/ebs`
- Prepare cache directory on EBS:
  - `mkdir -p /mnt/ebs/factseekr/article_faiss_cache`
- Move existing cache (if any) and create symlink:
  - `mv article_faiss_cache /mnt/ebs/factseekr/ 2>/dev/null || true`
  - `ln -s /mnt/ebs/factseekr/article_faiss_cache ./article_faiss_cache`
- Add an fstab entry if you need automatic remount on reboot.

## Cron Examples (adjust times/paths)
- Daily at 02:10 (refresh local partition and prewarm this month):
  - `10 2 * * * cd /path/to/repo && rm -rf article_faiss_cache/partition_YYYYMM && /usr/bin/python3 -m fastapitest.scripts.prewarm_articles --source partitions --prefix feature_faiss_db_openai_partition/partition_YYYYMM/ --limit 0 --concurrency 3 >> logs/prewarm_YYYYMM.log 2>&1`

- Monthly at 02:00 on the 1st (roll over to new month; placeholder for your index build step):
  - `0 2 1 * * cd /path/to/repo && /bin/bash -lc 'echo "[PLACEHOLDER] build and upload feature_faiss_db_openai_partition/partition_YYYYMM/{index.pkl,index.faiss} (pkl then faiss)"' >> logs/rollover.log 2>&1`

Tip: If you also want to keep the previous month hot, run prewarm twice (current month, previous month).

## Tuning and Tips
- Concurrency and rate: use `--concurrency` and `--limit` on `prewarm_articles` to control cost/throughput.
- Local cleanup: article caches are backed up to S3; you can prune old local `article_faiss_cache/<url_hash>` folders. They will be re-downloaded when needed.
- S3 versioning: enable on the title partition prefix to allow quick rollbacks if a bad index is uploaded.
- Health checks: confirm the presence of both `index.pkl` and `index.faiss` for each partition you rely on.

## Optional Future Enhancements (not required now)
- Add flags to `prewarm_articles` for forcing reload or selecting specific partitions without manual `rm -rf`.
- Parameterize cache root via environment variable to skip symlinks.

