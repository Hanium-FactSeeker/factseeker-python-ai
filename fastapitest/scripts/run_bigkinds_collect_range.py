import os
import sys
import logging
import csv
from datetime import datetime
from typing import Optional

import pandas as pd
from langchain_openai import OpenAIEmbeddings

# 내부 유틸은 기존 스크립트에서 재사용
from fastapitest.scripts.run_bigkinds_collect_and_build import (
    setup_driver,
    wait_for_download_complete,
    move_to_data_folder,
    select_national_dailies,
    choose_politics_category,
    set_date_range_robust,
    click_search_button,
    apply_analysis_article_filter,
    build_and_upload_month_partition,
)


def _ym_from_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m")


def download_bigkinds_range(user_id: str, user_pw: str, start_date: str, end_date: str, download_dir: str, headless: bool = True) -> str:
    driver, wait = setup_driver(download_dir, headless=headless)
    try:
        # 로그인
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.action_chains import ActionChains
        import time

        driver.get("https://www.bigkinds.or.kr")
        time.sleep(2)
        membership = WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "li.topMembership")))
        ActionChains(driver).move_to_element(membership).perform()
        time.sleep(0.6)
        login_link = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, '//a[text()="로그인"]')))
        driver.execute_script("arguments[0].click();", login_link)

        WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.ID, "login-user-id"))).send_keys(user_id)
        driver.find_element(By.ID, "login-user-password").send_keys(user_pw)
        WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.ID, "login-btn"))).click()
        time.sleep(3)

        # 검색 페이지 이동
        driver.get("https://www.bigkinds.or.kr/v2/news/index.do")
        time.sleep(2)

        # 필터: 언론사/정치
        logging.info("📰 언론사: 전국일간지 선택 시도")
        select_national_dailies(driver, wait)
        logging.info("🏷️ 통합분류: 정치 선택 시도")
        choose_politics_category(driver, wait)

        # 기간 설정(인자 기반)
        logging.info(f"📅 기간 설정(인자): {start_date} ~ {end_date}")
        set_date_range_robust(driver, start_date, end_date)
        time.sleep(0.4)

        # 검색 적용 및 분석기사 필터
        logging.info("🔎 검색 적용 클릭")
        click_search_button(driver, wait)
        time.sleep(2)

        logging.info("🧩 분석기사 체크/적용")
        apply_analysis_article_filter(driver, wait, max_retry=3)

        # 엑셀 다운로드
        logging.info("⬇️ 엑셀 다운로드 진행")
        step3_btn = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.ID, "collapse-step-3")))
        driver.execute_script("arguments[0].click();", step3_btn)
        time.sleep(0.5)
        excel_btn = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.news-download-btn.mobile-excel-download')))
        ActionChains(driver).move_to_element(excel_btn).click().perform()
        time.sleep(2)
        return wait_for_download_complete(download_dir)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    p = argparse.ArgumentParser(description="BigKinds 수집(기간 지정) + 타이틀 인덱스 빌드/업로드")
    p.add_argument("--start-date", dest="start_date", default=os.environ.get("START_DATE") or os.environ.get("BIGKINDS_START_DATE"), help="YYYY-MM-DD")
    p.add_argument("--end-date", dest="end_date", default=os.environ.get("END_DATE") or os.environ.get("BIGKINDS_END_DATE"), help="YYYY-MM-DD")
    p.add_argument("--download-dir", default=os.environ.get("DOWNLOAD_DIR", os.path.abspath("./downloads_range")))

    # env 기반 기본값 계산
    env_headless = os.environ.get("HEADLESS", "1") in ("1", "true", "TRUE", "yes", "YES")
    env_write_csv = os.environ.get("WRITE_CSV", "1") in ("1", "true", "TRUE", "yes", "YES")
    env_prewarm = os.environ.get("PREWARM_AFTER_UPLOAD", "1") in ("1", "true", "TRUE", "yes", "YES")

    # 토글 가능한 플래그들 (서로 배타 옵션 제공)
    g_headless = p.add_mutually_exclusive_group()
    g_headless.add_argument("--headless", dest="headless", action="store_true", help="브라우저를 헤드리스로 실행")
    g_headless.add_argument("--no-headless", dest="headless", action="store_false", help="브라우저 창 표시")
    p.set_defaults(headless=env_headless)

    g_csv = p.add_mutually_exclusive_group()
    g_csv.add_argument("--write-csv", dest="write_csv", action="store_true", help="CSV 저장(필수 컬럼만)")
    g_csv.add_argument("--no-write-csv", dest="write_csv", action="store_false", help="CSV 저장 비활성화")
    p.set_defaults(write_csv=env_write_csv)

    g_prewarm = p.add_mutually_exclusive_group()
    g_prewarm.add_argument("--prewarm", dest="prewarm", action="store_true", help="수집 후 본문 프리워밍 실행")
    g_prewarm.add_argument("--no-prewarm", dest="prewarm", action="store_false", help="프리워밍 비활성화")
    p.set_defaults(prewarm=env_prewarm)

    p.add_argument("--prewarm-concurrency", type=int, default=int(os.environ.get("PREWARM_CONCURRENCY", "3")))
    p.add_argument("--prewarm-limit", type=int, default=int(os.environ.get("PREWARM_LIMIT", "0")))
    p.add_argument("--bucket", default=os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db"))
    p.add_argument("--s3-prefix", default=os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/"))
    p.add_argument("--partition-month", default=None, help="YYYYMM (기본: start-date 기준)")
    args = p.parse_args()

    if not args.start_date or not args.end_date:
        raise SystemExit("--start-date, --end-date 둘 다 필요합니다.")

    user_id = os.environ.get("BIGKINDS_USER_ID")
    user_pw = os.environ.get("BIGKINDS_USER_PW")
    if not user_id or not user_pw:
        raise SystemExit("환경변수 BIGKINDS_USER_ID/BIGKINDS_USER_PW가 필요합니다.")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("환경변수 OPENAI_API_KEY가 필요합니다.")

    # 1) 다운로드
    downloaded = download_bigkinds_range(user_id, user_pw, args.start_date, args.end_date, args.download_dir, headless=args.headless)
    moved = move_to_data_folder(downloaded)
    logging.info(f"📦 다운로드 파일 정리: {moved}")

    # 2) 엑셀 로드 + (옵션) CSV 저장
    df = pd.read_excel(moved)
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    if args.write_csv:
        csv_path = os.path.splitext(moved)[0] + ".csv"
        try:
            required_cols = ["일자", "언론사", "제목", "URL", "특성추출(가중치순 상위 50개)"]
            out = pd.DataFrame()
            for c in required_cols:
                if c in df.columns:
                    out[c] = df[c]
                else:
                    out[c] = ""
            out = out.fillna("")
            out.to_csv(csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
            logging.info(f"📝 CSV 로컬 저장 완료(필수 컬럼): {csv_path}")
        except Exception as e:
            logging.warning(f"CSV 저장 실패(계속 진행): {e}")

    # 3) 타이틀 인덱스 빌드/업로드 (월 파티션)
    partition_month = args.partition_month or _ym_from_date(args.start_date)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    part_name, used_urls = build_and_upload_month_partition(df, embeddings, args.bucket, args.s3_prefix, partition_month)
    logging.info(f"✅ 업로드 완료: s3://{args.bucket}/{args.s3_prefix.rstrip('/')}/{part_name}/")

    # 4) (옵션) 프리워밍 실행
    if args.prewarm:
        try:
            # 내부 함수를 늦은 바인딩으로 가져와 호출 (순환 import 회피)
            from fastapitest.scripts.run_bigkinds_collect_and_build import _trigger_prewarm_after_upload
            rc = _trigger_prewarm_after_upload(
                urls=used_urls,
                concurrency=args.prewarm_concurrency,
                limit=args.prewarm_limit,
            )
            if rc != 0:
                logging.error(f"prewarm 실패(rc={rc})")
        except Exception as e:
            logging.error(f"prewarm 실행 중 오류(무시): {e}")


if __name__ == "__main__":
    main()
