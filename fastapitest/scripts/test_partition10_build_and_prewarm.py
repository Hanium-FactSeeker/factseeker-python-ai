import os
import re
import glob
import time
import shutil
import logging
import subprocess
from datetime import datetime
from typing import List

import boto3
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


TARGET_PARTITION = "partition_10"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_driver(download_dir: str, headless: bool) -> tuple[webdriver.Chrome, WebDriverWait]:
    opts = Options()
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0,
    }
    opts.add_experimental_option("prefs", prefs)
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(), options=opts)
    driver.maximize_window()
    wait = WebDriverWait(driver, 15)
    return driver, wait


def wait_for_download_complete(download_dir: str, timeout_sec: int = 600) -> str:
    ensure_dir(download_dir)
    start = time.time()
    while time.time() - start < timeout_sec:
        partials = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if not partials:
            files = [p for p in glob.glob(os.path.join(download_dir, "*")) if os.path.isfile(p)]
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                latest = files[0]
                ext = os.path.splitext(latest)[1].lower()
                if ext in (".xlsx", ".xls", ".csv"):
                    return latest
        time.sleep(1)
    raise TimeoutError("다운로드 완료 대기 시간 초과")


def download_bigkinds_range(user_id: str, user_pw: str, start_date: str, end_date: str, download_dir: str, headless: bool = True) -> str:
    """지정 날짜 범위(YYYY-MM-DD ~ YYYY-MM-DD)로 BigKinds 엑셀 다운로드.
    기존 로직을 참고해 로그인 → 언론사/정치 → 날짜 직접 입력 → 분석기사 체크 → 엑셀 다운로드.
    """
    driver, wait = setup_driver(download_dir, headless=headless)
    try:
        driver.get("https://www.bigkinds.or.kr")
        time.sleep(2)
        membership = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "li.topMembership")))
        ActionChains(driver).move_to_element(membership).perform()
        time.sleep(0.6)
        login_link = wait.until(EC.element_to_be_clickable((By.XPATH, '//a[text()="로그인"]')))
        driver.execute_script("arguments[0].click();", login_link)
        wait.until(EC.visibility_of_element_located((By.ID, "login-user-id"))).send_keys(user_id)
        driver.find_element(By.ID, "login-user-password").send_keys(user_pw)
        wait.until(EC.element_to_be_clickable((By.ID, "login-btn"))).click()
        time.sleep(3)

        driver.get("https://www.bigkinds.or.kr/v2/news/index.do")
        time.sleep(2)

        # 언론사: 전국일간지
        tab2 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#srch-tab2']")))
        tab2.click()
        time.sleep(0.5)
        driver.execute_script("document.getElementById('전국일간지').click();")
        time.sleep(0.5)

        # 통합분류: 정치
        tab3 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#srch-tab3']")))
        tab3.click()
        time.sleep(0.5)
        wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@data-role="display" and text()="정치"]'))).click()
        time.sleep(0.5)

        # 기간: 직접 날짜 입력
        tab1 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#srch-tab1']")))
        tab1.click()
        time.sleep(0.3)
        driver.execute_script(f"document.getElementById('search-begin-date').value = '{start_date}';")
        driver.execute_script(f"document.getElementById('search-end-date').value = '{end_date}';")
        driver.execute_script("$('#search-begin-date').trigger('change');")
        driver.execute_script("$('#search-end-date').trigger('change');")
        time.sleep(1)

        # 검색 적용
        search_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.btn.btn-search.news-search-btn.news-report-search-btn')))
        driver.execute_script("arguments[0].scrollIntoView(true);", search_btn)
        ActionChains(driver).move_to_element(search_btn).click().perform()
        time.sleep(3)

        # 분석기사 체크
        label_tm_use = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'label[for="filter-tm-use"]')))
        driver.execute_script("arguments[0].click();", label_tm_use)
        time.sleep(2)

        # 다운로드
        step3_btn = wait.until(EC.element_to_be_clickable((By.ID, "collapse-step-3")))
        driver.execute_script("arguments[0].click();", step3_btn)
        time.sleep(0.5)
        excel_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.news-download-btn.mobile-excel-download')))
        ActionChains(driver).move_to_element(excel_btn).click().perform()

        return wait_for_download_complete(download_dir)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def build_and_upload_partition10(df: pd.DataFrame, embeddings: OpenAIEmbeddings, bucket: str, s3_prefix_base: str) -> None:
    s3 = boto3.client("s3")
    part_name = TARGET_PARTITION
    s3_part_prefix = f"{s3_prefix_base.rstrip('/')}/{part_name}"
    work_dir = os.path.abspath(os.path.join(".tmp", part_name))
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    ensure_dir(work_dir)

    # 기존 인덱스 가져오기(있으면)
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{s3_part_prefix}/")
    if resp.get("Contents"):
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.endswith("index.faiss") or key.endswith("index.pkl"):
                dst = os.path.join(work_dir, os.path.basename(key))
                s3.download_file(bucket, key, dst)

    # 기존 인덱스 로드
    try:
        db = FAISS.load_local(work_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception:
        db = None

    # 컬럼 식별
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    title_candidates = ["제목", "기사제목", "title", "Title"]
    url_candidates = ["URL", "원문URL", "url", "링크", "link", "Link"]
    tcol = next((c for c in df.columns if c in title_candidates), None)
    ucol = next((c for c in df.columns if c in url_candidates), None)
    if not tcol or not ucol:
        raise ValueError(f"제목/URL 컬럼을 찾을 수 없습니다. 컬럼들: {list(df.columns)}")

    existing = set()
    if db:
        for d in db.docstore._dict.values():
            u = (d.metadata or {}).get("url")
            if isinstance(u, str):
                existing.add(u.strip())

    docs: List[Document] = []
    seen = set()
    for _, row in df.iterrows():
        title = str(row.get(tcol, "")).strip()
        url = str(row.get(ucol, "")).strip()
        if not title or not re.match(r"^https?://", url):
            continue
        if url in existing or url in seen:
            continue
        seen.add(url)
        docs.append(Document(page_content=title, metadata={"url": url}))

    if db and docs:
        new_db = FAISS.from_documents(docs, embeddings)
        db.merge_from(new_db)
        db.save_local(work_dir)
    elif not db and docs:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(work_dir)
    elif not db and not docs:
        logging.info("신규 추가 문서가 없어 기존 인덱스를 유지합니다.")

    # 업로드 순서: pkl → faiss
    if os.path.exists(os.path.join(work_dir, "index.pkl")) and os.path.exists(os.path.join(work_dir, "index.faiss")):
        for name in ("index.pkl", "index.faiss"):
            s3.upload_file(os.path.join(work_dir, name), bucket, f"{s3_part_prefix}/{name}")
        logging.info(f"업로드 완료: s3://{bucket}/{s3_part_prefix}/ (pkl→faiss)")
    else:
        logging.warning("업로드할 인덱스 파일이 없습니다.")

    shutil.rmtree(work_dir, ignore_errors=True)


def trigger_prewarm_partition10(concurrency: int = 3, limit: int = 0, s3_prefix_base: str = "feature_faiss_db_openai_partition/") -> int:
    prefix = f"{s3_prefix_base.rstrip('/')}/{TARGET_PARTITION}/"
    cmd = [
        os.sys.executable,
        "-m",
        "fastapitest.scripts.prewarm_articles",
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
    logging.info(f"prewarm 시작: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logging.error(f"prewarm 실패(rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    else:
        logging.info(f"prewarm 완료\nSTDOUT:\n{proc.stdout}")
    return proc.returncode


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 고정 테스트 기간
    start_date = os.environ.get("TEST_START_DATE", "2025-08-26")
    end_date = os.environ.get("TEST_END_DATE", "2025-08-31")

    # 환경 변수
    user_id = os.environ.get("BIGKINDS_USER_ID")
    user_pw = os.environ.get("BIGKINDS_USER_PW")
    if not user_id or not user_pw:
        raise SystemExit("환경변수 BIGKINDS_USER_ID/BIGKINDS_USER_PW 필요")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("환경변수 OPENAI_API_KEY 필요")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

    bucket = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
    s3_prefix = os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/")
    download_dir = os.environ.get("DOWNLOAD_DIR", os.path.abspath("./downloads_test"))
    headless = os.environ.get("HEADLESS", "1") in ("1", "true", "TRUE", "yes", "YES")

    logging.info(f"테스트 범위: {start_date} ~ {end_date} → {TARGET_PARTITION}")
    # 1) 다운로드
    downloaded = download_bigkinds_range(user_id, user_pw, start_date, end_date, download_dir, headless=headless)
    logging.info(f"다운로드 완료: {downloaded}")

    # 2) 빌드/업로드
    df = pd.read_excel(downloaded)
    build_and_upload_partition10(df, embeddings, bucket, s3_prefix)

    # 3) prewarm 트리거(자동 크롤링)
    rc = trigger_prewarm_partition10(concurrency=int(os.environ.get("PREWARM_CONCURRENCY", "3")), limit=int(os.environ.get("PREWARM_LIMIT", "0")), s3_prefix_base=s3_prefix)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
