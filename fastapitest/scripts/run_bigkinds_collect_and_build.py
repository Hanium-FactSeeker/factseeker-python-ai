import os
import re
import glob
import time
import shutil
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple

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


def _kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def compute_partition_month_kst() -> str:
    return _kst_now().strftime("%Y%m")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wait_for_download_complete(download_dir: str, timeout_sec: int = 600) -> str:
    """Wait until there is no .crdownload left and return the latest file path."""
    ensure_dir(download_dir)
    start = time.time()
    last_file = None
    while time.time() - start < timeout_sec:
        partials = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if not partials:
            candidates = [p for p in glob.glob(os.path.join(download_dir, "*")) if os.path.isfile(p)]
            if candidates:
                candidates.sort(key=os.path.getmtime, reverse=True)
                last_file = candidates[0]
                if os.path.splitext(last_file)[1].lower() in (".xlsx", ".xls", ".csv"):
                    return last_file
        time.sleep(1)
    raise TimeoutError(f"다운로드 완료 대기 시간 초과: {download_dir} (마지막 파일: {last_file})")


def move_to_data_folder(src_path: str) -> str:
    day = _kst_now()
    ym = day.strftime("%Y%m")
    ymd = day.strftime("%Y%m%d")
    dst_dir = os.path.join("data", "bigkinds", ym)
    ensure_dir(dst_dir)
    ext = os.path.splitext(src_path)[1].lower() or ".xlsx"
    dst_path = os.path.join(dst_dir, f"bigkinds_{ymd}{ext}")
    shutil.move(src_path, dst_path)
    return dst_path


def build_and_upload_month_partition(
    df: pd.DataFrame,
    embedding_model: OpenAIEmbeddings,
    bucket: str,
    s3_prefix_base: str,
    partition_month: str,
) -> str:
    s3_client = boto3.client("s3")
    part_name = f"partition_{partition_month}"
    s3_part_prefix = f"{s3_prefix_base.rstrip('/')}/{part_name}"

    # 준비: 로컬 작업 디렉터리
    work_dir = os.path.abspath(os.path.join(".tmp", part_name))
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    ensure_dir(work_dir)

    # 1) 기존 인덱스 다운로드(있으면)
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{s3_part_prefix}/")
    if resp.get("Contents"):
        for obj in resp["Contents"]:
            key = obj["Key"]
            if key.endswith("index.faiss") or key.endswith("index.pkl"):
                dst = os.path.join(work_dir, os.path.basename(key))
                s3_client.download_file(bucket, key, dst)

    # 2) 기존 인덱스 로드 또는 신규
    try:
        db = FAISS.load_local(work_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    except Exception:
        db = None

    # 3) 컬럼 감지 및 문서 생성(중복 URL 스킵)
    def _find_cols(df: pd.DataFrame) -> Tuple[str, str]:
        title_candidates = ["제목", "기사제목", "title", "Title"]
        url_candidates = ["URL", "원문URL", "url", "링크", "link", "Link"]
        t = next((c for c in df.columns if c in title_candidates), None)
        u = next((c for c in df.columns if c in url_candidates), None)
        if not t or not u:
            raise ValueError(f"제목/URL 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")
        return t, u

    df = df.copy()
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    tcol, ucol = _find_cols(df)

    existing_urls = set()
    if db:
        for d in db.docstore._dict.values():
            u = (d.metadata or {}).get("url")
            if isinstance(u, str):
                existing_urls.add(u.strip())

    docs: List[Document] = []
    seen = set()
    for _, row in df.iterrows():
        title = str(row.get(tcol, "")).strip()
        url = str(row.get(ucol, "")).strip()
        if not title or not re.match(r"^https?://", url):
            continue
        if url in existing_urls or url in seen:
            continue
        seen.add(url)
        docs.append(Document(page_content=title, metadata={"url": url}))

    if db and docs:
        new_db = FAISS.from_documents(docs, embedding_model)
        db.merge_from(new_db)
        db.save_local(work_dir)
    elif not db and docs:
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(work_dir)
    elif not db and not docs:
        raise ValueError("기존 인덱스도 없고 추가할 문서도 없습니다.")

    # 4) 업로드 순서: pkl -> faiss
    for name in ("index.pkl", "index.faiss"):
        s3_client.upload_file(os.path.join(work_dir, name), bucket, f"{s3_part_prefix}/{name}")

    shutil.rmtree(work_dir, ignore_errors=True)
    return part_name


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


def bigkinds_login_and_download(
    user_id: str,
    user_pw: str,
    download_dir: str,
    headless: bool = True,
) -> str:
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

        # 뉴스 검색 페이지 이동
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

        # 기간: 1일
        tab1 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#srch-tab1']")))
        tab1.click()
        time.sleep(0.3)
        one_day_label = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'label[for="date1-7"]')))
        driver.execute_script("arguments[0].click();", one_day_label)
        time.sleep(0.5)

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
        # 다운로드 완료 대기는 호출자에서 수행
        time.sleep(2)
        return wait_for_download_complete(download_dir)
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 환경 변수
    user_id = os.environ.get("BIGKINDS_USER_ID")
    user_pw = os.environ.get("BIGKINDS_USER_PW")
    if not user_id or not user_pw:
        raise SystemExit("환경변수 BIGKINDS_USER_ID/BIGKINDS_USER_PW가 필요합니다.")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("환경변수 OPENAI_API_KEY가 필요합니다.")

    bucket = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
    s3_prefix = os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/")
    download_dir = os.environ.get("DOWNLOAD_DIR", os.path.abspath("./downloads"))
    headless = os.environ.get("HEADLESS", "1") in ("1", "true", "TRUE", "yes", "YES")

    # 파티션 방식: 기본은 당월(서울시간)
    partition_month = os.environ.get("PARTITION_MONTH") or compute_partition_month_kst()

    # 1) 수집: BigKinds에서 1일 데이터 엑셀 다운로드
    logging.info("📥 BigKinds 다운로드 시작 (1일)")
    downloaded = bigkinds_login_and_download(user_id, user_pw, download_dir, headless=headless)
    moved = move_to_data_folder(downloaded)
    logging.info(f"📦 다운로드 파일 정리: {moved}")

    # 2) 빌드/업로드: 타이틀 인덱스 증분 병합 → S3 pkl→faiss 업로드
    logging.info("🧱 타이틀 인덱스 병합/업로드 시작")
    df = pd.read_excel(moved)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    part_name = build_and_upload_month_partition(df, embeddings, bucket, s3_prefix, partition_month)
    logging.info(f"✅ 업로드 완료: s3://{bucket}/{s3_prefix}{part_name}/ (pkl→faiss)")


if __name__ == "__main__":
    main()
