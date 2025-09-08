import os
import sys
import re
import glob
import time
import shutil
import logging
import subprocess
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional

import boto3
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys


def _kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def compute_partition_month_kst() -> str:
    return _kst_now().strftime("%Y%m")


def _parse_hhmm(value: str) -> Optional[tuple[int, int]]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", value.strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if 0 <= h <= 23 and 0 <= mi <= 59:
        return h, mi
    return None


def _next_kst_datetime(hour: int, minute: int) -> datetime:
    now = _kst_now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _sleep_until_kst(hour: int, minute: int) -> None:
    target = _next_kst_datetime(hour, minute)
    now = _kst_now()
    secs = (target - now).total_seconds()
    mins = int(secs // 60)
    logging.info(f"⏰ 예약 실행 대기: {target.strftime('%Y-%m-%d %H:%M KST')} (~{mins}분)")
    # 긴 대기는 중간중간 로그를 찍으며 기다린다
    remaining = secs
    while remaining > 0:
        chunk = 300 if remaining > 600 else 60 if remaining > 120 else 10 if remaining > 30 else remaining
        time.sleep(chunk)
        remaining -= chunk
        if remaining > 0:
            left_m = int(max(0, remaining) // 60)
            logging.info(f"... 대기 중 (~{left_m}분 남음)")


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


## CSV는 로컬 저장만 수행(요청에 따라 S3 업로드는 제거)


# Note: merge_excels_to_csv_and_build integration removed. This script always collects 1-day via Selenium.


def build_and_upload_month_partition(
    df: pd.DataFrame,
    embedding_model: OpenAIEmbeddings,
    bucket: str,
    s3_prefix_base: str,
    partition_month: str,
) -> Tuple[str, List[str]]:
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
    used_urls: List[str] = []
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
        used_urls.append(url)

    # 임베딩/FAISS 단계에서 쿼터 초과 등으로 예외가 발생하더라도 종료하지 않도록 보호
    try:
        if db and docs:
            new_db = FAISS.from_documents(docs, embedding_model)
            db.merge_from(new_db)
            db.save_local(work_dir)
        elif not db and docs:
            db = FAISS.from_documents(docs, embedding_model)
            db.save_local(work_dir)
        elif not db and not docs:
            # 기존 인덱스도 없고 추가 문서도 없으면 업로드할 것이 없으므로 종료(비쿼터 사유)
            raise ValueError("기존 인덱스도 없고 추가할 문서도 없습니다.")
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("insufficient_quota", "quota", "rate limit", "429")):
            logging.error(f"OpenAI 임베딩/FAISS 처리 중 쿼터 또는 레이트리밋 오류: {e}")
        else:
            logging.error(f"임베딩/FAISS 처리 중 오류: {e}")
        # 작업 디렉터리 정리 후 업로드 단계는 건너뛰되, 프로그램은 계속 진행
        shutil.rmtree(work_dir, ignore_errors=True)
        return part_name, used_urls

    # 4) 업로드 순서: pkl -> faiss
    for name in ("index.pkl", "index.faiss"):
        try:
            s3_client.upload_file(os.path.join(work_dir, name), bucket, f"{s3_part_prefix}/{name}")
        except Exception as e:
            logging.error(f"S3 업로드 실패(계속 진행): {name} -> s3://{bucket}/{s3_part_prefix}/{name} : {e}")
            # 업로드 중단하되 전체 종료는 하지 않음
            break

    shutil.rmtree(work_dir, ignore_errors=True)
    return part_name, used_urls


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
    # Reduce noisy logs and headless quirks
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    opts.add_argument("--log-level=3")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 15)
    return driver, wait

# -------------------------------
# Selenium helpers (robust)
# -------------------------------
def _accept_unexpected_alerts(driver, wait_timeout: float = 1.0) -> Optional[str]:
    try:
        WebDriverWait(driver, wait_timeout).until(EC.alert_is_present())
        al = driver.switch_to.alert
        txt = al.text
        al.accept()
        return txt
    except TimeoutException:
        return None


def _scroll_into_view(driver, el):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center', inline:'center'});", el)
    except Exception:
        pass


def _open_step_panel(driver, panel_id: str):
    try:
        btn = driver.find_element(By.ID, panel_id)
        expanded = (btn.get_attribute("aria-expanded") or "").lower()
        if expanded in ("", "false"):
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.3)
    except Exception:
        pass


def _open_tab(driver, css_selector: str):
    try:
        tab = WebDriverWait(driver, 7).until(EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)))
        driver.execute_script("arguments[0].click();", tab)
        time.sleep(0.2)
    except Exception:
        pass


def select_national_dailies(driver, wait):
    _open_tab(driver, "a[href='#srch-tab2']")
    time.sleep(0.5)
    try:
        b = driver.find_element(By.ID, "방송사")
        _scroll_into_view(driver, b)
        b.click()
        time.sleep(0.2)
    except Exception:
        pass
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, '전국일간지')))
        driver.execute_script("document.getElementById('전국일간지').click();")
        time.sleep(0.2)
        return
    except Exception:
        pass
    try:
        lbl = driver.find_element(By.CSS_SELECTOR, "label[for='전국일간지']")
        _scroll_into_view(driver, lbl)
        driver.execute_script("arguments[0].click();", lbl)
        time.sleep(0.2)
        return
    except Exception:
        pass
    try:
        node = driver.find_element(By.XPATH, "//*[contains(normalize-space(.), '전국일간지')]")
        _scroll_into_view(driver, node)
        driver.execute_script("arguments[0].click();", node)
        time.sleep(0.2)
    except Exception:
        logging.warning("전국일간지 선택에 실패했습니다. 계속 진행합니다.")


def choose_politics_category(driver, wait):
    _open_tab(driver, "a[href='#srch-tab3']")
    time.sleep(0.3)
    try:
        node = wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@data-role="display" and text()="정치"]')))
        _scroll_into_view(driver, node)
        node.click()
        time.sleep(0.2)
    except Exception:
        logging.warning("통합분류 '정치' 선택 실패. 계속 진행합니다.")


def _ensure_date_direct_input_mode(driver) -> None:
    js = r'''
    return (function(){
      function clickNode(n){ try{ n.scrollIntoView({block:'center'}); n.click(); return true; }catch(e){ return false; } }
      const labels = ['직접입력','사용자 지정','수동입력','수동','직접'];
      for (const t of labels){
        const nodes = Array.from(document.querySelectorAll('label,button,a,span,div'));
        for (const n of nodes){
          const s = (n.innerText||'').replace(/\s+/g,' ').trim();
          if (s && s.includes(t)) { if (clickNode(n)) return 'text'; }
        }
      }
      const radios = Array.from(document.querySelectorAll('input[type=radio],input[type=button]'));
      for (const r of radios){
        const s = ((r.id||'')+' '+(r.value||'')+' '+(r.name||'')).toLowerCase();
        if (/(direct|manual|custom|user|free)/.test(s)){
          if (clickNode(r)) return 'radio';
        }
      }
      return '';
    })();
    '''
    try:
        mode = driver.execute_script(js)
        if mode:
            time.sleep(0.2)
    except Exception:
        pass


def set_date_range_with_events(driver, start_str: str, end_str: str):
    js = r'''
    (function(){
      function setAndDispatch(id, val){
        var el = document.getElementById(id);
        if(!el){return;}
        try { el.removeAttribute('readonly'); } catch(_){ }
        el.value = val;
        if (window.jQuery){
          try { window.jQuery(el).val(val).trigger('input').trigger('change'); } catch(_){ }
        }
        el.dispatchEvent(new Event('input', {bubbles:true}));
        el.dispatchEvent(new Event('change', {bubbles:true}));
      }
      setAndDispatch('search-begin-date', arguments[0]);
      setAndDispatch('search-end-date', arguments[1]);
    })();
    '''
    driver.execute_script(js, start_str, end_str)


def set_date_range_robust(driver, start_str: str, end_str: str, retries: int = 3) -> None:
    def norm(d: str) -> str:
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", d)
        if not m: return d
        y, mo, da = m.groups()
        return f"{y}-{int(mo):02d}-{int(da):02d}"

    start_str = norm(start_str)
    end_str = norm(end_str)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", end_str):
        raise ValueError(f"날짜 형식은 YYYY-MM-DD 이어야 합니다. start={start_str}, end={end_str}")

    _open_tab(driver, "a[href='#srch-tab1']")
    _open_step_panel(driver, "collapse-step-1")
    _ensure_date_direct_input_mode(driver)

    for attempt in range(1, retries + 1):
        try:
            js = r'''
            (function(s,e){
              function setVal(id, val){
                var el = document.getElementById(id);
                if(!el) return false;
                try { el.removeAttribute('readonly'); } catch(_){ }
                el.value = val;
                if (window.jQuery){
                  try { window.jQuery(el).val(val).trigger('input').trigger('change'); } catch(_){ }
                }
                el.dispatchEvent(new Event('input', {bubbles:true}));
                el.dispatchEvent(new Event('change', {bubbles:true}));
                return true;
              }
              var ok1 = setVal('search-begin-date', s);
              var ok2 = setVal('search-end-date', e);
              document.activeElement && document.activeElement.blur && document.activeElement.blur();
              document.body && document.body.click && document.body.click();
              return ok1 && ok2;
            })(arguments[0], arguments[1]);
            '''
            driver.execute_script(js, start_str, end_str)
        except Exception:
            pass

        time.sleep(0.5)
        _accept_unexpected_alerts(driver, wait_timeout=0.7)

        begin_val, end_val = driver.execute_script(
            "return [document.getElementById('search-begin-date')?.value, document.getElementById('search-end-date')?.value];"
        )
        if begin_val == start_str and end_val == end_str:
            return

        try:
            b = driver.find_element(By.ID, 'search-begin-date')
            e = driver.find_element(By.ID, 'search-end-date')
            for el, val in ((b, start_str), (e, end_str)):
                driver.execute_script("arguments[0].removeAttribute('readonly');", el)
                _scroll_into_view(driver, el)
                el.click(); time.sleep(0.05)
                el.send_keys(Keys.CONTROL, 'a'); time.sleep(0.05)
                el.send_keys(val); time.sleep(0.05)
                el.send_keys(Keys.TAB)
            time.sleep(0.5)
            _accept_unexpected_alerts(driver, wait_timeout=0.7)
            begin_val, end_val = driver.execute_script(
                "return [document.getElementById('search-begin-date')?.value, document.getElementById('search-end-date')?.value];"
            )
            if begin_val == start_str and end_val == end_str:
                return
        except Exception:
            pass

        logging.info(f"날짜 재시도 필요(시도 {attempt}/{retries}): begin={begin_val}, end={end_val}")
        time.sleep(0.4)

    set_date_range_with_events(driver, start_str, end_str)
    time.sleep(0.4)
    _accept_unexpected_alerts(driver, wait_timeout=0.7)


def _dismiss_common_overlays(driver):
    try:
        texts = ["확인", "동의", "동의합니다", "닫기", "오늘 그만 보기", "X", "닫  기"]
        candidates = driver.find_elements(
            By.XPATH,
            "//*[contains(@class,'modal') or contains(@class,'layer') or contains(@class,'popup') or contains(@class,'overlay') or contains(@class,'dim')]//button|//a"
        )
        for btn in candidates:
            label = (btn.text or "").strip()
            if any(t in label for t in texts):
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.2)
                except Exception:
                    pass
    except Exception:
        pass


def click_search_button(driver, wait):
    _dismiss_common_overlays(driver)
    selectors = [
        (By.CSS_SELECTOR, "button.news-report-search-btn"),
        (By.CSS_SELECTOR, "button.news-search-btn"),
        (By.CSS_SELECTOR, "button.btn-search"),
        (By.XPATH, "//button[contains(normalize-space(.),'검색하기')]"),
        (By.XPATH, "//a[contains(normalize-space(.),'검색하기')]"),
        (By.XPATH, "//button[contains(.,'검색')]")
    ]
    for by, sel in selectors:
        try:
            el = wait.until(EC.presence_of_element_located((by, sel)))
            _scroll_into_view(driver, el)
            try:
                el = wait.until(EC.element_to_be_clickable((by, sel)))
                ActionChains(driver).move_to_element(el).pause(0.05).click().perform()
            except Exception:
                driver.execute_script("arguments[0].click();", el)
            time.sleep(0.3)
            return True
        except Exception:
            continue
    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ENTER)
        time.sleep(0.3)
        return True
    except Exception:
        pass
    raise TimeoutException("검색 버튼 클릭 실패")


def apply_analysis_article_filter(driver, wait, max_retry: int = 3) -> bool:
    _open_step_panel(driver, "collapse-step-2")
    for i in range(1, max_retry + 1):
        try:
            cb = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "filter-tm-use")))
            _scroll_into_view(driver, cb)
            try:
                label = driver.find_element(By.CSS_SELECTOR, "label[for='filter-tm-use']")
                _scroll_into_view(driver, label)
                driver.execute_script("arguments[0].click();", label)
                time.sleep(0.2)
            except Exception:
                pass
            is_checked = driver.execute_script("return !!document.getElementById('filter-tm-use')?.checked;")
            if not is_checked:
                js_force = r'''
                (function(){
                  var el = document.getElementById('filter-tm-use');
                  if(!el) return false;
                  el.checked = true;
                  el.dispatchEvent(new Event('click', {bubbles:true}));
                  el.dispatchEvent(new Event('input', {bubbles:true}));
                  el.dispatchEvent(new Event('change', {bubbles:true}));
                  return el.checked === true;
                })();
                '''
                is_checked = driver.execute_script(js_force)
                time.sleep(0.2)
            if not is_checked:
                try:
                    _scroll_into_view(driver, cb)
                    cb.click()
                    time.sleep(0.2)
                    is_checked = driver.execute_script("return !!document.getElementById('filter-tm-use')?.checked;")
                except Exception:
                    pass
            if not is_checked:
                logging.warning(f"분석기사 체크 실패(시도 {i}/{max_retry}) - 재시도")
                _accept_unexpected_alerts(driver, wait_timeout=0.5)
                continue
            # 적용하기 버튼
            apply_candidates = [
                (By.XPATH, "//button[contains(normalize-space(.), '적용하기')]"),
                (By.XPATH, "//a[contains(normalize-space(.), '적용하기')]"),
                (By.CSS_SELECTOR, "button.btn-apply"),
                (By.CSS_SELECTOR, "button.apply"),
                (By.CSS_SELECTOR, "button.filter-apply"),
            ]
            applied = False
            for by, sel in apply_candidates:
                try:
                    btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, sel)))
                    _scroll_into_view(driver, btn)
                    driver.execute_script("arguments[0].click();", btn)
                    applied = True
                    break
                except Exception:
                    continue
            if not applied:
                try:
                    btn = driver.find_element(By.XPATH, "//button[contains(., '적용하기') or contains(., '적용')]")
                    _scroll_into_view(driver, btn)
                    driver.execute_script("arguments[0].click();", btn)
                    applied = True
                except Exception:
                    pass
            if not applied:
                logging.warning(f"적용하기 버튼 클릭 실패(시도 {i}/{max_retry})")
                _accept_unexpected_alerts(driver, wait_timeout=0.5)
                continue
            _accept_unexpected_alerts(driver, wait_timeout=1.0)
            time.sleep(1.2)
            still_checked = driver.execute_script("return !!document.getElementById('filter-tm-use')?.checked;")
            logging.info("✅ 분석기사 체크 및 적용하기 완료 (checked=%s)", still_checked)
            return True
        except Exception as e:
            logging.warning(f"분석기사 처리 중 예외(시도 {i}/{max_retry}): {e}")
            _accept_unexpected_alerts(driver, wait_timeout=0.8)
            try:
                driver.save_screenshot(f"error_filter_try{i}.png")
            except Exception:
                pass
            time.sleep(0.6)
    return False


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

        # 언론사/정치
        logging.info("📰 언론사: 전국일간지 선택 시도")
        select_national_dailies(driver, wait)
        logging.info("🏷️ 통합분류: 정치 선택 시도")
        choose_politics_category(driver, wait)

        # 기간: 당일(KST) 1일 범위 직접 입력
        kst_today = _kst_now().strftime("%Y-%m-%d")
        logging.info(f"📅 기간 설정: {kst_today} ~ {kst_today}")
        set_date_range_robust(driver, kst_today, kst_today)
        time.sleep(0.4)

        # 검색 적용
        logging.info("🔎 검색 적용 클릭")
        click_search_button(driver, wait)
        time.sleep(2)

        # 분석기사 체크 → 적용하기
        logging.info("🧩 분석기사 체크/적용")
        apply_analysis_article_filter(driver, wait, max_retry=3)

        # 다운로드
        logging.info("⬇️ 엑셀 다운로드 열기")
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


def _trigger_prewarm_after_upload(urls: List[str], concurrency: int = 3, limit: int = 0) -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmd = [os.sys.executable, "-m", "fastapitest.scripts.prewarm_articles"]
    # 파일 모드로 URL 전달(제목 프리로드 중복 방지)
    tmp_dir = os.path.join(repo_root, ".tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    url_file = os.path.join(tmp_dir, f"urls_partition_daily.txt")
    with open(url_file, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u.strip() + "\n")
    cmd += ["--source", "file", "--file", url_file, "--concurrency", str(concurrency), "--limit", str(limit)]
    logging.info(f"prewarm 시작: {' '.join(cmd)} (cwd={repo_root})")
    # 실시간 출력 스트리밍
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        with subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as p:
            assert p.stdout is not None
            for line in p.stdout:
                logging.info(f"prewarm> {line.rstrip()}")
            rc = p.wait()
    except Exception as e:
        logging.error(f"prewarm 실행 오류: {e}")
        return 1
    if rc != 0:
        logging.error(f"prewarm 실패(rc={rc})")
    else:
        logging.info("prewarm 완료")
    return rc


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 백그라운드 실행 모드: 자기 자신을 분리된 세션으로 재실행하고 즉시 반환
    if os.environ.get("RUN_BIGKINDS_BACKGROUND", "0") in ("1", "true", "TRUE", "yes", "YES") \
       and os.environ.get("RUN_BIGKINDS_DETACHED", "0") != "1":
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            logs_dir = os.path.join(repo_root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_path = os.path.join(logs_dir, f"run_bigkinds_{ts}.log")
            cmd = [sys.executable, "-m", "fastapitest.scripts.run_bigkinds_collect_and_build"]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["RUN_BIGKINDS_DETACHED"] = "1"
            with open(log_path, "a", buffering=1, encoding="utf-8") as f:
                p = subprocess.Popen(
                    cmd,
                    cwd=repo_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    start_new_session=True,
                )
            logging.info(f"🧩 run_bigkinds 백그라운드 시작 (PID={p.pid}, log={log_path})")
            return
        except Exception as e:
            logging.error(f"run_bigkinds 백그라운드 실행 실패: {e}")
            # 계속해서 포그라운드 모드로 진행

    # 예약 실행 옵션: RUN_AT_KST=HH:MM, RUN_DAILY=1
    run_at_kst = os.environ.get("RUN_AT_KST") or os.environ.get("RUN_BIGKINDS_RUN_AT_KST")
    run_daily = os.environ.get("RUN_DAILY", "0") in ("1", "true", "TRUE", "yes", "YES")
    if run_at_kst:
        parsed = _parse_hhmm(run_at_kst)
        if not parsed:
            logging.warning(f"RUN_AT_KST 형식이 올바르지 않습니다(HH:MM): {run_at_kst}. 즉시 실행합니다.")
        else:
            h, mi = parsed
            _sleep_until_kst(h, mi)
            # 만약 매일 반복이면, 이 함수가 한 번 실행 후 루프 돌도록 아래에서 처리

    def _run_once():
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
        # CSV 파이프라인 옵션 (로컬 저장만; S3 업로드는 비활성)
        write_csv = os.environ.get("WRITE_CSV", "1") in ("1", "true", "TRUE", "yes", "YES")
        logging.info("🧾 실행 파라미터: bucket=%s, prefix=%s, headless=%s", bucket, s3_prefix, headless)

        # 파티션 방식: 기본은 당월(서울시간)
        partition_month = os.environ.get("PARTITION_MONTH") or compute_partition_month_kst()

        # 1) 1일 수집: BigKinds 엑셀 다운로드 → CSV(선택) → DF 로드
        logging.info("📥 BigKinds 다운로드 시작 (1일, KST)")
        downloaded = bigkinds_login_and_download(user_id, user_pw, download_dir, headless=headless)
        moved = move_to_data_folder(downloaded)
        logging.info(f"📦 다운로드 파일 정리: {moved}")
        # 엑셀 로드 및 정규화
        df = pd.read_excel(moved)
        df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
        # CSV 저장(필수 컬럼만)
        if write_csv:
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
        logging.info("🧱 타이틀 인덱스 병합/업로드 시작")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
        part_name, used_urls = build_and_upload_month_partition(df, embeddings, bucket, s3_prefix, partition_month)
        logging.info(f"✅ 업로드 완료: s3://{bucket}/{s3_prefix}{part_name}/ (pkl→faiss)")

        # 3) (옵션) 업로드 직후 본문 프리워밍: 파일 모드로 새 URL만 처리
        prewarm_enabled = os.environ.get("PREWARM_AFTER_UPLOAD", "1") in ("1", "true", "TRUE", "yes", "YES")
        if prewarm_enabled:
            logging.info(f"🔥 본문 프리워밍 시작 (urls={len(used_urls)})")
            _trigger_prewarm_after_upload(
                urls=used_urls,
                concurrency=int(os.environ.get("PREWARM_CONCURRENCY", "3")),
                limit=int(os.environ.get("PREWARM_LIMIT", "0")),
            )

    if run_daily and run_at_kst and _parse_hhmm(run_at_kst):
        while True:
            _run_once()
            # 다음 실행까지 대기
            h, mi = _parse_hhmm(run_at_kst)
            _sleep_until_kst(h, mi)
    else:
        _run_once()

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
    # CSV 파이프라인 옵션 (로컬 저장만; S3 업로드는 비활성)
    write_csv = os.environ.get("WRITE_CSV", "1") in ("1", "true", "TRUE", "yes", "YES")
    logging.info("🧾 실행 파라미터: bucket=%s, prefix=%s, headless=%s", bucket, s3_prefix, headless)

    # 파티션 방식: 기본은 당월(서울시간)
    partition_month = os.environ.get("PARTITION_MONTH") or compute_partition_month_kst()

    # 1) 1일 수집: BigKinds 엑셀 다운로드 → CSV(선택) → DF 로드
    logging.info("📥 BigKinds 다운로드 시작 (1일, KST)")
    downloaded = bigkinds_login_and_download(user_id, user_pw, download_dir, headless=headless)
    moved = move_to_data_folder(downloaded)
    logging.info(f"📦 다운로드 파일 정리: {moved}")
    # 엑셀 로드 및 정규화
    df = pd.read_excel(moved)
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    # CSV 저장(필수 컬럼만)
    if write_csv:
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
    logging.info("🧱 타이틀 인덱스 병합/업로드 시작")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    part_name, used_urls = build_and_upload_month_partition(df, embeddings, bucket, s3_prefix, partition_month)
    logging.info(f"✅ 업로드 완료: s3://{bucket}/{s3_prefix}{part_name}/ (pkl→faiss)")

    # 3) (옵션) 업로드 직후 본문 프리워밍: 파일 모드로 새 URL만 처리
    prewarm_enabled = os.environ.get("PREWARM_AFTER_UPLOAD", "1") in ("1", "true", "TRUE", "yes", "YES")
    if prewarm_enabled:
        logging.info(f"🔥 본문 프리워밍 시작 (urls={len(used_urls)})")
        _trigger_prewarm_after_upload(
            urls=used_urls,
            concurrency=int(os.environ.get("PREWARM_CONCURRENCY", "3")),
            limit=int(os.environ.get("PREWARM_LIMIT", "0")),
        )


if __name__ == "__main__":
    main()
