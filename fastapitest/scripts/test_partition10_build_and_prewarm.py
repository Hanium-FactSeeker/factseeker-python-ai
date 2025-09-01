# test_partition10_build_and_prewarm.py
# -*- coding: utf-8 -*-
import os
import re
import glob
import time
import shutil
import logging
import warnings
import subprocess
from typing import List, Optional

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

TARGET_PARTITION = "partition_10"
REQUIRED_CSV_COLUMNS = ["일자", "언론사", "제목", "URL", "특성추출(가중치순 상위 50개)"]


# -------------------------------
# Filesystem helpers
# -------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wait_for_download_complete(download_dir: str, timeout_sec: int = 600) -> str:
    """크롬 다운로드 완료 대기(.crdownload 없어질 때까지), 엑셀/CSV 확장자 파일이 보이면 경로 반환."""
    ensure_dir(download_dir)
    start = time.time()
    last_file: Optional[str] = None
    while time.time() - start < timeout_sec:
        partials = glob.glob(os.path.join(download_dir, "*.crdownload"))
        if not partials:
            files = [p for p in glob.glob(os.path.join(download_dir, "*")) if os.path.isfile(p)]
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                last_file = files[0]
                ext = os.path.splitext(last_file)[1].lower()
                if ext in (".xlsx", ".xls", ".csv"):
                    return last_file
        time.sleep(1)
    raise TimeoutError(f"다운로드 완료 대기 시간 초과 (last={last_file})")


# -------------------------------
# Selenium utils
# -------------------------------
def _accept_unexpected_alerts(driver, wait_timeout: float = 1.0) -> Optional[str]:
    """예상치 못한 alert이 떠있으면 OK로 닫고 텍스트 반환."""
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
    """주어진 패널(#collapse-step-X)이 접혀 있으면 펼친다."""
    try:
        btn = driver.find_element(By.ID, panel_id)
        expanded = (btn.get_attribute("aria-expanded") or "").lower()
        if expanded in ("", "false"):
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(0.3)
    except Exception:
        pass


def _open_tab(driver, css_selector: str):
    """탭 a[href='#srch-tabX']를 클릭."""
    try:
        tab = WebDriverWait(driver, 7).until(EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)))
        driver.execute_script("arguments[0].click();", tab)
        time.sleep(0.2)
    except Exception:
        pass


# -------------------------------
# Media (언론사) & Category (통합분류)
# -------------------------------
def select_national_dailies(driver, wait):
    """언론사 탭에서 '전국일간지'를 안정적으로 체크."""
    logging.info("📰 언론사 탭 진입 및 '전국일간지' 선택 시도")
    _open_tab(driver, "a[href='#srch-tab2']")
    time.sleep(0.5)

    # 일부 레이아웃에서 먼저 '방송사'를 눌러야 다른 옵션이 활성화됨
    try:
        b = driver.find_element(By.ID, "방송사")
        _scroll_into_view(driver, b)
        b.click()
        time.sleep(0.2)
    except Exception:
        pass

    # ID로 직접 클릭 → 실패 시 라벨/fallback XPATH
    try:
        wait.until(EC.presence_of_element_located((By.ID, '전국일간지')))
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
        logging.warning("⚠️ 전국일간지 선택에 실패했습니다. 계속 진행합니다.")


def choose_politics_category(driver, wait):
    """통합분류 탭에서 '정치' 선택."""
    logging.info("🏷️ 통합분류 탭 진입 및 '정치' 선택 시도")
    _open_tab(driver, "a[href='#srch-tab3']")
    time.sleep(0.3)
    try:
        node = wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@data-role="display" and text()="정치"]')))
        _scroll_into_view(driver, node)
        node.click()
        time.sleep(0.2)
    except Exception:
        logging.warning("⚠️ 통합분류 '정치' 선택 실패. 계속 진행합니다.")


# -------------------------------
# Date input
# -------------------------------
def _ensure_date_direct_input_mode(driver) -> None:
    """기간 입력을 '직접입력/사용자 지정' 모드로 전환 시도."""
    js = r'''
    return (function(){
      function clickNode(n){
        try{ n.scrollIntoView({block:'center'}); n.click(); return true; }catch(e){ return false; }
      }
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
        try { el.removeAttribute('readonly'); } catch(_){}
        el.value = val;
        if (window.jQuery){
          try { window.jQuery(el).val(val).trigger('input').trigger('change'); } catch(_){}
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
    """
    날짜 입력을 최대한 확실하게 적용한다.
    - STEP1 패널 강제 오픈
    - 직접입력 모드 전환
    - readonly 제거 후 값 주입 + 이벤트 트리거
    - 폴백으로 키보드 입력
    - 알럿 발생 시 자동 수락 후 재시도
    """
    def norm(d: str) -> str:
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", d)
        if not m:
            return d
        y, mo, da = m.groups()
        return f"{y}-{int(mo):02d}-{int(da):02d}"

    start_str = norm(start_str)
    end_str = norm(end_str)

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", end_str):
        raise ValueError(f"날짜 형식은 YYYY-MM-DD 이어야 합니다. start={start_str}, end={end_str}")

    logging.info(f"📅 기간 설정 시도: {start_str} ~ {end_str}")
    # STEP1 확보 + 직접입력 모드
    _open_tab(driver, "a[href='#srch-tab1']")
    _open_step_panel(driver, "collapse-step-1")
    _ensure_date_direct_input_mode(driver)

    for attempt in range(1, retries + 1):
        # A. JS 주입 + 이벤트
        try:
            js = r'''
            (function(s,e){
              function setVal(id, val){
                var el = document.getElementById(id);
                if(!el) return false;
                try { el.removeAttribute('readonly'); } catch(_){}
                el.value = val;
                if (window.jQuery){
                  try { window.jQuery(el).val(val).trigger('input').trigger('change'); } catch(_){}
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

        # B. 폴백: 키보드 입력
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

        logging.info(f"⟳ 날짜 재시도 필요(시도 {attempt}/{retries}): begin={begin_val}, end={end_val}")
        time.sleep(0.4)

    # 마지막 시도
    set_date_range_with_events(driver, start_str, end_str)
    time.sleep(0.4)
    _accept_unexpected_alerts(driver, wait_timeout=0.7)


# -------------------------------
# 분석기사 필터
# -------------------------------
def apply_analysis_article_filter(driver, wait, max_retry: int = 3) -> bool:
    """'분석기사' 체크 후 '적용하기'까지 신뢰성 있게 수행."""
    logging.info("🧩 '분석기사' 체크 및 '적용하기' 수행")
    _open_step_panel(driver, "collapse-step-2")  # 필터/조건 패널

    for i in range(1, max_retry + 1):
        try:
            cb = WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.ID, "filter-tm-use")))
            _scroll_into_view(driver, cb)

            # 1) 라벨 우선 클릭
            try:
                label = driver.find_element(By.CSS_SELECTOR, "label[for='filter-tm-use']")
                _scroll_into_view(driver, label)
                driver.execute_script("arguments[0].click();", label)
                time.sleep(0.2)
            except Exception:
                pass

            # 2) 체크 상태 강제 + 이벤트 트리거
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
                # 마지막 시도: 체크박스 본체 클릭
                try:
                    _scroll_into_view(driver, cb)
                    cb.click()
                    time.sleep(0.2)
                    is_checked = driver.execute_script("return !!document.getElementById('filter-tm-use')?.checked;")
                except Exception:
                    pass

            if not is_checked:
                logging.warning(f"⚠️ 분석기사 체크 실패(시도 {i}/{max_retry}) - 다시 시도")
                _accept_unexpected_alerts(driver, wait_timeout=0.5)
                continue

            # 3) '적용하기' 버튼 클릭 (다중 셀렉터)
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
                # 최후의 수단: 텍스트 포함 임의 버튼 탐색
                try:
                    btn = driver.find_element(By.XPATH, "//button[contains(., '적용하기') or contains(., '적용')]")
                    _scroll_into_view(driver, btn)
                    driver.execute_script("arguments[0].click();", btn)
                    applied = True
                except Exception:
                    pass

            if not applied:
                logging.warning(f"⚠️ 적용하기 버튼 클릭 실패(시도 {i}/{max_retry})")
                _accept_unexpected_alerts(driver, wait_timeout=0.5)
                continue

            _accept_unexpected_alerts(driver, wait_timeout=1.0)
            time.sleep(1.2)

            still_checked = driver.execute_script("return !!document.getElementById('filter-tm-use')?.checked;")
            logging.info("✅ 분석기사 체크 및 적용하기 완료 (checked=%s)", still_checked)
            return True

        except Exception as e:
            logging.warning(f"⚠️ 분석기사 처리 중 예외(시도 {i}/{max_retry}): {e}")
            _accept_unexpected_alerts(driver, wait_timeout=0.8)
            try:
                driver.save_screenshot(f"error_filter_try{i}.png")
            except Exception:
                pass
            time.sleep(0.6)

    return False


# -------------------------------
# Overlays & Search button
# -------------------------------
def _is_visible(driver, el) -> bool:
    try:
        if not el.is_displayed():
            return False
        rect = driver.execute_script(
            "var r=arguments[0].getBoundingClientRect(); return [r.width,r.height];", el
        )
        return (rect[0] > 0 and rect[1] > 0)
    except Exception:
        return False


def _dismiss_common_overlays(driver):
    """쿠키/모달/레이어 팝업 닫기 시도."""
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


def click_search_button(driver, wait, max_retry: int = 2):
    """
    BigKinds 페이지에서 '검색하기'를 확실히 누른다.
    - 오버레이/쿠키 팝업 닫기
    - 다중 셀렉터 탐색
    - JS 클릭 폴백
    """
    _dismiss_common_overlays(driver)

    selectors = [
        (By.CSS_SELECTOR, "button.news-report-search-btn"),
        (By.CSS_SELECTOR, "button.news-search-btn"),
        (By.CSS_SELECTOR, "button.btn-search"),
        (By.XPATH, "//button[contains(normalize-space(.),'검색하기')]"),
        (By.XPATH, "//a[contains(normalize-space(.),'검색하기')]"),
        (By.XPATH, "//button[contains(.,'검색')]"),
        (By.XPATH, "//a.contains(.,'검색')]"),
        (By.CSS_SELECTOR, "button[class*='search']"),
    ]

    for attempt in range(1, max_retry + 1):
        logging.info(f"🔎 검색 버튼 클릭 시도 {attempt}/{max_retry}")
        _dismiss_common_overlays(driver)
        for by, sel in selectors:
            try:
                el = wait.until(EC.presence_of_element_located((by, sel)))
                if not _is_visible(driver, el):
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.1)
                try:
                    el = wait.until(EC.element_to_be_clickable((by, sel)))
                    ActionChains(driver).move_to_element(el).pause(0.05).click().perform()
                except Exception:
                    driver.execute_script("arguments[0].click();", el)
                time.sleep(0.3)
                return True
            except Exception:
                continue

        # 엔터키 폴백
        try:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ENTER)
            time.sleep(0.3)
            return True
        except Exception:
            pass

        time.sleep(0.4)

    raise TimeoutException("검색 버튼을 찾거나 클릭하지 못했습니다.")


# -------------------------------
# Driver setup
# -------------------------------
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

    # 잡음 로그 억제 (DEPRECATED_ENDPOINT 등)
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    opts.add_argument("--log-level=3")
    opts.add_argument("--window-size=1400,900")
    # 데스크톱 UA 고정(헤드리스 모바일 레이아웃 회피)
    opts.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")

    logging.info("🧩 WebDriver 초기화 (headless=%s, download_dir=%s)", headless, os.path.abspath(download_dir))
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    wait = WebDriverWait(driver, 15)
    return driver, wait


# -------------------------------
# BigKinds flow
# -------------------------------
def download_bigkinds_range(user_id: str, user_pw: str, start_date: str, end_date: str, download_dir: str, headless: bool = True) -> str:
    """지정 날짜 범위(YYYY-MM-DD ~ YYYY-MM-DD)로 BigKinds 엑셀 다운로드."""
    logging.info("🚀 BigKinds 수집 시작: %s ~ %s", start_date, end_date)
    driver, wait = setup_driver(download_dir, headless=headless)
    try:
        # 1) 접속 + 로그인
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

        # 2) 뉴스 검색 페이지
        driver.get("https://www.bigkinds.or.kr/v2/news/index.do")
        time.sleep(2)

        # 3) 언론사: 전국일간지
        select_national_dailies(driver, wait)

        # 4) 통합분류: 정치
        choose_politics_category(driver, wait)

        # 5) 기간: 직접 날짜 입력
        set_date_range_robust(driver, start_date, end_date)
        time.sleep(0.6)

        # 6) 검색 적용 (검색하기)
        click_search_button(driver, wait)
        time.sleep(3)

        # 7) 분석기사 체크 → '적용하기'까지
        ok_tm = apply_analysis_article_filter(driver, wait, max_retry=3)
        if not ok_tm:
            logging.warning("분석기사 체크/적용 보장 실패 (진행은 계속)")

        # 8) STEP3 열고 엑셀 다운로드
        _open_step_panel(driver, "collapse-step-3")
        time.sleep(0.4)
        excel_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.news-download-btn.mobile-excel-download')))
        _scroll_into_view(driver, excel_btn)
        ActionChains(driver).move_to_element(excel_btn).click().perform()

        # 9) 다운로드 완료
        path = wait_for_download_complete(download_dir)
        logging.info("📥 다운로드 완료: %s", path)
        return path

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# -------------------------------
# FAISS partition build/upload
# -------------------------------
def build_and_upload_partition10(df: pd.DataFrame, embeddings: OpenAIEmbeddings, bucket: str, s3_prefix_base: str) -> List[str]:
    logging.info("🧱 파티션 빌드/업로드 시작 (bucket=%s, prefix=%s)", bucket, s3_prefix_base)
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
    logging.info("🗂️ 입력 데이터 컬럼: %s", df.columns.tolist())
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
    used_urls: List[str] = []
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
        used_urls.append(url)

    logging.info("🔢 신규 문서 수: %d (기존:%d)", len(docs), len(existing))
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
            boto3.client("s3").upload_file(os.path.join(work_dir, name), bucket, f"{s3_part_prefix}/{name}")
        logging.info(f"✅ 업로드 완료: s3://{bucket}/{s3_part_prefix}/ (pkl→faiss)")
    else:
        logging.warning("⚠️ 업로드할 인덱스 파일이 없습니다.")

    shutil.rmtree(work_dir, ignore_errors=True)
    return used_urls


# -------------------------------
# Prewarm trigger
# -------------------------------
def trigger_prewarm_partition10(concurrency: int = 3, limit: int = 0, s3_prefix_base: str = "feature_faiss_db_openai_partition/", urls: Optional[List[str]] = None) -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cmd = [os.sys.executable, "-m", "fastapitest.scripts.prewarm_articles"]
    if urls:
        logging.info("🔥 프리워밍 모드: file (urls=%d)", len(urls))
        # Write URLs to a temp file and use file mode to avoid title preloading duplication
        tmp_dir = os.path.join(repo_root, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        url_file = os.path.join(tmp_dir, f"urls_{TARGET_PARTITION}.txt")
        try:
            with open(url_file, "w", encoding="utf-8") as f:
                for u in urls:
                    f.write(u.strip() + "\n")
        except Exception as e:
            logging.error(f"URL 파일 생성 실패: {e}")
            return 1
        cmd += ["--source", "file", "--file", url_file]
    else:
        logging.info("🔥 프리워밍 모드: partitions (prefix 기반)")
        prefix = f"{s3_prefix_base.rstrip('/')}/{TARGET_PARTITION}/"
        cmd += ["--source", "partitions", "--prefix", prefix, "--force-reload"]
    cmd += ["--concurrency", str(concurrency), "--limit", str(limit)]
    logging.info(f"🧭 prewarm 시작: {' '.join(cmd)} (cwd={repo_root})")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if proc.returncode != 0:
        logging.error(f"❌ prewarm 실패(rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    else:
        logging.info(f"✅ prewarm 완료\nSTDOUT:\n{proc.stdout}")
    return proc.returncode


# -------------------------------
# Entrypoint
# -------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    # Suppress noisy openpyxl default style warning
    warnings.filterwarnings(
        "ignore",
        message="Workbook contains no default style, apply openpyxl's default",
        category=UserWarning,
    )

    # 고정 테스트 기간(환경변수로도 오버라이드 가능)
    start_date = os.environ.get("TEST_START_DATE", "2025-08-26")
    end_date = os.environ.get("TEST_END_DATE", "2025-09-01")

    # 필수 환경 변수
    user_id = os.environ.get("BIGKINDS_USER_ID")
    user_pw = os.environ.get("BIGKINDS_USER_PW")
    if not user_id or not user_pw:
        raise SystemExit("환경변수 BIGKINDS_USER_ID/BIGKINDS_USER_PW 필요")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise SystemExit("환경변수 OPENAI_API_KEY 필요")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

    # 선택 환경 변수
    bucket = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
    s3_prefix = os.environ.get("S3_INDEX_PREFIX", "feature_faiss_db_openai_partition/")
    download_dir = os.environ.get("DOWNLOAD_DIR", os.path.abspath("./downloads_test"))
    headless = os.environ.get("HEADLESS", "1") in ("1", "true", "TRUE", "yes", "YES")
    logging.info("🧾 실행 파라미터: bucket=%s, prefix=%s, download_dir=%s, headless=%s", bucket, s3_prefix, download_dir, headless)

    logging.info(f"테스트 범위: {start_date} ~ {end_date} → {TARGET_PARTITION}")

    # 1) BigKinds 다운로드
    downloaded = download_bigkinds_range(user_id, user_pw, start_date, end_date, download_dir, headless=headless)
    logging.info(f"📦 다운로드 파일 경로: {downloaded}")

    # 2) CSV 로컬 저장(옵션, 필수 컬럼만) + 빌드/업로드
    df = pd.read_excel(downloaded)
    df.columns = [str(c).strip().replace("\n", "") for c in df.columns]
    logging.info("🧮 엑셀 로드 완료: rows=%d", len(df))
    write_csv = os.environ.get("WRITE_CSV", "1") in ("1", "true", "TRUE", "yes", "YES")
    if write_csv:
        csv_path = os.path.splitext(downloaded)[0] + ".csv"
        try:
            out = pd.DataFrame()
            for c in REQUIRED_CSV_COLUMNS:
                if c in df.columns:
                    out[c] = df[c]
                else:
                    out[c] = ""
            out = out.fillna("")
            out.to_csv(csv_path, index=False, encoding="utf-8-sig")
            logging.info("📝 CSV 로컬 저장 완료(필수 컬럼): %s", csv_path)
        except Exception as e:
            logging.warning("CSV 저장 실패(계속 진행): %s", e)
    used_urls = build_and_upload_partition10(df, embeddings, bucket, s3_prefix)

    # 3) prewarm 트리거(자동 크롤링)
    rc = trigger_prewarm_partition10(
        concurrency=int(os.environ.get("PREWARM_CONCURRENCY", "3")),
        limit=int(os.environ.get("PREWARM_LIMIT", "0")),
        s3_prefix_base=s3_prefix,
        urls=used_urls,
    )
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
