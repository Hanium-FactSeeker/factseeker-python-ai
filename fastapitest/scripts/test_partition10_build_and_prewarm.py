# test_partition10_build_and_prewarm.py
# -*- coding: utf-8 -*-
import os
import re
import glob
import time
import shutil
import logging
import subprocess
from datetime import datetime
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
        # 최후의 수단: 스킵 (다운로드는 가능하지만 결과 범위가 넓어질 수 있음)
        logging.warning("전국일간지 선택에 실패했습니다. 계속 진행합니다.")


def choose_politics_category(driver, wait):
    """통합분류 탭에서 '정치' 선택."""
    _open_tab(driver, "a[href='#srch-tab3']")
    time.sleep(0.3)
    try:
        node = wait.until(EC.element_to_be_clickable((By.XPATH, '//span[@data-role="display" and text()="정치"]')))
        _scroll_into_view(driver, node)
        node.click()
        time.sleep(0.2)
    except Exception:
        logging.warning("통합분류 '정치' 선택 실패. 계속 진행합니다.")


# -------------------------------
# Date input
# -------------------------------
def _ensure_date_direct_input_mode(driver) -> None:
    """기간 입력을 '직접입력/사용자 지정' 모드로 전환 시도."""
    js = """
    return (function(){
      function clickNode(n){
        try{ n.scrollIntoView({block:'center'}); n.click(); return true; }catch(e){ return false; }
      }
      const labels = ['직접입력','사용자 지정','수동입력','수동','직접'];
      for (const t of labels){
        const nodes = Array.from(document.querySelectorAll('label,button,a,span,div'));
        for (const n of nodes){
          const s = (n.innerText||'').replace(/\\s+/g,' ').trim();
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
    """
    try:
        mode = driver.execute_script(js)
        if mode:
            time.sleep(0.2)
    except Exception:
        pass


def set_date_range_with_events(driver, start_str: str, end_str: str):
    js = """
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
    """
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

    # STEP1 확보 + 직접입력 모드
    _open_tab(driver, "a[href='#srch-tab1']")
    _open_step_panel(driver, "collapse-step-1")
    _ensure_date_direct_input_mode(driver)

    for attempt in range(1, retries + 1):
        # A. JS 주입 + 이벤트
        try:
            js = """
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
