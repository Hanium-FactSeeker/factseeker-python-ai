# core/lambdas.py
import os
import re
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

# --- optional deps (존재 안 해도 터지지 않게) ---
try:
    import aiohttp
except Exception:
    aiohttp = None

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from newspaper import Article
except Exception:
    Article = None

# Selenium은 있는 경우에만 사용
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except Exception:
    webdriver = None
    ChromeOptions = None
    By = None
    WebDriverWait = None
    EC = None

# 유튜브 자막
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except Exception:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------------------
def extract_video_id(youtube_url: str) -> Optional[str]:
    """
    유튜브 URL에서 video_id 추출
    """
    try:
        if "youtu.be/" in youtube_url:
            return youtube_url.rstrip("/").split("/")[-1].split("?")[0]
        parsed = urlparse(youtube_url)
        if parsed.netloc.endswith("youtube.com"):
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
            # /shorts/<id>
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] == "shorts":
                return parts[1]
    except Exception:
        pass
    return None


def fetch_youtube_transcript(youtube_url: str) -> str:
    """
    YouTubeTranscriptApi가 있으면 사용, 없으면 빈 문자열 반환(에러 없이)
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        logger.warning("유효하지 않은 유튜브 URL")
        return ""

    if YouTubeTranscriptApi is None:
        logger.warning("YouTubeTranscriptApi 미설치: transcript 생략")
        return ""

    try:
        # 한/영 우선 시도
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred = None
        for lang_code in ("ko", "en", "a.en"):
            try:
                preferred = transcript_list.find_manually_created_transcript([lang_code])
                break
            except Exception:
                try:
                    preferred = transcript_list.find_transcript([lang_code])
                    break
                except Exception:
                    continue

        if preferred is None:
            preferred = transcript_list.find_transcript(transcript_list._translation_languages or ["en"])

        lines = preferred.fetch()
        text = " ".join([seg.get("text", "").replace("\n", " ").strip() for seg in lines if seg.get("text")])
        return text.strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning("자막 사용 불가/없음")
        return ""
    except Exception as e:
        logger.exception(f"자막 추출 실패: {e}")
        return ""


def clean_news_title(title: str) -> str:
    """
    뉴스 제목 전처리: 괄호/대괄호 태그 제거, 공백 정돈
    """
    if not title:
        return ""
    t = re.sub(r"\[[^\]]*\]", " ", title)
    t = re.sub(r"\([^\)]*\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ------------------------------------------------------------------------------
# Google CSE
# ------------------------------------------------------------------------------
async def search_news_google_cs(query: str) -> List[Dict[str, Any]]:
    """
    Google Custom Search (CSE)로 뉴스 검색.
    환경변수:
      - GOOGLE_CSE_API_KEY
      - GOOGLE_CSE_CX
    둘 중 하나라도 없으면, 에러 없이 [] 반환
    """
    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cx = os.environ.get("GOOGLE_CSE_CX") or os.environ.get("GOOGLE_CSE_CX_ID")
    if not api_key or not cx:
        logger.warning("CSE 키/식별자 미설정 → 검색 생략")
        return []

    if aiohttp is None:
        logger.warning("aiohttp 미설치 → 검색 생략")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": 10,
        "safe": "off",
        "hq": "site:news",
        "gl": "kr",
        "lr": "lang_ko",
    }
    logger.info(f"Google CSE로 뉴스 검색: {query}")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"CSE 응답 오류: HTTP {resp.status}")
                    return []
                data = await resp.json()
                items = data.get("items", []) or []
                # 최소한의 표준화
                out = []
                for it in items:
                    out.append({
                        "title": it.get("title"),
                        "link": it.get("link"),
                        "snippet": it.get("snippet"),
                        "displayLink": it.get("displayLink"),
                    })
                return out
    except Exception as e:
        logger.exception(f"CSE 호출 실패: {e}")
        return []


# ------------------------------------------------------------------------------
# 기사 본문 추출
# ------------------------------------------------------------------------------
def _parse_with_newspaper(url: str) -> str:
    if Article is None:
        return ""
    try:
        art = Article(url, language="ko")
        art.download()
        art.parse()
        text = (art.text or "").strip()
        return text
    except Exception:
        return ""


def _parse_with_requests_bs4(url: str) -> str:
    if requests is None or BeautifulSoup is None:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code != 200:
            return ""
        html = resp.text
        # 언론사별 가장 흔한 본문 컨테이너 후보
        soup = BeautifulSoup(html, "html.parser")
        selectors = [
            "article", ".article", "#article", ".art_body", "#newsEndContents",
            ".news_body", ".article_body", "#articleBody", "#articeBody", ".article_view",
            ".story-news", ".media_end_head", ".view_con", "#CmAdContent", ".tts_body",
            "section#article-view", "div#article-view-content-div", ".article-txt", ".at_contents"
        ]
        best = ""
        for sel in selectors:
            node = soup.select_one(sel)
            if node and node.get_text(strip=True):
                text = node.get_text(separator=" ", strip=True)
                if len(text) > len(best):
                    best = text
        if not best:
            # 전체 텍스트라도
            best = soup.get_text(separator=" ", strip=True)
        return best.strip()
    except Exception as e:
        logger.warning(f"requests+BeautifulSoup 실패: {url} -> {e}")
        return ""


def _selenium_driver() -> Optional["webdriver.Chrome"]:
    """
    가능한 경로로 최대한 시도:
    - 환경변수 CHROMEDRIVER (상대경로 허용)
    - 시스템 PATH
    """
    if webdriver is None or ChromeOptions is None:
        return None
    try:
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,2000")
        bin_path = os.environ.get("CHROME_BINARY")
        if bin_path:
            options.binary_location = bin_path

        driver_path = os.environ.get("CHROMEDRIVER")
        if driver_path:
            return webdriver.Chrome(driver_path, options=options)  # 상대경로 OK

        # 드라이버 경로 미지정 → 시스템 PATH 내 크롬드라이버 사용
        return webdriver.Chrome(options=options)
    except Exception as e:
        logger.warning(f"Selenium 드라이버 초기화 실패: {e}")
        return None


def _parse_with_selenium(url: str, wait_selector: Optional[str] = None) -> str:
    driver = _selenium_driver()
    if driver is None:
        return ""
    try:
        driver.get(url)
        # 선택자 기다리기 (없으면 전체 텍스트)
        text = ""
        try:
            if wait_selector and WebDriverWait is not None and EC is not None and By is not None:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector)))
                node = driver.find_element(By.CSS_SELECTOR, wait_selector)
                text = node.text.strip()
        except Exception:
            pass

        if not text:
            # 흔한 본문 선택자들
            candidates = [
                "article", ".article", "#article", ".art_body", "#newsEndContents",
                ".news_body", ".article_body", "#articleBody", "#articeBody", ".article_view",
                ".view_con", ".news_view", ".content", ".container"
            ]
            for sel in candidates:
                try:
                    if By:
                        nodes = driver.find_elements(By.CSS_SELECTOR, sel)
                        for n in nodes:
                            t = n.text.strip()
                            if len(t) > len(text):
                                text = t
                except Exception:
                    continue

        if not text:
            # 최후의 수단: 전체 페이지 텍스트
            text = driver.find_element(By.TAG_NAME, "body").text.strip() if By else ""

        return text
    except Exception as e:
        logger.warning(f"Selenium 파싱 실패: {url} -> {e}")
        return ""
    finally:
        try:
            driver.quit()
        except Exception:
            pass


async def get_article_text(url: str) -> str:
    """
    비동기 기사 본문 추출:
      1) newspaper
      2) requests+BeautifulSoup
      3) Selenium generic
    사이트/상황 따라 어느 단계에서든 성공하면 즉시 반환
    """
    logger.info(f"📰 비동기로 기사 텍스트 가져오기 시도: {url}")

    # 1) newspaper (스레드 풀)
    loop = asyncio.get_running_loop()
    text = ""
    try:
        text = await loop.run_in_executor(None, _parse_with_newspaper, url)
        if len(text) >= 600:
            logger.info(f"✅ newspaper로 기사 텍스트 추출 완료 ({len(text)}자): {url}")
            return text
        else:
            logger.warning(f"⚠️ newspaper 결과 불충분. BeautifulSoup 선택자 폴백: {url}")
    except Exception as e:
        logger.warning(f"⚠️ newspaper 크롤링 실패. 다음 시도: {url} -> {e}")

    # 2) requests+BeautifulSoup
    try:
        text = await loop.run_in_executor(None, _parse_with_requests_bs4, url)
        if len(text) >= 600:
            logger.info(f"✅ requests+BeautifulSoup (언론사별) 성공 ({len(text)}자): {url}")
            return text
        else:
            logger.warning(f"⚠️ 언론사별 선택자 결과가 너무 짧음. Selenium 폴백.")
    except Exception as e:
        logger.warning(f"⚠️ requests 단계 실패. Selenium 폴백: {url} -> {e}")

    # 3) Selenium (도메인 특화 우선 selector 없음: generic)
    logger.warning(f"⚠️ 최종 폴백: Selenium (Generic) 시도: {url}")
    s_text = await loop.run_in_executor(None, _parse_with_selenium, url, None)
    if len(s_text) >= 300:
        logger.info("✅ Selenium (Generic)으로 최종 본문 추출 성공")
        return s_text
    elif s_text:
        logger.warning("Selenium (Generic): 본문이 너무 짧거나 비어있음.")
    else:
        logger.warning("⚠️ Selenium (Generic)도 불충분/실패.")
    return ""


# ------------------------------------------------------------------------------
# 점수 계산
# ------------------------------------------------------------------------------
def calculate_source_diversity_score(evidence: List[Dict[str, Any]]) -> int:
    """
    동일 도메인/제목 중복 제거한 다양한 출처 개수에 따라 0~5점
    """
    try:
        if not evidence:
            return 0
        unique = set()
        for item in evidence:
            st = (item or {}).get("source_title")
            if st:
                unique.add(st.lower().strip())
                continue
            url = (item or {}).get("url")
            if url:
                try:
                    dom = urlparse(url).netloc
                    if dom:
                        unique.add(dom.lower())
                except Exception:
                    pass
        n = len(unique)
        if n >= 4:
            return 5
        if n == 3:
            return 4
        if n == 2:
            return 3
        if n == 1:
            return 1
        return 0
    except Exception as e:
        logger.exception(f"source diversity 계산 실패: {e}")
        return 0


def calculate_fact_check_confidence(features: Dict[str, Any]) -> int:
    """
    간단 가중치 합산 (0~100)
      - evidence_count: 0~5개 → 0~60점
      - source_diversity: 0~5점 → 0~40점
    """
    try:
        ev = max(0, min(int(features.get("evidence_count", 0)), 5))
        sd = max(0, min(int(features.get("source_diversity", 0)), 5))
        score = round(ev * 12 + sd * 8)  # 5*12 + 5*8 = 100
        return max(0, min(score, 100))
    except Exception:
        return 0
