import os
import re
import time
import json
import asyncio
import logging
import hashlib
from urllib.parse import urlparse, urlunparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from newspaper import Article

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from openai import OpenAI
import yt_dlp


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------
# Utilities
# -----------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\n){3,}', '\n\n', text)
    text = re.sub(r'Copyright\s*.*무단전재.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*.*All rights reserved.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'저작권자\s*.*무단복제.*', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_video_id(url: str):
    try:
        m = re.search(r"(?:v=|/|youtu\.be/|shorts/|embed/)([0-9A-Za-z_-]{11})", url)
        if m:
            vid = m.group(1)
            logging.info(f"[디버깅] URL에서 추출된 video_id: {vid}")
            return vid
    except Exception as e:
        logging.error(f"유튜브 video_id 추출 실패: {e}")
    return None


# -----------------------------
# Title cleaner (완화 + 세이프가드)
# -----------------------------
_MEDIA = (
    r"(문화일보|중앙일보|경향신문|머니투데이|MBN|연합뉴스|SBS 뉴스|MBC 뉴스|KBS 뉴스|동아일보|"
    r"조선일보|한겨레|국민일보|서울신문|세계일보|노컷뉴스|헤럴드경제|매일경제|한국경제|아시아경제|"
    r"YTN|JTBC|TV조선|채널A|데일리안|뉴시스|뉴스1|연합뉴스TV|뉴스핌|이데일리|파이낸셜뉴스|"
    r"아주경제|UPI뉴스|ZUM 뉴스|네이트 뉴스|다음 뉴스)"
)

def clean_news_title(title: str) -> str:
    if not title:
        return ""
    raw = title

    # HTML 태그 제거
    t = re.sub(r"<[^>]+>", " ", raw)

    # 시작부의 짧은 대괄호 태그 제거 (예: [단독], [속보])
    t = re.sub(r"^\s*\[[^\]]{1,12}\]\s*", "", t)

    # 양끝의 언론사 표기 제거 (| 또는 - 로 구분된 경우)
    t = re.sub(rf"^\s*{_MEDIA}\s*[\|\-]\s*", "", t)
    t = re.sub(rf"\s*[\|\-]\s*{_MEDIA}\s*$", "", t)

    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()

    # 세이프가드: 너무 짧아지면 원제목 유지
    if len(t) < 2:
        return raw.strip()
    return t


# -----------------------------
# CSE 검색 (다운시프트 포함)
# -----------------------------
async def search_news_google_cs(query: str):
    logging.info(f"Google CSE로 뉴스 검색: {query}")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    if not google_api_key or not google_cse_id:
        logging.error("Google CSE 키/엔진 ID 누락")
        return []

    def _simplify_ko(q: str) -> str:
        stop = ["내부", "기류", "증언", "나오고", "있다", "있다는", "하려", "움직임이", "위원장", "발언을", "제지했다"]
        tokens = re.split(r"\s+", q)
        tokens = [t for t in tokens if t and t not in stop]
        return " ".join(tokens) or q

    def _mk_params(q: str, or_terms: str | None = None, start: int | None = None):
        p = {
            "key": google_api_key,
            "cx": google_cse_id,
            "q": q,
            "num": 10,
            "hl": "ko",
            "gl": "kr",
            "fields": "items(title,htmlTitle,link,displayLink,snippet),searchInformation(totalResults)"
        }
        if or_terms:
            p["orTerms"] = or_terms
        if start:
            p["start"] = start
        return p

    url = "https://www.googleapis.com/customsearch/v1"

    async def _cse_fetch(session: aiohttp.ClientSession, params: dict, max_retries: int = 2):
        delay = 0.5
        for attempt in range(max_retries + 1):
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                    data = await resp.json()
                    if "error" in data:
                        msg = data["error"].get("message")
                        logging.error(f"Google CSE API 오류: {msg}")
                        if msg and "quota" in msg.lower():
                            logging.warning("경고: CSE 쿼터 초과 가능성")
                        return []
                    return data.get("items") or []
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logging.warning(f"CSE 요청 실패(재시도 {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    return []

    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        # 1차: 원 쿼리 (+2페이지까지)
        for attempt in range(1, 3):
            params = _mk_params(query, start=1 + (attempt - 1) * 10 if attempt > 1 else None)
            items = await _cse_fetch(session, params)
            if items:
                for it in items[:5]:
                    logging.debug(f"[CSE] raw_title={it.get('title')!r}")
                return items

        # 2차: 방송3사 OR 확장
        if re.search(r"\b(KBS|MBC|EBS)\b", query, flags=re.IGNORECASE):
            params = _mk_params(_simplify_ko(query), or_terms="KBS MBC EBS")
            items = await _cse_fetch(session, params)
            if items:
                for it in items[:5]:
                    logging.debug(f"[CSE] raw_title={it.get('title')!r}")
                return items

        # 3차: 강제 축약
        simplified = _simplify_ko(query)
        if simplified != query:
            params = _mk_params(simplified)
            items = await _cse_fetch(session, params)
            if items:
                for it in items[:5]:
                    logging.debug(f"[CSE] raw_title={it.get('title')!r}")
                return items

    logging.warning("📭 CSE 결과 0건 (모든 다운시프트 실패)")
    return []


# -----------------------------
# Article extraction (언론사 선택자 + Selenium)
# -----------------------------
def _extract_article_content_with_selectors(html_content: str, url: str) -> str:
    # 경향신문 구형 URL → 신형 전환
    if "news.khan.co.kr/kh_news/khan_art_view.html" in url:
        m = re.search(r'artid=(\d+)', url)
        if m:
            art_id = m.group(1)
            url = f"https://www.khan.co.kr/article/{art_id}"

    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    selectors = {
        "hani.co.kr": "div.article-text p.text",
        "khan.co.kr": "#articleBody p, div#articleBody p, #articleBody p.content_text",
        "segye.com": "article.viewBox2",
        "hankookilbo.com": "div.col-main p.read",
        "asiatoday.co.kr": "div.news_bm",
        "seoul.co.kr": "div.viewContent",
        "donga.com": "section.news_view",
        "naeil.com": "div.article-view p",
    }

    selector = None
    for dom, sel in selectors.items():
        if dom in domain:
            selector = sel
            break
    if not selector:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')
    article_elements = []

    if any(d in domain for d in ["hani.co.kr", "khan.co.kr", "hankookilbo.com", "naeil.com"]):
        elements = soup.select(selector)
        for p_tag in elements:
            for br in p_tag.find_all('br'):
                br.replace_with('\n')
            text = p_tag.get_text(separator=' ').strip()
            if text:
                article_elements.append(text)
    elif any(d in domain for d in ["segye.com", "asiatoday.co.kr", "seoul.co.kr", "donga.com"]):
        main_content_div = soup.select_one(selector)
        if main_content_div:
            for tag in main_content_div.find_all([
                'script', 'style', 'img', 'table', 'figure', 'figcaption', 'aside', 'nav', 'footer', 'header',
                'iframe', 'video', 'audio', 'meta', 'link', 'form', 'input', 'button', 'select', 'textarea', 'svg',
                'canvas', 'map', 'area', 'object', 'param', 'embed', 'source', 'track', 'picture', 'portal', 'slot',
                'template', 'noscript', 'ins', 'del', 'bdo', 'bdi', 'rp', 'rt', 'rtc', 'ruby', 'data', 'time', 'mark',
                'small', 'sub', 'sup', 'abbr', 'acronym', 'address', 'b', 'big', 'blockquote', 'center', 'cite', 'code',
                'dd', 'dfn', 'dir', 'dl', 'dt', 'em', 'font', 'i', 'kbd', 'li', 'menu', 'ol', 'pre', 'q', 's', 'samp',
                'strike', 'strong', 'tt', 'u', 'var', 'ul', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
            ]):
                tag.decompose()

            for br in main_content_div.find_all('br'):
                br.replace_with('\n')

            paragraphs = []
            for content in main_content_div.contents:
                if getattr(content, "name", None) == 'p':
                    text = content.get_text(separator=' ').strip()
                    if text:
                        paragraphs.append(text)
                elif isinstance(content, str) and content.strip():
                    paragraphs.append(content.strip())

            article_elements = paragraphs

    full_text = '\n\n'.join(filter(None, article_elements))
    return full_text


def extract_chosun_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    try:
        logging.info(f"📰 Selenium으로 크롤링 시도: {url}")
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        article_selector = "article#article-view-content-div, article.layout__article-main section.article-body"
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, article_selector)))

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        article_content = []
        container = soup.select_one('article.layout__article-main section.article-body') or \
                    soup.select_one('article#article-view-content-div')

        if container:
            paragraphs = container.find_all("p")
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and not any(k in text for k in ["chosun.com", "기자", "Copyright", "무단전재"]):
                    article_content.append(text)
            full_text = '\n'.join(article_content)

            if full_text and len(full_text) > 100:
                logging.info("✅ Selenium으로 본문 추출 성공")
                return full_text
            else:
                logging.warning("Selenium으로 본문을 찾았으나 내용이 너무 짧거나 비어있습니다.")
                return ""
        else:
            logging.warning("Selenium에서도 조선일보 본문 요소를 찾지 못했습니다.")
            return ""
    except (TimeoutException, NoSuchElementException) as e:
        logging.error(f"❌ Selenium 크롤링 중 요소 탐색 실패 또는 타임아웃: {e}")
        return ""
    except Exception as e:
        logging.exception(f"❌ Selenium 크롤링 중 오류 발생: {e}")
        return ""
    finally:
        if driver:
            driver.quit()


def _extract_generic_with_selenium(url: str) -> str:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    try:
        logging.info(f"📰 Selenium (Generic)으로 크롤링 시도: {url}")
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        wait = WebDriverWait(driver, 10)

        generic_article_selectors = [
            "div.article_content", "div#articleBodyContents", "div#article_body",
            "div.news_content", "article.article_view", "div.view_content",
            "div.article-text", "div.article-body", "div.entry-content",
            "div.contents_area", "div.news_view", "div.viewContent",
            "article.viewBox2", "div.col-main", "div.news_bm", "section.news_view",
            "div.article-view"
        ]

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ", ".join(generic_article_selectors))))
        except TimeoutException:
            logging.warning("⚠️ Selenium (Generic): 본문 요소가 10초 내에 로드되지 않았습니다. 전체 페이지 소스 사용.")

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        container = None
        for selector in generic_article_selectors:
            container = soup.select_one(selector)
            if container:
                break

        if container:
            for tag in container.find_all([
                'script', 'style', 'img', 'table', 'figure', 'figcaption', 'aside', 'nav', 'footer', 'header',
                'iframe', 'video', 'audio', 'meta', 'link', 'form', 'input', 'button', 'select', 'textarea', 'svg',
                'canvas', 'map', 'area', 'object', 'param', 'embed', 'source', 'track', 'picture', 'portal', 'slot',
                'template', 'noscript', 'ins', 'del', 'bdo', 'bdi', 'rp', 'rt', 'rtc', 'ruby', 'data', 'time', 'mark',
                'small', 'sub', 'sup', 'abbr', 'acronym', 'address', 'b', 'big', 'blockquote', 'center', 'cite', 'code',
                'dd', 'dfn', 'dir', 'dl', 'dt', 'em', 'font', 'i', 'kbd', 'li', 'menu', 'ol', 'pre', 'q', 's', 'samp',
                'strike', 'strong', 'tt', 'u', 'var', 'ul', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
            ]):
                tag.decompose()

            for br in container.find_all('br'):
                br.replace_with('\n')

            paragraphs = []
            for content in container.contents:
                if getattr(content, "name", None) == 'p':
                    text = content.get_text(separator=' ').strip()
                    if text:
                        paragraphs.append(text)
                elif isinstance(content, str) and content.strip():
                    paragraphs.append(content.strip())

            full_text = '\n\n'.join(filter(None, paragraphs))
            if full_text and len(full_text) > 100:
                logging.info("✅ Selenium (Generic)으로 본문 추출 성공")
                return full_text
            else:
                logging.warning("Selenium (Generic)으로 본문을 찾았으나 내용이 너무 짧거나 비어있습니다.")
                return ""
        else:
            logging.warning("Selenium (Generic): 특정 본문 요소를 찾지 못했습니다. 페이지 전체 텍스트를 시도합니다.")
            full_text = soup.get_text(separator='\n', strip=True)
            if full_text and len(full_text) > 100:
                logging.info("✅ Selenium (Generic)으로 전체 페이지 텍스트 추출 성공")
                return full_text
            else:
                logging.warning("Selenium (Generic)으로 전체 페이지 텍스트도 너무 짧거나 비어있습니다.")
                return ""
    except (TimeoutException, NoSuchElementException) as e:
        logging.error(f"❌ Selenium (Generic) 크롤링 중 요소 탐색 실패 또는 타임아웃: {e}")
        return ""
    except Exception as e:
        logging.exception(f"❌ Selenium (Generic) 크롤링 중 오류 발생: {e}")
        return ""
    finally:
        if driver:
            driver.quit()


# -----------------------------
# Async article fetch orchestrator
# -----------------------------
async def get_article_text(url: str) -> str:
    logging.info(f"📰 비동기로 기사 텍스트 가져오기 시도: {url}")
    parsed_url = urlparse(url)
    clean_url = urlunparse(parsed_url._replace(query='', fragment=''))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    # 조선일보 우선 시도
    if "chosun.com" in parsed_url.netloc:
        logging.info("⭐ 조선일보 기사 감지. Selenium 크롤링을 먼저 시도합니다.")
        try:
            text = await asyncio.to_thread(extract_chosun_with_selenium, url)
            if text and len(text) > 100:
                return _clean_text(text)
            else:
                logging.warning("⚠️ Selenium으로 본문 추출 실패 또는 내용이 불충분합니다. 다음 방법을 시도하지 않습니다.")
                return ""
        except Exception as e:
            logging.error(f"❌ asyncio.to_thread Selenium 실행 중 오류: {e}")
            return ""

    # aiohttp + newspaper
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(clean_url, timeout=30) as response:
                response.raise_for_status()
                html_content = await response.text()

                article = Article(clean_url, language='ko')
                article.download(input_html=html_content)
                article.parse()

                if article.text and len(article.text) > 300:
                    logging.info(f"✅ newspaper로 기사 텍스트 추출 완료 ({len(article.text)}자): {url}")
                    return _clean_text(article.text)
                else:
                    logging.warning(f"⚠️ newspaper 크롤링 결과가 불충분함. BeautifulSoup 폴백 시도: {url}")
    except aiohttp.ClientError as e:
        logging.warning(f"⚠️ aiohttp 클라이언트 오류 발생. 다음 방법 시도: {url} -> {e}")
    except Exception as e:
        logging.warning(f"⚠️ newspaper 크롤링 실패. 다음 방법 시도: {url} -> {e}")

    # requests + BeautifulSoup (언론사별 선택자)
    try:
        logging.warning(f"⚠️ requests+BeautifulSoup으로 본문 추출 재시도 (언론사별 선택자 적용): {url}")
        response = requests.get(clean_url, headers=headers, timeout=30)
        response.raise_for_status()
        html_content = response.text

        extracted = _extract_article_content_with_selectors(html_content, url)
        if extracted and len(extracted) > 100:
            cleaned_final_text = _clean_text(extracted)
            logging.info(f"✅ requests+BeautifulSoup (언론사별)으로 본문 추출 완료 ({len(cleaned_final_text)}자): {url}")
            return cleaned_final_text
        else:
            logging.warning("⚠️ requests+BeautifulSoup (언론사별)로 본문 내용이 너무 짧음. 최종 폴백 시도.")
    except requests.exceptions.RequestException as e:
        logging.warning(f"⚠️ requests+BeautifulSoup (언론사별) 실패. 최종 폴백 시도: {url} -> {e}")
    except Exception as e:
        logging.warning(f"⚠️ BeautifulSoup 파싱 실패 (언론사별). 최종 폴백 시도: {url} -> {e}")

    # 최종 폴백: Selenium (Generic)
    logging.warning(f"⚠️ 모든 방법으로 본문 추출 실패. 최종 폴백: Selenium (Generic) 시도: {url}")
    try:
        text = await asyncio.to_thread(_extract_generic_with_selenium, url)
        if text and len(text) > 100:
            logging.info("✅ Selenium (Generic)으로 최종 본문 추출 성공")
            return _clean_text(text)
        else:
            logging.warning("⚠️ Selenium (Generic)으로도 본문 추출 실패 또는 내용이 불충분합니다.")
            return ""
    except Exception as e:
        logging.error(f"❌ asyncio.to_thread Selenium (Generic) 실행 중 오류: {e}")
        return ""


# -----------------------------
# YouTube transcript (yt-dlp + Whisper)
# -----------------------------
def fetch_youtube_transcript(video_url: str) -> str:
    vid = extract_video_id(video_url)
    logging.info(f"[디버깅] 추출된 video_id: {vid}")
    if not vid:
        logging.error("유효한 YouTube URL이 아닙니다.")
        return ""

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 오류: {e}")
        return ""

    cookies_path = "/home/ubuntu/factseeker-python-ai/fastapitest/cookies.txt"
    outtmpl = f"{vid}.%(ext)s"
    downloaded_paths: list[str] = []

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'cookiefile': cookies_path,
            'quiet': True,
        }

        logging.info(f"🎬 yt-dlp로 음원 다운로드 시작: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            sinfo = ydl.sanitize_info(info)
            if 'requested_downloads' in sinfo:
                downloaded_paths = [d['filepath'] for d in sinfo['requested_downloads']]
            elif '_filename' in sinfo:
                downloaded_paths.append(sinfo['_filename'])

            if not downloaded_paths:
                raise RuntimeError("yt-dlp 다운로드 실패")

            audio_file = downloaded_paths[0]

        logging.info(f"✅ 음원 다운로드 완료: {audio_file}")

        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko"
            )
        logging.info("✅ Whisper API로 자막 추출 완료")
        return transcript.text or ""

    except Exception as e:
        logging.exception(f"yt-dlp 또는 Whisper 처리 중 오류: {e}")
        return ""
    finally:
        for p in downloaded_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                    logging.info(f"🗑️ 임시 파일 삭제 완료: {p}")
            except Exception:
                pass


# -----------------------------
# Scoring helpers
# -----------------------------
def calculate_fact_check_confidence(criteria_scores: dict) -> int:
    if not criteria_scores:
        return 0
    total_possible = 0
    total_actual = 0
    for _, score in criteria_scores.items():
        if not (0 <= score <= 5):
            logging.error(f"오류: 점수 '{score}'가 유효 범위(0-5)를 벗어남")
            return 0
        total_possible += 5
        total_actual += score
    if total_possible == 0:
        return 0
    pct = (total_actual / total_possible) * 100
    return max(0, min(100, round(pct)))


def calculate_source_diversity_score(evidence: list[dict]) -> int:
    if not evidence:
        return 0
    unique = set()
    for item in evidence:
        st = item.get("source_title")
        if st:
            unique.add(st.lower())
            continue
        url = item.get("url")
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
