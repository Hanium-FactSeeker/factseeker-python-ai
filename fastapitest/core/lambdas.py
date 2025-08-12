import os
import re
import asyncio
import logging
from urllib.parse import urlparse, urlunparse, urljoin

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
# 텍스트 정리 & 유틸
# -----------------------------
def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\n){3,}', '\n\n', text)
    text = re.sub(r'Copyright\s*.*무단전재.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*.*All rights reserved.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'저작권자\s*.*무단복제.*', '', text, flags=re.IGNORECASE)
    return text.strip()


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
    t = re.sub(r"<[^>]+>", " ", raw)
    t = re.sub(r"^\s*\[[^\]]{1,12}\]\s*", "", t)
    t = re.sub(rf"^\s*{_MEDIA}\s*[\|\-]\s*", "", t)
    t = re.sub(rf"\s*[\|\-]\s*{_MEDIA}\s*$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 2:
        return raw.strip()
    return t


def extract_video_id(url: str) -> str | None:
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
# Google CSE 뉴스 검색
# -----------------------------
async def search_news_google_cs(query: str):
    logging.info(f"Google CSE로 뉴스 검색: {query}")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    if not google_api_key or not google_cse_id:
        logging.error("Google CSE 키/엔진 ID 누락")
        return []

    def _mk_params(q: str, start: int | None = None):
        p = {
            "key": google_api_key,
            "cx": google_cse_id,
            "q": q,
            "num": 10,
            "hl": "ko",
            "gl": "kr",
            "fields": "items(title,htmlTitle,link,displayLink,snippet),searchInformation(totalResults)"
        }
        if start:
            p["start"] = start
        return p

    url = "https://www.googleapis.com/customsearch/v1"
    async with aiohttp.ClientSession(headers={"Accept": "application/json"}) as session:
        # 첫 페이지
        async with session.get(url, params=_mk_params(query), timeout=15) as resp:
            try:
                data = await resp.json()
            except Exception:
                txt = await resp.text()
                logging.error(f"CSE JSON 파싱 실패(status={resp.status}): {txt[:200]}")
                return []
            if "error" in data:
                msg = data["error"].get("message")
                logging.error(f"Google CSE API 오류: {msg}")
                return []
            items = data.get("items") or []
            if items:
                return items
        # 두 번째 페이지 시도
        async with session.get(url, params=_mk_params(query, start=11), timeout=15) as resp2:
            data2 = await resp2.json()
            return data2.get("items") or []

    # 이곳에 도달할 일은 거의 없음
    return []


# -----------------------------
# 상대경로 → 절대경로 보정 (핵심 패치 1)
# -----------------------------
RELATIVE_HOST_MAP = {
    "/news/newsView.php": "www.seoul.co.kr",  # 서울신문 상대 URL 보정
    # 필요시 추가 매핑
}

def absolutize_url(url: str, base_domain: str | None = None) -> str:
    """
    상대/스킴없는 URL을 절대 URL로 변환한다.
    base_domain(예: 'www.seoul.co.kr')이 주어지면 우선 사용.
    """
    if not url:
        return ""
    p = urlparse(url)

    # 이미 절대 URL
    if p.scheme:
        return url

    # //example.com/.. 형태
    if url.startswith("//"):
        return "https:" + url

    # 검색결과의 displayLink 같은 도메인이 있을 때
    if base_domain:
        if not base_domain.startswith("http"):
            base_domain = "https://" + base_domain
        return urljoin(base_domain, url)

    # 자주 나오는 상대경로 패턴
    for prefix, host in RELATIVE_HOST_MAP.items():
        if url.startswith(prefix):
            return urljoin(f"https://{host}", url)

    # 여기까지도 매칭 안 되면 실패
    return ""


# -----------------------------
# 언론사별 선택자 추출
# -----------------------------
def _extract_article_content_with_selectors(html_content: str, url: str) -> str:
    # 경향신문 구형 URL 보정
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
                'script','style','img','table','figure','figcaption','aside','nav','footer','header',
                'iframe','video','audio','meta','link','form','input','button','select','textarea','svg',
                'canvas','map','area','object','param','embed','source','track','picture','portal','slot',
                'template','noscript','ins','del','bdo','bdi','rp','rt','rtc','ruby','data','time','mark',
                'small','sub','sup','abbr','acronym','address','b','big','blockquote','center','cite','code',
                'dd','dfn','dir','dl','dt','em','font','i','kbd','li','menu','ol','pre','q','s','samp',
                'strike','strong','tt','u','var','ul','h1','h2','h3','h4','h5','h6'
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


# -----------------------------
# Selenium 크롤링 (조선/제네릭)
# -----------------------------
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
        container = soup.select_one('article.layout__article-main section.article-body') or \
                    soup.select_one('article#article-view-content-div')

        if not container:
            logging.warning("Selenium에서도 조선일보 본문 요소를 찾지 못했습니다.")
            return ""

        article_content = []
        for p in container.find_all("p"):
            text = p.get_text(strip=True)
            if text and not any(k in text for k in ["chosun.com", "기자", "Copyright", "무단전재"]):
                article_content.append(text)
        full_text = '\n'.join(article_content)

        if full_text and len(full_text) > 100:
            logging.info("✅ Selenium으로 본문 추출 성공")
            return full_text

        logging.warning("Selenium으로 본문을 찾았으나 내용이 너무 짧거나 비어있습니다.")
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

        selectors = [
            "div.article_content","div#articleBodyContents","div#article_body",
            "div.news_content","article.article_view","div.view_content",
            "div.article-text","div.article-body","div.entry-content",
            "div.contents_area","div.news_view","div.viewContent",
            "article.viewBox2","div.col-main","div.news_bm","section.news_view",
            "div.article-view"
        ]

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ", ".join(selectors))))
        except TimeoutException:
            logging.warning("⚠️ Selenium (Generic): 본문 요소가 10초 내 로드되지 않음. 전체 소스 사용.")

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        container = None
        for sel in selectors:
            container = soup.select_one(sel)
            if container:
                break

        if container:
            for tag in container.find_all([
                'script','style','img','table','figure','figcaption','aside','nav','footer','header',
                'iframe','video','audio','meta','link','form','input','button','select','textarea','svg',
                'canvas','map','area','object','param','embed','source','track','picture','portal','slot',
                'template','noscript','ins','del','bdo','bdi','rp','rt','rtc','ruby','data','time','mark',
                'small','sub','sup','abbr','acronym','address','b','big','blockquote','center','cite','code',
                'dd','dfn','dir','dl','dt','em','font','i','kbd','li','menu','ol','pre','q','s','s','samp',
                'strike','strong','tt','u','var','ul','h1','h2','h3','h4','h5','h6'
            ]):
                tag.decompose()

            for br in container.find_all('br'):
                br.replace_with('\n')

            paragraphs = []
            for content in container.contents:
                if getattr(content, "name", None) == 'p':
                    tx = content.get_text(separator=' ').strip()
                    if tx:
                        paragraphs.append(tx)
                elif isinstance(content, str) and content.strip():
                    paragraphs.append(content.strip())

            full_text = '\n\n'.join(filter(None, paragraphs))
            if full_text and len(full_text) > 100:
                logging.info("✅ Selenium (Generic)으로 본문 추출 성공")
                return full_text
            logging.warning("Selenium (Generic): 본문이 너무 짧거나 비어있음.")
            return ""
        else:
            logging.warning("Selenium (Generic): 본문 요소 미발견. 전체 텍스트 시도.")
            full_text = soup.get_text(separator='\n', strip=True)
            if full_text and len(full_text) > 100:
                logging.info("✅ Selenium (Generic)으로 전체 페이지 텍스트 추출 성공")
                return full_text
            logging.warning("Selenium (Generic): 전체 텍스트도 불충분.")
            return ""
    except (TimeoutException, NoSuchElementException) as e:
        logging.error(f"❌ Selenium (Generic) 요소 탐색 실패/타임아웃: {e}")
        return ""
    except Exception as e:
        logging.exception(f"❌ Selenium (Generic) 오류: {e}")
        return ""
    finally:
        if driver:
            driver.quit()


# -----------------------------
# 기사 본문 추출 오케스트레이터
# -----------------------------
async def get_article_text(url: str, base_domain: str | None = None) -> str:
    """
    (핵심 패치) url이 상대경로여도 base_domain과 매핑을 이용해 절대경로로 만든 뒤 진행.
    """
    fixed = absolutize_url(url, base_domain=base_domain)
    if not fixed:
        logging.warning(f"⚠️ 상대 경로 URL(베이스 없음) 스킵: {url}")
        return ""

    url = fixed
    logging.info(f"📰 비동기로 기사 텍스트 가져오기 시도: {url}")

    parsed_url = urlparse(url)
    clean_url = urlunparse(parsed_url._replace(query='', fragment=''))
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}

    # 조선일보는 바로 Selenium
    if "chosun.com" in parsed_url.netloc:
        logging.info("⭐ 조선일보 기사 감지. Selenium 크롤링 우선.")
        try:
            text = await asyncio.to_thread(extract_chosun_with_selenium, url)
            if text and len(text) > 100:
                return _clean_text(text)
            logging.warning("⚠️ Selenium(조선) 추출 실패/불충분. 종료.")
            return ""
        except Exception as e:
            logging.error(f"❌ Selenium(조선) 실행 오류: {e}")
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
                    logging.warning(f"⚠️ newspaper 결과 불충분. BeautifulSoup 선택자 폴백: {url}")
    except aiohttp.ClientError as e:
        logging.warning(f"⚠️ aiohttp 오류. 다음 시도: {url} -> {e}")
    except Exception as e:
        logging.warning(f"⚠️ newspaper 크롤링 실패. 다음 시도: {url} -> {e}")

    # requests + 선택자
    try:
        logging.warning(f"⚠️ requests+BeautifulSoup (언론사별 선택자) 재시도: {url}")
        resp = requests.get(clean_url, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text

        extracted = _extract_article_content_with_selectors(html, url)
        if extracted and len(extracted) > 100:  # <-- 'and' (오타 방지)
            cleaned = _clean_text(extracted)
            logging.info(f"✅ requests+BeautifulSoup (언론사별) 성공 ({len(cleaned)}자): {url}")
            return cleaned
        else:
            logging.warning("⚠️ 언론사별 선택자 결과가 너무 짧음. Selenium 폴백.")
    except requests.exceptions.RequestException as e:
        logging.warning(f"⚠️ requests 단계 실패. Selenium 폴백: {url} -> {e}")
    except Exception as e:
        logging.warning(f"⚠️ BeautifulSoup 파싱 실패. Selenium 폴백: {url} -> {e}")

    # Selenium (Generic) 최종 폴백
    logging.warning(f"⚠️ 최종 폴백: Selenium (Generic) 시도: {url}")
    try:
        text = await asyncio.to_thread(_extract_generic_with_selenium, url)
        if text and len(text) > 100:
            logging.info("✅ Selenium (Generic)으로 최종 본문 추출 성공")
            return _clean_text(text)
        logging.warning("⚠️ Selenium (Generic)도 불충분/실패.")
        return ""
    except Exception as e:
        logging.error(f"❌ Selenium (Generic) 실행 오류: {e}")
        return ""


# -----------------------------
# YouTube 자막 (yt-dlp + Whisper)
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

    cookies_path = os.getenv("YTDLP_COOKIES", "/home/ubuntu/factseeker-python-ai/fastapitest/cookies.txt")
    outtmpl = f"{vid}.%(ext)s"
    downloaded_paths: list[str] = []

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'cookiefile': cookies_path if os.path.exists(cookies_path) else None,
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

        logging.info(f"✅ 음원 다운로드 완료: {os.path.abspath(audio_file)}")

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
# 점수 계산 유틸(복구 포함)
# -----------------------------
def calculate_fact_check_confidence(criteria_scores: dict) -> int:
    """
    개별 기준(0~5점)의 합으로 0~100% 환산.
    """
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
    """
    제공된 근거(evidence)의 출처 다양성을 0~5점으로 환산.
    - 서로 다른 source_title(또는 도메인) 개수를 셉니다.
    - 1개=1점, 2개=3점, 3개=4점, 4개 이상=5점, 없음=0점
    """
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
