import re
import asyncio
import aiohttp
import hashlib
import os
import requests
from urllib.parse import urlparse, urlunparse, parse_qs
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import logging

# Whisper 관련 라이브러리 추가
from openai import OpenAI
import yt_dlp

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_video_id(url: str):
    """
    유튜브 URL에서 11자리 video_id를 추출합니다.
    다양한 형태의 URL을 처리할 수 있도록 정규식을 개선했습니다.
    """
    try:
        # 모든 가능한 YouTube URL 패턴을 처리하는 정규식
        match = re.search(
            r"(?:v=|/|youtu\.be/|shorts/|embed/)([0-9A-Za-z_-]{11})", url
        )
        if match:
            video_id = match.group(1)
            logging.info(f"[디버깅] URL에서 추출된 video_id: {video_id}")
            return video_id

    except Exception as e:
        logging.exception(f"extract_video_id 오류: {e}")

    logging.error("❌ video_id 추출 실패")
    return None

def fetch_youtube_transcript(video_url):
    """
    EC2 내부 쿠키를 사용하여 yt-dlp로 음원 다운로드 후 Whisper로 자막 추출
    - openai 라이브러리 v1.0.0+에 맞춰 API 호출 방식 수정
    - yt-dlp 다운로드 파일에 확장자를 포함하도록 수정
    """
    video_id = extract_video_id(video_url)
    logging.info(f"[디버깅] 추출된 video_id: {video_id}")
    if not video_id:
        logging.error("유효한 YouTube URL이 아닙니다.")
        return ""
    
    # OpenAI v1.0.0+ API 사용
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logging.error(f"OpenAI 클라이언트 초기화 오류: {e}")
        return ""

    cookies_path = "/home/ubuntu/factseeker-python-ai/fastapitest/cookies.txt"
    # yt-dlp가 다운로드할 파일명을 지정합니다. 파일 확장자를 포함하도록 템플릿 수정.
    temp_audio_file_template = f"{video_id}.%(ext)s"
    downloaded_file_paths = []

    try:
        ydl_opts = {
            'format': 'bestaudio/best', # 최적의 오디오 포맷을 다운로드합니다.
            'outtmpl': temp_audio_file_template,
            'cookiefile': cookies_path,
            'quiet': True,
        }

        logging.info(f"🎬 yt-dlp로 음원 다운로드 시작: {video_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
            # 다운로드된 파일의 실제 경로를 가져옵니다.
            download_info = ydl.sanitize_info(info)
            if 'requested_downloads' in download_info:
                downloaded_file_paths = [d['filepath'] for d in download_info['requested_downloads']]
            elif '_filename' in download_info:
                # 단일 파일 다운로드의 경우
                downloaded_file_paths.append(download_info['_filename'])
            
            if not downloaded_file_paths:
                raise Exception("yt-dlp 다운로드 실패")

            actual_audio_file = downloaded_file_paths[0]

        logging.info(f"✅ 음원 다운로드 완료: {actual_audio_file}")

        with open(actual_audio_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
        logging.info("✅ Whisper API로 자막 추출 완료")
        return transcript.text

    except Exception as e:
        logging.exception(f"yt-dlp 또는 Whisper 처리 중 오류: {e}")
        return ""
    finally:
        # 다운로드된 모든 임시 파일 삭제
        for file_path in downloaded_file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"🗑️ 임시 파일 삭제 완료: {file_path}")


def extract_chosun_with_selenium(url):
    """
    Selenium을 사용하여 조선일보 기사 본문을 추출합니다.
    (로컬 테스트 환경에 맞게 ChromeDriver 경로 설정 필요)
    """
    options = Options()
    options.add_argument("--headless")  # 브라우저 안 띄우고 실행
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")  # GPU 사용 비활성화 (일부 환경에서 필요)
    options.add_argument("--window-size=1920,1080")  # 창 크기 설정
    
    driver = None
    try:
        driver = webdriver.Chrome(options=options)  # PATH에 chromedriver가 있다고 가정
        logging.info(f"🌐 Selenium으로 URL 접속 시도: {url}")
        driver.get(url)
        time.sleep(3)  # JS 렌더링 기다림 (충분히 기다려야 함)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        article_content = []

        article_body_container = soup.select_one('article.layout__article-main section.article-body')
        if not article_body_container:
            article_body_container = soup.select_one('section.article-body')
            
        if article_body_container:
            paragraphs = article_body_container.find_all("p")
            for i, p in enumerate(paragraphs):
                text = p.get_text(strip=True)
                if text and not any(k in text for k in ["chosun.com", "기자", "Copyright", "무단전재"]):
                    article_content.append(text)
            full_text = "\n".join(article_content)
            if full_text and len(full_text) > 100:
                return full_text
            else:
                logging.warning("Selenium으로 본문을 찾았으나 내용이 너무 짧거나 비어있습니다.")
                return ""
        else:
            logging.warning("Selenium에서도 조선일보 본문 요소를 찾지 못했습니다.")
            return ""
    except Exception as e:
        logging.exception(f"Selenium 크롤링 중 오류 발생 ({url}): {e}")
        return ""
    finally:
        if driver:
            driver.quit()


def get_article_text(url):
    """
    주어진 URL에서 뉴스 기사 텍스트를 크롤링하고 파싱합니다.
    조선일보에 대한 특별 처리 로직을 포함합니다.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    parsed_url = urlparse(url)
    clean_url = urlunparse(parsed_url._replace(query='', fragment=''))

    # 1. 조선일보에 대한 Selenium 특별 처리
    if "chosun.com" in parsed_url.netloc:
        selenium_text = extract_chosun_with_selenium(clean_url)
        if selenium_text and len(selenium_text) > 300:
            return selenium_text
        else:
            logging.warning(f"Selenium으로 조선일보 본문 추출 실패 또는 내용 부족. newspaper 및 requests 폴백 시도.")

    # 2. newspaper 라이브러리를 사용한 일반적인 크롤링 시도
    try:
        article = Article(clean_url, language="ko", headers=headers)
        article.download()
        article.parse()
        newspaper_text = article.text
        
        if newspaper_text and len(newspaper_text) > 300:
            return newspaper_text
        else:
            logging.warning(f"newspaper 크롤링 결과가 불충분함 ({len(newspaper_text)}자). requests + BeautifulSoup 폴백 시도: {clean_url}")

    except Exception as e:
        logging.exception(f"newspaper 크롤링 실패 ({clean_url}): {e}. requests + BeautifulSoup 폴백 시도.")
    
    # 3. requests와 BeautifulSoup를 사용한 마지막 폴백
    try:
        response = requests.get(clean_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = []

        # 조선일보이거나 다른 언론사라도 본문 선택자 로직 재시도
        if "chosun.com" in parsed_url.netloc:
            article_body_section = soup.select_one('section.article-body')  
            if article_body_section:
                paragraphs = article_body_section.find_all('p')  
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and not any(keyword in text for keyword in ["chosun.com", "기자", "Copyright", "무단전재"]):
                        article_content.append(text)
                
        # 일반적인 언론사 본문 선택자
        else:
            body_elements = soup.select('div.article_content, div#articleBodyContents, div#article_body, div.news_content, article.article_view, div.view_content')
            for elem in body_elements:
                text = elem.get_text(separator='\n', strip=True)
                if text:
                    article_content.append(text)
        
        full_text = '\n'.join([c for c in article_content if c]) # None 값 제거
        if len(full_text) > 300:
            return full_text
        else:
            logging.warning(f"requests+BeautifulSoup로 본문 내용이 너무 짧음 ({len(full_text)}자): {clean_url}")
            return ""

    except requests.exceptions.RequestException as e:
        logging.exception(f"HTTP 요청 중 오류 발생 ({clean_url}): {e}")
        return ""
    except Exception as e:
        logging.exception(f"BeautifulSoup 텍스트 추출 중 오류 발생 ({clean_url}): {e}")
        return ""


def clean_news_title(title):
    """
    뉴스 제목에서 언론사명, 슬로건, 특수 태그, 불필요한 기호 등을 제거하여 정제합니다.
    """
    patterns_to_remove = [
        r'대한민국 오후를 여는 유일석간 문화일보',  # 문화일보 슬로건
        r'\| 문화일보', r'문화일보',  # 문화일보 관련
        r'\| 중앙일보', r'중앙일보',  # 중앙일보 관련
        r'\| 경향신문', r'경향신문',  # 경향신문 관련
        r'머니투데이', r'MBN', r'연합뉴스', r'SBS 뉴스', r'MBC 뉴스', r'KBS 뉴스',
        r'동아일보', r'조선일보', r'한겨레', r'국민일보', r'서울신문', r'세계일보',
        r'노컷뉴스', r'헤럴드경제', r'매일경제', r'한국경제', r'아시아경제',
        r'YTN', r'JTBC', r'TV조선', r'채널A', r'데일리안', r'뉴시스',
        r'뉴스1', r'연합뉴스TV', r'뉴스핌', r'이데일리', r'파이낸셜뉴스',
        r'아주경제', r'UPI뉴스', r'ZUM 뉴스', r'네이트 뉴스', r'다음 뉴스',
    ]

    patterns_to_remove_regex = [
        r'\[.*?\]',  # 예: [단독], [속보], [종합], [사진], [영상], [팩트체크]
        r'\(.*?\)',  # 예: (서울), (종합), (영상)
        r'\{.*?\}',  # 예: {뉴스초점}
        r'<[^>]+>',  # HTML 태그 잔여물
        r'\[\s*\w+\s*\]',  # 공백 포함 대괄호 태그
    ]

    symbols_to_remove = [
        r'\|', r'\:', r'\_', r'\-', r'\+', r'=', r'/', r'\\'  # |, :, _, -, +, =, /, \ 등
    ]

    cleaned_title = title
    
    for pattern in patterns_to_remove:
        cleaned_title = re.sub(re.escape(pattern), '', cleaned_title, flags=re.IGNORECASE).strip()
    
    for pattern_regex in patterns_to_remove_regex + symbols_to_remove:
        cleaned_title = re.sub(pattern_regex, '', cleaned_title).strip()
    
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
    
    return cleaned_title

async def search_news_google_cs(query):
    logging.info(f"Google CSE로 뉴스 검색: {query}")
    """Google Custom Search API를 사용하여 뉴스를 검색합니다."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": query,
        "num": 10,
        "hl": "ko",
        "gl": "kr"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if "error" in data:
                logging.error(f"Google CSE API 오류: {data['error']['message']}")
                if "quotaExceeded" in data['error']['message']:
                    logging.warning("경고: Google CSE API 할당량(Quota)이 초과되었을 수 있습니다. Google Cloud Console에서 확인해주세요.")
                return []
            return data.get("items", [])

def calculate_fact_check_confidence(criteria_scores):
    """
    팩트체크 결과를 바탕으로 신뢰도를 0%에서 100%까지 계산합니다.
    """
    if not criteria_scores:
        return 0

    total_possible_score = 0
    total_actual_score = 0

    for criterion, score in criteria_scores.items():
        if not (0 <= score <= 5):
            logging.error(f"오류: '{criterion}'의 점수 '{score}'가 유효한 범위(0-5)를 벗어났습니다. 신뢰도를 계산할 수 없습니다.")
            return 0
        total_possible_score += 5
        total_actual_score += score

    if total_actual_score == 0 and total_possible_score > 0:
        return 0

    if total_possible_score == 0:
        return 0

    confidence_percentage = (total_actual_score / total_possible_score) * 100

    return max(0, min(100, round(confidence_percentage)))

def calculate_source_diversity_score(evidence):
    """
    제공된 근거(evidence)의 출처 다양성을 계산하여 0-5점 사이의 점수를 반환합니다.
    """
    if not evidence:
        return 0
    
    unique_sources = set()
    for item in evidence:
        # source_title이 있다면 이를 사용 (소문자로 변환하여 대소문자 구분 없이 처리)
        if item.get("source_title"):
            unique_sources.add(item["source_title"].lower())
        # source_title이 없으면 URL의 도메인을 사용
        elif item.get("url"):
            try:
                domain = urlparse(item["url"]).netloc
                if domain:
                    unique_sources.add(domain.lower())
            except Exception:
                # URL 파싱 오류 시 무시
                pass

    num_unique_sources = len(unique_sources)

    if num_unique_sources >= 4:
        return 5
    elif num_unique_sources == 3:
        return 4
    elif num_unique_sources == 2:
        return 3
    elif num_unique_sources == 1:
        return 1  # 단일 출처라도 다양성 점수는 낮게 책정
    else:
        return 0  # 출처가 없거나 유효하지 않은 경우
