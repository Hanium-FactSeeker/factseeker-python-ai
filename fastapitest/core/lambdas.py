import re
import asyncio
import aiohttp
import hashlib
import os
import requests
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import logging

# Whisper 관련 라이브러리 추가
import openai
import yt_dlp

# Google API Key 및 CSE ID는 main.py에서 로드되므로, 여기서 직접 참조하지 않습니다.
# 대신, 필요한 경우 함수 인자로 받거나 전역 설정 객체로 관리할 수 있습니다.
# 여기서는 get_article_text가 외부 의존성을 가지지 않도록 수정합니다.


def extract_video_id(url):
    """YouTube URL에서 비디오 ID를 추출합니다."""
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None


async def fetch_youtube_transcript(video_id):
    """
    yt-dlp와 OpenAI Whisper API를 사용하여 YouTube 비디오의 자막을 생성합니다.
    """
    if not video_id:
        logging.error("비디오 ID가 유효하지 않습니다.")
        return ""

    # OpenAI API 키 설정
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return ""

    temp_audio_file = None
    try:
        # 1. yt-dlp를 사용하여 YouTube 영상의 음원 다운로드
        audio_filename = f"{video_id}.mp3"
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': audio_filename,
            'cookiefile': '/home/ubuntu/factseeker-python-ai/fastapitest/cookies.txt', 
            'no_check_certificate': True,  # ✅ EC2에 업로드한 쿠키 파일 사용
            'quiet': True,
        }
        
        logging.info(f"🎶 yt-dlp로 YouTube 음원 다운로드 시작: {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=True)
        
        temp_audio_file = audio_filename
        logging.info(f"✅ 음원 다운로드 완료: {temp_audio_file}")

        # 2. Whisper API 호출
        with open(temp_audio_file, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                language="ko"  # 한국어 모델 지정
            )
        
        logging.info("✅ Whisper API로 자막 생성 완료")
        return transcript.text

    except yt_dlp.utils.DownloadError as e:
        logging.exception(f"yt-dlp 다운로드 실패: {e}")
        return ""
    except openai.error.OpenAIError as e:
        logging.exception(f"Whisper API 호출 실패: {e}")
        return ""
    except Exception as e:
        logging.exception(f"YouTube 음원 처리 중 예상치 못한 오류 발생: {e}")
        return ""
    finally:
        # 3. 임시 파일 삭제
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            logging.info(f"🗑️ 임시 파일 삭제 완료: {temp_audio_file}")


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
    
    # --- 중요: 로컬 환경에 맞는 ChromeDriver 경로 설정 ---
    # chromedriver_path = "/path/to/your/chromedriver" # 실제 경로로 변경
    # driver = webdriver.Chrome(executable_path=chromedriver_path, options=options)
    # 로컬에서 테스트할 때는 PATH에 chromedriver가 있거나, 위에 주석처리된 라인의 주석을 풀고 경로를 지정해주세요.
    # 아니면, webdriver_manager 라이브러리를 사용해 자동 설치할 수 있습니다.
    # from webdriver_manager.chrome import ChromeDriverManager
    # driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    # ---------------------------------------------------

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
    is_chosun = "chosun.com" in parsed_url.netloc
    clean_url = urlunparse(parsed_url._replace(query='', fragment=''))

    if is_chosun:
        selenium_text = extract_chosun_with_selenium(clean_url)
        if selenium_text and len(selenium_text) > 300:
            return selenium_text
        else:
            logging.warning(f"Selenium으로 조선일보 본문 추출 실패 또는 내용 부족. newspaper 및 requests 폴백 시도.")

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
    
    try:
        response = requests.get(clean_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = []

        if is_chosun:
            article_body_section = soup.select_one('section.article-body')  
            if article_body_section:
                paragraphs = article_body_section.find_all('p')  
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and not any(keyword in text for keyword in ["chosun.com", "기자", "Copyright", "무단전재"]):
                        article_content.append(text)
                if article_content:
                    full_text = '\n'.join(article_content)
                    if len(full_text) > 300:
                        return full_text
                    else:
                        logging.warning(f"requests+BeautifulSoup로 조선일보 본문 추출했으나 내용이 짧음 ({len(full_text)}자).")
                else:
                    logging.warning(f"requests+BeautifulSoup로도 조선일보 본문 요소 ('section.article-body')를 찾을 수 없음.")
            else:
                body_elements = soup.select('div.article_content, div#articleBodyContents, div#article_body, div.news_content, article.article_view, div.view_content')
                for elem in body_elements:
                    text = elem.get_text(separator='\n', strip=True)
                    if text:
                        article_content.append(text)
            
            full_text = '\n'.join(article_content)
            if len(full_text) > 300:
                return full_text
            else:
                logging.warning(f"requests+BeautifulSoup로 일반 기사 본문 내용이 너무 짧음 ({len(full_text)}자): {clean_url}")
            
            return full_text if len(full_text) > 300 else ""

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
                from urllib.parse import urlparse
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