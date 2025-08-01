import re
import asyncio
import aiohttp
import hashlib
import os
import requests
import json
import logging
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import boto3

# 🔁 YouTube 자막 추출 Lambda 연동 버전
async def fetch_youtube_transcript(youtube_url):
    lambda_client = boto3.client("lambda", region_name="ap-northeast-2")
    payload = {"youtube_url": youtube_url}

    try:
        response = lambda_client.invoke(
            FunctionName="YoutubeTranscriptExtractor",  # 여기를 실제 Lambda 이름으로 교체
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        result = json.loads(response["Payload"].read())
        if result.get("statusCode") != 200:
            logging.error(f"Lambda 실패: {result.get('body')}")
            return ""
        body = json.loads(result["body"])
        return body.get("transcript", "")
    except Exception as e:
        logging.exception(f"Lambda 자막 호출 실패: {e}")
        return ""

def extract_video_id(url):
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    return match.group(1) if match else None

def extract_chosun_with_selenium(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        logging.info(f"🌐 Selenium으로 URL 접속 시도: {url}")
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        article_content = []
        article_body_container = soup.select_one('article.layout__article-main section.article-body') or soup.select_one('section.article-body')
        if article_body_container:
            paragraphs = article_body_container.find_all("p")
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and not any(k in text for k in ["chosun.com", "기자", "Copyright", "무단전재"]):
                    article_content.append(text)
            full_text = "\n".join(article_content)
            return full_text if full_text and len(full_text) > 100 else ""
        else:
            return ""
    except Exception as e:
        logging.exception(f"Selenium 크롤링 중 오류 발생 ({url}): {e}")
        return ""
    finally:
        if driver:
            driver.quit()

def get_article_text(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    parsed_url = urlparse(url)
    is_chosun = "chosun.com" in parsed_url.netloc
    clean_url = urlunparse(parsed_url._replace(query='', fragment=''))
    if is_chosun:
        selenium_text = extract_chosun_with_selenium(clean_url)
        if selenium_text and len(selenium_text) > 300:
            return selenium_text
    try:
        article = Article(clean_url, language="ko", headers=headers)
        article.download()
        article.parse()
        if article.text and len(article.text) > 300:
            return article.text
    except Exception:
        pass
    try:
        response = requests.get(clean_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        article_content = []
        if is_chosun:
            article_body_section = soup.select_one('section.article-body')
            if article_body_section:
                for p in article_body_section.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and not any(keyword in text for keyword in ["chosun.com", "기자", "Copyright", "무단전재"]):
                        article_content.append(text)
            else:
                body_elements = soup.select('div.article_content, div#articleBodyContents, div#article_body, div.news_content, article.article_view, div.view_content')
                for elem in body_elements:
                    text = elem.get_text(separator='\n', strip=True)
                    if text:
                        article_content.append(text)
        full_text = '\n'.join(article_content)
        return full_text if len(full_text) > 300 else ""
    except Exception:
        return ""

def clean_news_title(title):
    patterns = [
        r'\[.*?\]', r'\(.*?\)', r'\{.*?\}', r'<[^>]+>', r'\[\s*\w+\s*\]',
        r'\|', r'\:', r'\_', r'\-', r'\+', r'=', r'/', r'\\'
    ]
    media_keywords = ["중앙일보", "경향신문", "문화일보", "조선일보", "동아일보", "한겨레"]
    cleaned_title = title
    for p in media_keywords:
        cleaned_title = re.sub(re.escape(p), '', cleaned_title, flags=re.IGNORECASE)
    for p in patterns:
        cleaned_title = re.sub(p, '', cleaned_title).strip()
    return re.sub(r'\s+', ' ', cleaned_title).strip()

async def search_news_google_cs(query):
    logging.info(f"Google CSE로 뉴스 검색: {query}")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": os.getenv("GOOGLE_API_KEY"),
        "cx": os.getenv("GOOGLE_CSE_ID"),
        "q": query,
        "num": 10,
        "hl": "ko",
        "gl": "kr"
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return data.get("items", []) if "error" not in data else []

def calculate_fact_check_confidence(criteria_scores):
    if not criteria_scores:
        return 0
    total_possible = 5 * len(criteria_scores)
    total_actual = sum(min(5, max(0, s)) for s in criteria_scores.values())
    return round((total_actual / total_possible) * 100) if total_possible else 0

def calculate_source_diversity_score(evidence):
    if not evidence:
        return 0
    sources = set()
    for item in evidence:
        if item.get("source_title"):
            sources.add(item["source_title"].lower())
        elif item.get("url"):
            domain = urlparse(item["url"]).netloc.lower()
            sources.add(domain)
    return min(len(sources), 5)
