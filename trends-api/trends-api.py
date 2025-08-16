import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fastapi import FastAPI

# --- 기존 크롤링 코드를 함수로 재구성 ---
def get_google_trends():
    # 이 함수는 호출될 때마다 크롤링을 수행합니다.
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    url = 'https://trends.google.co.kr/trending?geo=KR&category=14&hours=168'
    korean_keywords = []

    try:
        driver.get(url)
        time.sleep(5)
        
        rows = driver.find_elements(By.CSS_SELECTOR, "tbody tr.enOdEe-wZVHld-xMbwt")
        
        for row in rows:
            if len(korean_keywords) >= 5:
                break
            
            keyword_element = row.find_element(By.CSS_SELECTOR, "td:nth-child(2) div")
            keyword = keyword_element.get_attribute('textContent').strip()
            
            if re.match("^[가-힣0-9\s]+$", keyword):
                korean_keywords.append(keyword)
    
    finally:
        driver.quit()
        
    return korean_keywords

app = FastAPI()

# '/api/trends' 주소로 GET 요청이 오면 get_google_trends 함수를 실행
@app.get("/api/trends")
async def trends_api(): # FastAPI는 비동기(async) 함수를 사용합니다.
    trends_data = get_google_trends()
    # dict를 반환하면 FastAPI가 알아서 JSON으로 변환해줍니다.
    return {"trends": trends_data}