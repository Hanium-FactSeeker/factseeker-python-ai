FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential libxml2-dev libxslt1-dev chromium-driver \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# fastapitest 폴더 안의 앱 코드만 복사
COPY fastapitest/ /app/

RUN pip install --upgrade pip \
  && pip install --no-cache-dir \
     fastapi uvicorn[standard] python-dotenv \
     faiss-cpu langchain-openai langchain-community \
     boto3 aiohttp requests beautifulsoup4 newspaper3k \
     youtube-transcript-api selenium numpy scikit-learn

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
