import os
import re
import asyncio
import json
import logging
import shutil
import hashlib
import numpy as np
import boto3
from botocore.exceptions import ClientError
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from core.lambdas import (
    extract_video_id,
    fetch_youtube_transcript,
    search_news_google_cs,
    get_article_text,
    clean_news_title,
    calculate_fact_check_confidence,
    calculate_source_diversity_score
)
from core.llm_chains import (
    build_claim_extractor,
    build_claim_summarizer,
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier,
    get_chat_llm
)
from core.faiss_manager import CHUNK_CACHE_DIR

# --- 설정값 ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
MAX_CLAIMS_TO_FACT_CHECK = 10
MAX_ARTICLES_PER_CLAIM = 15
DISTANCE_THRESHOLD = 0.8
CANDIDATE_URL_POOL_SIZE = 30
# --- 설정값 끝 ---

try:
    s3 = boto3.client('s3')
except Exception as e:
    s3 = None
    logging.error(f"S3 클라이언트 초기화 실패: {e}")

embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500
)

url_locks = {}

def url_to_cache_key(url):
    return hashlib.md5(url.encode()).hexdigest()

def get_article_faiss_path(url):
    return os.path.join(CHUNK_CACHE_DIR, url_to_cache_key(url))

def upload_to_s3(local_dir, s3_key):
    if not s3:
        return
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(s3_key, file)
            try:
                s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
            except ClientError as e:
                logging.error(f"S3 업로드 실패: {s3_path} - {e}")
                raise

def download_from_s3(local_dir, s3_key):
    if not s3:
        return False
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key)
        if 'Contents' not in response:
            return False
        os.makedirs(local_dir, exist_ok=True)
        for obj in response['Contents']:
            s3_path = obj['Key']
            local_path = os.path.join(local_dir, os.path.basename(s3_path))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(S3_BUCKET_NAME, s3_path, local_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 다운로드 오류: {s3_key} - {e}")
        return False

async def ensure_article_faiss(url):
    lock = url_locks.setdefault(url, asyncio.Lock())
    async with lock:
        cache_key = url_to_cache_key(url)
        local_path = get_article_faiss_path(url)
        s3_key = f"article_faiss_cache/{cache_key}"

        if os.path.exists(local_path):
            try:
                db = FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
                return list(db.docstore._dict.values())
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
        
        if s3 and download_from_s3(local_path, s3_key):
            try:
                db = FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
                return list(db.docstore._dict.values())
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
        
        text = await get_article_text(url)
        if not text or len(text) < 200:
            return None
            
        doc = Document(page_content=text, metadata={"url": url})
        db = FAISS.from_documents([doc], embed_model)
        db.save_local(local_path)
        
        if s3:
            upload_to_s3(local_path, s3_key)
                
        return [doc]

async def find_relevant_article_urls(claim, faiss_partition_dirs):
    summarizer = build_claim_summarizer()
    try:
        summary_result = await summarizer.ainvoke({"claim": claim})
        summarized_query = summary_result.content.strip()
    except Exception:
        summarized_query = claim

    search_results = await search_news_google_cs(summarized_query)
    cse_titles = [clean_news_title(item.get('title', '')) for item in search_results[:10]]
    if not cse_titles:
        return []

    cse_title_embs = embed_model.embed_documents(cse_titles)
    search_vectors = np.array(cse_title_embs, dtype=np.float32)

    matched_urls = set()
    for faiss_dir in faiss_partition_dirs:
        if len(matched_urls) >= CANDIDATE_URL_POOL_SIZE:
            break
        try:
            db = FAISS.load_local(faiss_dir, embed_model, allow_dangerous_deserialization=True)
            if db.index.ntotal > 0:
                _, I = db.index.search(search_vectors, k=5)
                for i in I.flatten():
                    doc = db.docstore._dict.get(db.index_to_docstore_id[i])
                    if doc and doc.metadata.get("url"):
                        matched_urls.add(doc.metadata["url"])
        except Exception as e:
            logging.error(f"FAISS 파티션 검색 실패: {faiss_dir} - {e}")
            
    return list(matched_urls)

async def run_fact_check(youtube_url, faiss_partition_dirs):
    try:
        video_id = extract_video_id(youtube_url)
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}
        
        extractor = build_claim_extractor()
        claims_result = await extractor.ainvoke({"transcript": transcript})
        claims = [line.strip() for line in claims_result.content.split('\n') if line.strip()]

        if not claims:
            return {"video_id": video_id, "video_url": youtube_url, "video_total_confidence_score": 0, "claims": []}

        reducer = build_reduce_similar_claims_chain()
        reduced_result = await reducer.ainvoke({"claims_json": json.dumps(claims, ensure_ascii=False)})
        
        claims_to_check = json.loads(reduced_result.content)
        claims_to_check = claims_to_check[:MAX_CLAIMS_TO_FACT_CHECK]

    except Exception as e:
        return {"error": f"주장 추출/정제 중 오류: {e}"}

    async def process_claim_step(claim):
        candidate_urls = await find_relevant_article_urls(claim, faiss_partition_dirs)
        if not candidate_urls:
            return {"claim": claim, "result": "insufficient_evidence", "confidence_score": 0, "evidence": []}

        successful_docs = []
        urls_to_try = list(candidate_urls)

        while len(successful_docs) < MAX_ARTICLES_PER_CLAIM and urls_to_try:
            needed = MAX_ARTICLES_PER_CLAIM - len(successful_docs)
            batch = urls_to_try[:needed]
            urls_to_try = urls_to_try[needed:]
            
            results = await asyncio.gather(*[ensure_article_faiss(url) for url in batch])
            successful_docs.extend(doc for result in results if result for doc in result)
        
        if not successful_docs:
            return {"claim": claim, "result": "insufficient_evidence", "confidence_score": 0, "evidence": []}

        fact_checker = build_factcheck_chain()
        is_claim_false = False
        validated_evidence = []

        async def factcheck_doc(doc):
            nonlocal is_claim_false
            try:
                result = await fact_checker.ainvoke({"claim": claim, "context": doc.page_content})
                content = result.content
                
                parsed = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in content.split('\n') if ":" in line)}
                
                if parsed.get("관련성") == "예":
                    fact_check = parsed.get("주장 사실 여부")
                    if fact_check == "거짓":
                        is_claim_false = True
                    
                    if fact_check != "알수없음":
                        return {
                            "url": doc.metadata.get("url"),
                            "fact_check_result": fact_check,
                            "justification": parsed.get("간단한 설명", ""),
                            "snippet": parsed.get("핵심 근거 문장", "")
                        }
            except Exception as e:
                logging.error(f"LLM 팩트체크 중 오류: {e}")
            return None

        tasks = [factcheck_doc(doc) for doc in successful_docs]
        validated_evidence = [res for res in await asyncio.gather(*tasks) if res]

        score = calculate_fact_check_confidence({
            "source_diversity": calculate_source_diversity_score(validated_evidence),
            "evidence_count": len(validated_evidence)
        })

        result = "insufficient_evidence"
        if is_claim_false:
            result = "likely_false"
        elif validated_evidence:
            result = "likely_true"

        return {"claim": claim, "result": result, "confidence_score": score, "evidence": validated_evidence[:3]}

    outputs = await asyncio.gather(*[process_claim_step(claim) for claim in claims_to_check])
    
    total_score = round(sum(o['confidence_score'] for o in outputs) / len(outputs)) if outputs else 0
    
    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    channel_type, reason = (classification.content.split('\n')[0].split(':', 1)[1].strip(), classification.content.split('\n')[1].split(':', 1)[1].strip())

    return {
        "video_id": video_id,
        "video_url": youtube_url,
        "video_total_confidence_score": total_score,
        "claims": outputs,
        "channel_type": channel_type,
        "channel_type_reason": reason
    }