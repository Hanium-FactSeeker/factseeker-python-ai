import os
import re
import asyncio
import json
import logging
import shutil
from urllib.parse import urlparse, urlunparse
import boto3 # boto3 임포트 추가

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# core.lambdas 및 core.llm_chains, core.faiss_manager는 기존과 동일하다고 가정
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
    build_claim_summarizer, # 사용자 요청에 따라 역할 변경됨
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier,
    get_chat_llm
)
# FAISS_DB_PATH, CHUNK_CACHE_DIR 등은 main.py나 개별 기사 FAISS와 관련
from core.faiss_manager import get_or_build_faiss, CHUNK_CACHE_DIR 

MAX_CLAIMS_TO_FACT_CHECK = 10
embed_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500)


async def run_fact_check(youtube_url: str):
    """
    주어진 유튜브 URL에 대한 팩트체크 프로세스를 실행합니다.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        logging.error(f"유효하지 않은 유튜브 URL: {youtube_url}")
        return None

    # 1. 유튜브 스크립트 추출
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        logging.error(f"스크립트를 찾을 수 없습니다: {video_id}")
        return None

    # 2. 주장 추출
    claim_extractor_chain = build_claim_extractor()
    raw_claims = transcript.split('.') 
    
    extracted_claims = []
    for claim_text in raw_claims:
        if len(claim_text.strip()) > 10:
            result = await claim_extractor_chain.ainvoke({"text": claim_text.strip()})
            if result and result.content != "None":
                extracted_claims.append(result.content)
    
    # 3. 주장 중복 제거
    if not extracted_claims:
        logging.info("추출된 주장이 없어 팩트체크를 진행할 수 없습니다.")
        return {
            "video_id": video_id,
            "video_total_confidence_score": 0,
            "claims": []
        }
    
    reduce_claims_chain = build_reduce_similar_claims_chain()
    reduced_claims_content = await reduce_claims_chain.ainvoke({"claims": extracted_claims})
    
    try:
        claims_to_check = json.loads(reduced_claims_content.content)
        claims_to_check = claims_to_check[:MAX_CLAIMS_TO_FACT_CHECK]
    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 오류: {e}")
        claims_to_check = extracted_claims[:MAX_CLAIMS_TO_FACT_CHECK]
    
    logging.info(f"✂️ 중복 제거 후 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")

    # 4. 각 주장에 대한 팩트체크 실행
    fact_check_chain = build_factcheck_chain()
    search_query_chain = build_claim_summarizer()

    async def process_claim_step(claim_with_idx):
        idx, claim = claim_with_idx
        logging.info(f"--- 팩트체크 시작: ({idx+1}/{len(claims_to_check)}) '{claim}'")
        
        # 주장을 검색 키워드로 변환 (사용자 정의 체인 사용)
        search_query_result = await search_query_chain.ainvoke({"claim": claim})
        search_query = search_query_result.content
        logging.info(f"--- 뉴스 검색 키워드 생성: '{claim}' -> '{search_query}'")
        
        # 뉴스 검색
        news_urls = await search_news_google_cs(search_query)
        if not news_urls:
            logging.info(f"근거를 찾지 못함: '{claim}'")
            return {
                "claim": claim,
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": []
            }
        
        validated_evidence = []
        unique_urls = set()
        for news_item in news_urls:
            url = news_item.get("link")
            if not url or url in unique_urls:
                continue
            unique_urls.add(url)

            try:
                article_text = await get_article_text(url)
                if not article_text or len(article_text) < 100:
                    continue

                faiss_db, _ = await get_or_build_faiss(url=url, article_text=article_text, embed_model=embed_model)

                docs = faiss_db.similarity_search(query=claim, k=3)
                context = " ".join([doc.page_content for doc in docs])

                if not context:
                    continue

                fact_check_result = await fact_check_chain.ainvoke({
                    "claim": claim,
                    "context": context
                })

                # LLM의 JSON 응답을 올바르게 파싱
                parsed_result = json.loads(fact_check_result.content.strip())
                is_relevant = parsed_result.get("관련성") == "예"
                is_fact_match = parsed_result.get("사실 설명") == "예"

                if is_relevant and is_fact_match:
                    validated_evidence.append({
                        "url": url,
                        "source_title": clean_news_title(url),
                        # 요약 로직은 삭제하고, 추후 필요시 추가
                    })
                    if len(validated_evidence) >= 3:
                        break

            except Exception as e:
                logging.error(f"기사 처리 중 오류 발생 ({url}): {e}")
                continue
        
        confidence_score = 0
        if validated_evidence:
            confidence_score = calculate_fact_check_confidence(validated_evidence)
            
        return {
            "claim": claim,
            "result": "likely_true" if validated_evidence else "insufficient_evidence",
            "confidence_score": confidence_score,
            "evidence": validated_evidence[:3]
        }

    claim_tasks = [
        process_claim_step((idx, claim))
        for idx, claim in enumerate(claims_to_check)
    ]
    outputs = await asyncio.gather(*claim_tasks, return_exceptions=True)
    
    outputs = [output for output in outputs if not isinstance(output, Exception)]

    avg_score = round(sum(o['confidence_score'] for o in outputs) / len(outputs)) if outputs else 0
    evidence_ratio = sum(1 for o in outputs if o["result"] == "likely_true") / len(outputs) if outputs else 0
    
    if len(outputs) < 3:
        summary = f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
    else:
        evidence_ratio_percent = evidence_ratio * 100
        summary = f"증거 확보된 주장 비율: {evidence_ratio_percent:.1f}%"

    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    
    # LLM의 JSON 응답을 올바르게 파싱
    try:
        channel_classification_json = json.loads(classification.content)
        channel_type = channel_classification_json.get("type", "기타")
        reason = channel_classification_json.get("reason", "")
    except json.JSONDecodeError:
        channel_type = "기타"
        reason = "분류 결과 파싱 오류"

    return {
        "video_id": video_id,
        "video_total_confidence_score": avg_score,
        "confidence_summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason,
        "claims": outputs
    }
