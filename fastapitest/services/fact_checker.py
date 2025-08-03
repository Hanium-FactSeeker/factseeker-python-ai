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
    build_claim_summarizer,
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier,
    get_chat_llm
)
# FAISS_DB_PATH, CHUNK_CACHE_DIR 등은 main.py나 개별 기사 FAISS와 관련
from core.faiss_manager import get_or_build_faiss, CHUNK_CACHE_DIR 

MAX_CLAIMS_TO_FACT_CHECK = 10
embed_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500)

async def search_and_retrieve_docs(claim):
    """
    주장(claim)에 대해 뉴스 검색을 수행하고, 관련성 있는 기사를 찾아 텍스트를 반환합니다.
    """
    search_results = await search_news_google_cs(claim)
    
    docs = []
    retrieved_urls = set()
    
    for item in search_results:
        url = item.get("link")
        source_title = item.get("title")
        snippet = item.get("snippet")
        
        if not url or url in retrieved_urls:
            continue
        
        try:
            # 기사 텍스트 추출 (캐시 포함)
            article_text = await get_or_build_faiss(
                url=url, 
                embed_model=embed_model,
            )
            
            if article_text and len(article_text) > 200:
                # 기사 텍스트를 Document 객체로 변환
                doc = Document(
                    page_content=article_text,
                    metadata={
                        "source_title": source_title,
                        "url": url,
                        "snippet": snippet
                    }
                )
                docs.append(doc)
                retrieved_urls.add(url)
        except Exception as e:
            logging.error(f"기사 처리 중 오류 발생 ({url}): {e}")
            continue

    return docs


async def run_fact_check(youtube_url):
    """
    주어진 유튜브 URL에 대한 팩트체크 프로세스를 실행합니다.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    logging.info(f"유튜브 분석 시작: {youtube_url}")

    try:
        # 0. 유튜브 스크립트 추출
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}
        
        # 1. 팩트체크 가능한 주장 추출
        extractor = build_claim_extractor()
        result = await extractor.ainvoke({"transcript": transcript})
        # LangChain AIMessage 객체에서 content 속성을 사용하도록 수정
        claims = [line.strip() for line in result.content.strip().split('\n') if line.strip()]
        
        # 주장이 없는 경우
        if not claims:
            return {
                "video_id": video_id,
                "video_total_confidence_score": 0,
                "claims": []
            }
        
        logging.info(f"🔎 팩트체크를 위한 주장 {len(claims)}개 추출 완료: {claims}")
        
        # 2. 유사하거나 중복되는 주장 제거
        reducer = build_reduce_similar_claims_chain()
        reduced_result = await reducer.ainvoke({"claims": claims})
        # LangChain AIMessage 객체에서 content 속성을 사용하도록 수정
        claims_to_check = [
            line.strip() for line in reduced_result.content.strip().split('\n')
            if line.strip()
        ][:MAX_CLAIMS_TO_FACT_CHECK]
        
        logging.info(f"✂️ 중복 제거 후 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")

        if not claims_to_check:
            return {
                "video_id": video_id,
                "video_total_confidence_score": 0,
                "claims": []
            }
        
    except Exception as e:
        logging.exception(f"주장 추출/정제 중 오류: {e}")
        return {"error": f"Failed to extract claims: {e}"}

    # 3. 각 주장에 대해 팩트체크 수행 (병렬 처리)
    async def process_claim_step(claim_tuple):
        idx, claim = claim_tuple
        logging.info(f"--- 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")
        
        # 뉴스 검색 및 문서 검색
        docs = await search_and_retrieve_docs(claim)
        if not docs:
            logging.info(f"근거를 찾지 못함: '{claim}'")
            return {
                "claim": claim,
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": []
            }

        faiss_db_temp_path = os.path.join(CHUNK_CACHE_DIR, f"temp_faiss_{video_id}_{idx}")
        if os.path.exists(faiss_db_temp_path):
            shutil.rmtree(faiss_db_temp_path)

        # 문서들을 FAISS 벡터스토어에 저장
        faiss_db = FAISS.from_documents(docs, embed_model)
        faiss_db.save_local(faiss_db_temp_path)
        logging.info(f"✅ 기사 문서 {len(docs)}개로 임시 FAISS DB 생성 완료")
        
        # 관련성 높은 문서 검색
        retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
        validated_evidence = []
        
        # 팩트체크 체인 실행
        fact_checker = build_factcheck_chain()
        for i, doc in enumerate(docs):
            try:
                check_result = await fact_checker.ainvoke({
                    "claim": claim,
                    "context": doc.page_content
                })
                # LangChain AIMessage 객체에서 content 속성을 사용하도록 수정
                result_content = check_result.content

                # LLM 결과 파싱
                relevance = re.search(r"관련성: (.+)", result_content)
                fact_check_result = re.search(r"사실 여부: (.+)", result_content)
                justification = re.search(r"판단 근거: (.+)", result_content)
                
                if relevance and fact_check_result and justification:
                    if "예" in relevance.group(1):
                        validated_evidence.append({
                            "source_title": doc.metadata.get("source_title"),
                            "url": doc.metadata.get("url"),
                            "snippet": doc.metadata.get("snippet"),
                            "relevance": "yes",
                            "fact_check_result": fact_check_result.group(1).strip(),
                            "justification": justification.group(1).strip()
                        })
                        logging.info(f"    - 근거 확보 ({i+1}): {doc.metadata.get('source_title')}")
                else:
                    logging.warning(f"    - LLM 응답 형식 오류 ({i+1}): {result_content}")
            except Exception as e:
                logging.error(f"    - LLM 팩트체크 체인 실행 중 오류: {e}")

        # 임시 FAISS DB 삭제
        if os.path.exists(faiss_db_temp_path):
            shutil.rmtree(faiss_db_temp_path)

        # 신뢰도 점수 계산
        diversity_score = calculate_source_diversity_score(validated_evidence)
        confidence_score = calculate_fact_check_confidence({
            "source_diversity": diversity_score,
            "evidence_count": len(validated_evidence) if len(validated_evidence) <= 5 else 5
        })

        logging.info(f"--- 팩트체크 완료: '{claim}' -> 신뢰도: {confidence_score}%")

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
    # confidence_summary 문자열 생성 로직 변경
    if len(outputs) < 3:
        summary = f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
    else:
        # 백분율 포맷팅
        evidence_ratio_percent = evidence_ratio * 100
        summary = f"증거 확보된 주장 비율: {evidence_ratio_percent:.1f}%"

    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    channel_type, reason = parse_channel_type(classification.content)

    return {
        "video_id": video_id,
        "video_total_confidence_score": avg_score,
        "claims": outputs,
        "summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason
    }

def parse_channel_type(llm_output: str):
    """LLM 출력에서 채널 유형과 이유를 파싱합니다."""
    # 정규표현식을 사용하여 채널 유형과 이유를 추출
    channel_type_match = re.search(r"채널 유형:\s*(.+)", llm_output)
    reason_match = re.search(r"판단 근거:\s*(.+)", llm_output)

    channel_type = channel_type_match.group(1).strip() if channel_type_match else "알 수 없음"
    reason = reason_match.group(1).strip() if reason_match else "LLM 응답에서 판단 근거를 찾을 수 없습니다."

    return channel_type, reason
