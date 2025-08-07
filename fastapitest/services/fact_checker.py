import os
import re
import asyncio
import json
import logging
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from core.lambdas import (
    extract_video_id,
    fetch_youtube_transcript,
    search_news_google_cs,
    clean_news_title,
    calculate_fact_check_confidence,
    calculate_source_diversity_score
)
from core.llm_chains import (
    build_claim_extractor,
    build_claim_summarizer,
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier
)
# 캐시 관리 함수를 직접 임포트합니다.
from core.faiss_manager import get_documents_with_caching, CHUNK_CACHE_DIR

# --- 상수 정의 ---
MAX_CLAIMS_TO_FACT_CHECK = 10

# --- 모델 초기화 ---
embed_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500)


async def search_and_retrieve_docs(claim: str) -> list[Document]:
    """
    주장을 기반으로 뉴스 기사를 검색하고, 캐싱 로직을 통해 문서를 가져옵니다.
    """
    summarizer = build_claim_summarizer()
    try:
        summary_result = await summarizer.ainvoke({"claim": claim})
        summarized_query = summary_result.content.strip()
        logging.info(f"🔍 요약된 검색어: '{summarized_query}'")
    except Exception as e:
        logging.error(f"Claim 요약 실패: {e}, 원문으로 검색 진행")
        summarized_query = claim

    # 1. Google Custom Search를 통해 관련 뉴스 URL 검색 (상위 5개 결과 사용)
    search_results = await search_news_google_cs(summarized_query)
    cse_urls = [item.get('link') for item in search_results[:5]]

    if not cse_urls:
        logging.warning(f"'{summarized_query}'에 대한 뉴스 검색 결과가 없습니다.")
        return []

    # 2. 각 URL에 대해 캐싱 로직을 사용하여 문서 가져오기
    logging.info(f"총 {len(cse_urls)}개의 URL에 대해 캐시 확인 및 문서화 시작.")
    tasks = [get_documents_with_caching(url, embed_model) for url in cse_urls]
    results = await asyncio.gather(*tasks)

    # 3. 모든 문서들을 하나의 리스트로 통합
    all_docs = [doc for doc_list in results for doc in doc_list]

    logging.info(f"📰 최종 문서(청크) 수: {len(all_docs)}")
    logging.info(f"[DEBUG] search_and_retrieve_docs: docs 길이={len(all_docs)}, claim='{claim}'")
    return all_docs


async def run_fact_check(youtube_url: str):
    """
    유튜브 영상의 주장을 팩트체크하는 전체 프로세스를 실행합니다.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    logging.info(f"유튜브 분석 시작: {youtube_url}")
    try:
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}

        # --- 1. 주장 추출 ---
        extractor = build_claim_extractor()
        result = await extractor.ainvoke({"transcript": transcript})
        claims = [line.strip() for line in result.content.strip().split('\n') if line.strip()]

        if not claims:
            return {
                "video_id": video_id,
                "video_url": youtube_url,
                "video_total_confidence_score": 0,
                "claims": []
            }

        # --- 2. 유사 주장 정제 및 축소 ---
        reducer = build_reduce_similar_claims_chain()
        claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
        reduced_result = await reducer.ainvoke({"claims_json": claims_json})

        claims_to_check = []
        try:
            json_match = re.search(r"```json\s*(\[.*?\])\s*```", reduced_result.content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                claims_to_check = json.loads(json_content)
            else: # JSON 형식이 아닐 경우, 줄 단위로 파싱
                claims_to_check = [
                    line.strip() for line in reduced_result.content.strip().split('\n')
                    if line.strip() and not line.strip().startswith(('```json', '```', '[', ']'))
                ]
        except json.JSONDecodeError:
            logging.error("JSON 파싱 오류. 줄 단위로 파싱합니다.")
            claims_to_check = [
                line.strip() for line in reduced_result.content.strip().split('\n')
                if line.strip() and not line.strip().startswith(('```json', '```', '[', ']'))
            ]

        claims_to_check = claims_to_check[:MAX_CLAIMS_TO_FACT_CHECK]
        logging.info(f"✂️ 중복 제거 후 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")

        if not claims_to_check:
            return {
                "video_id": video_id,
                "video_url": youtube_url,
                "video_total_confidence_score": 0,
                "claims": []
            }

    except Exception as e:
        logging.exception(f"주장 추출/정제 중 오류: {e}")
        return {"error": f"Failed to extract claims: {e}"}


    # --- 3. 각 주장에 대한 팩트체크 병렬 처리 ---
    async def process_claim_step(idx, claim):
        logging.info(f"[DEBUG] process_claim_step 진입: {idx} - '{claim}'")
        logging.info(f"--- 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")

        docs = await search_and_retrieve_docs(claim)

        if not docs:
            logging.warning(f"근거 문서를 찾지 못함: '{claim}'")
            return {
                "claim": claim,
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": []
            }

        # --- 4. 팩트체크를 위한 임시 벡터 DB 생성 ---
        faiss_db_temp_path = os.path.join(CHUNK_CACHE_DIR, f"temp_faiss_{video_id}_{idx}")
        if os.path.exists(faiss_db_temp_path):
            shutil.rmtree(faiss_db_temp_path)

        faiss_db = FAISS.from_documents(docs, embed_model)
        faiss_db.save_local(faiss_db_temp_path)
        logging.info(f"✅ 기사 문서 {len(docs)}개로 임시 FAISS DB 생성 완료")

        # --- 5. LLM을 이용한 근거 검증 병렬 처리 ---
        fact_checker = build_factcheck_chain()
        async def factcheck_doc(doc):
            try:
                check_result = await fact_checker.ainvoke({
                    "claim": claim,
                    "context": doc.page_content
                })
                result_content = check_result.content
                relevance = re.search(r"관련성: (.+)", result_content)
                fact_check_result = re.search(r"사실 설명 여부: (.+)", result_content)
                justification = re.search(r"간단한 설명: (.+)", result_content)
                snippet = re.search(r"핵심 근거 문장: (.+)", result_content)

                if relevance and "예" in relevance.group(1) and fact_check_result and justification:
                    
                    # ✨✨✨ 수정된 부분 ✨✨✨
                    # '사실 설명 여부'가 '아니오'인 경우는 결과에서 제외합니다.
                    fact_check_value = fact_check_result.group(1).strip()
                    if "아니오" in fact_check_value:
                        logging.info(f"Filtered out evidence with 'fact_check_result: 아니오' for url: {doc.metadata.get('url')}")
                        return None # '아니오'인 경우 None을 반환하여 필터링

                    return {
                        "url": doc.metadata.get("url"),
                        "relevance": "yes",
                        "fact_check_result": fact_check_value,
                        "justification": justification.group(1).strip(),
                        "snippet": snippet.group(1).strip() if snippet else ""
                    }
            except Exception as e:
                logging.error(f"   - LLM 팩트체크 체인 실행 중 오류: {e}")
            return None

        factcheck_tasks = [factcheck_doc(doc) for doc in docs]
        factcheck_results = await asyncio.gather(*factcheck_tasks)
        validated_evidence = [res for res in factcheck_results if res]

        if os.path.exists(faiss_db_temp_path):
            shutil.rmtree(faiss_db_temp_path)

        # --- 6. 신뢰도 점수 계산 ---
        diversity_score = calculate_source_diversity_score(validated_evidence)
        confidence_score = calculate_fact_check_confidence({
            "source_diversity": diversity_score,
            "evidence_count": min(len(validated_evidence), 5)
        })

        logging.info(f"--- 팩트체크 완료: '{claim}' -> 신뢰도: {confidence_score}%")

        return {
            "claim": claim,
            "result": "likely_true" if validated_evidence else "insufficient_evidence",
            "confidence_score": confidence_score,
            "evidence": validated_evidence[:3]  # 상위 3개 근거만 반환
        }

    claim_tasks = [process_claim_step(idx, claim) for idx, claim in enumerate(claims_to_check)]
    outputs = await asyncio.gather(*claim_tasks, return_exceptions=True)
    outputs = [output for output in outputs if not isinstance(output, Exception)]

    # --- 7. 최종 결과 종합 ---
    avg_score = round(sum(o['confidence_score'] for o in outputs) / len(outputs)) if outputs else 0
    evidence_ratio = sum(1 for o in outputs if o["result"] == "likely_true") / len(outputs) if outputs else 0
    summary = f"증거 확보된 주장 비율: {evidence_ratio*100:.1f}%" if len(outputs) >= 3 else f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"

    # 채널 유형 분류
    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    channel_type, reason = parse_channel_type(classification.content)

    logging.info(f"[DEBUG] 리턴 claims 길이: {len(outputs)}, claims: {outputs}")

    return {
        "video_id": video_id,
        "video_url": youtube_url,
        "video_total_confidence_score": avg_score,
        "claims": outputs,
        "summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason
    }

def parse_channel_type(llm_output: str) -> tuple[str, str]:
    """LLM의 채널 유형 분류 결과를 파싱합니다."""
    channel_type_match = re.search(r"채널 유형:\s*(.+)", llm_output)
    reason_match = re.search(r"분류 근거:\s*(.+)", llm_output)
    channel_type = channel_type_match.group(1).strip() if channel_type_match else "알 수 없음"
    reason = reason_match.group(1).strip() if reason_match else "LLM 응답에서 판단 근거를 찾을 수 없습니다."
    return channel_type, reason