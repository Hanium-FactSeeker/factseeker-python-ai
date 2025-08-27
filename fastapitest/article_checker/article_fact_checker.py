import re
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from article_checker.chains import build_article_claim_extractor
from core.lambdas import (
    get_article_text,
    calculate_fact_check_confidence,
    calculate_source_diversity_score,
)
from core.llm_chains import (
    build_claim_summarizer,
    build_reduce_similar_claims_chain,
    build_factcheck_chain,
    build_keyword_extractor_chain,
    build_three_line_summarizer_chain,
)

# Reuse evidence retrieval pipeline from the existing YouTube flow
from services.fact_checker import (
    search_and_retrieve_docs_once,
    MAX_CONCURRENT_CLAIMS,
    MAX_CONCURRENT_FACTCHECKS,
    MAX_EVIDENCES_PER_CLAIM,
)


async def _extract_claims_from_article(article_text: str) -> List[str]:
    extractor = build_article_claim_extractor()
    result = await extractor.ainvoke({"article_text": article_text})
    lines = [line.strip() for line in (result.content or "").split("\n") if line.strip()]
    return lines


async def _reduce_claims(claims: List[str]) -> List[str]:
    reducer = build_reduce_similar_claims_chain()
    claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
    reduced_result = await reducer.ainvoke({"claims_json": claims_json})

    claims_to_check: List[str] = []
    try:
        json_match = re.search(r"```json\s*(\[.*?\])\s*```", reduced_result.content or "", re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
            claims_to_check = json.loads(json_content)
        else:
            claims_to_check = [line.strip() for line in (reduced_result.content or "").strip().split('\n') if line.strip()]
    except json.JSONDecodeError:
        claims_to_check = [
            line.strip() for line in (reduced_result.content or "").strip().split('\n')
            if line.strip() and not line.strip().startswith(('```json', '```', '[', ']'))
        ]
    return claims_to_check


async def run_article_fact_check(article_url: str, faiss_partition_dirs: List[str]) -> Dict[str, Any]:
    """Article-first fact-check pipeline mirroring the YouTube flow.

    Returns a result dict similar to services.fact_checker.run_fact_check but
    with article-specific top-level fields and without YouTube identifiers.
    """
    logging.info(f"기사 분석 시작: {article_url}")

    # 1) Fetch article body
    article_text = await get_article_text(article_url)
    if not article_text:
        return {"error": "Failed to load article"}

    # 5) Extract keywords and summary
    keyword_extractor = build_keyword_extractor_chain()
    summarizer = build_three_line_summarizer_chain()

    keywords_task = keyword_extractor.ainvoke({"text": article_text})
    summary_task = summarizer.ainvoke({"text": article_text})

    keywords_result, summary_result = await asyncio.gather(keywords_task, summary_task)

    extracted_keywords = keywords_result.content.strip() if keywords_result.content else ""
    three_line_summary = summary_result.content.strip() if summary_result.content else ""

    # 2) Extract claims from article body (article-specific chain)
    try:
        claims_raw = await _extract_claims_from_article(article_text)
        if not claims_raw:
            return {
                "article_url": article_url,
                "article_total_confidence_score": 0,
                "claims": [],
            }

        # 3) Reduce/normalize claims (reuse existing reducer)
        claims_to_check = await _reduce_claims(claims_raw)
        # Keep parity with service defaults (max 10)
        claims_to_check = claims_to_check[:10]
        if not claims_to_check:
            return {
                "article_url": article_url,
                "article_total_confidence_score": 0,
                "claims": [],
            }
        logging.info(f"✂️ 기사 기반 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")
    except Exception as e:
        logging.exception(f"주장 추출/정제 중 오류: {e}")
        return {"error": f"Failed to extract claims: {e}"}

    # 4) For each claim: CSE + FAISS titles → fetch evidence and fact-check
    async def process_claim_step(idx: int, claim: str) -> Dict[str, Any]:
        logging.info(f"--- 기사 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")
        url_set = set()
        validated_evidence: List[Dict[str, Any]] = []
        fact_checker = build_factcheck_chain()

        async def factcheck_doc(doc):
            try:
                check_result = await fact_checker.ainvoke({"claim": claim, "context": doc.page_content})
                result_content = check_result.content or ""
                relevance = re.search(r"관련성: (.+)", result_content)
                fact_check_result_match = re.search(r"사실 설명 여부: (.+)", result_content)
                justification = re.search(r"간단한 설명: (.+)", result_content)
                snippet = re.search(r"핵심 근거 문장: (.+)", result_content, re.DOTALL)
                url = doc.metadata.get("url")

                if (
                    relevance and fact_check_result_match and justification
                    and "예" in relevance.group(1)
                    and url and url not in url_set
                ):
                    url_set.add(url)
                    return {
                        "url": url,
                        "relevance": "yes",
                        "fact_check_result": fact_check_result_match.group(1).strip(),
                        "justification": justification.group(1).strip(),
                        "snippet": snippet.group(1).strip() if snippet else "",
                    }
            except Exception as e:
                logging.error(f"    - LLM 팩트체크 체인 실행 중 오류: {e}")
            return None

        factcheck_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FACTCHECKS)

        async def limited_factcheck_doc(doc):
            async with factcheck_semaphore:
                return await factcheck_doc(doc)

        new_docs = await search_and_retrieve_docs_once(claim, faiss_partition_dirs, set())
        if not new_docs:
            logging.info(f"근거 문서를 찾지 못함: '{claim}'")
            return {
                "claim": claim,
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": [],
            }

        for i in range(0, len(new_docs), MAX_CONCURRENT_FACTCHECKS):
            if len(validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                break
            batch = new_docs[i : i + MAX_CONCURRENT_FACTCHECKS]
            factcheck_results = await asyncio.gather(*[limited_factcheck_doc(doc) for doc in batch])
            for res in factcheck_results:
                if res and isinstance(res, dict):  # None이 아니고 dict 타입인지 확인
                    validated_evidence.append(res)
                    logging.info(f"✅ [네이버] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                    if len(validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                        break

        diversity_score = calculate_source_diversity_score(validated_evidence)
        confidence_score = calculate_fact_check_confidence(
            {"source_diversity": diversity_score, "evidence_count": min(len(validated_evidence), 5)}
        )

        logging.info(
            f"--- 기사 팩트체크 완료: '{claim}' -> 신뢰도: {confidence_score}% (증거 {len(validated_evidence)}개)"
        )

        # 신뢰도가 20% 이하일 경우 Google CSE로 재시도
        naver_confidence = confidence_score
        naver_evidence = validated_evidence.copy()
        
        if confidence_score <= 20:
            logging.info(f"신뢰도 {confidence_score}%로 인한 Google CSE 재시도: '{claim}'")
            
            # Google CSE로 재검색
            google_docs = await search_and_retrieve_docs_once(claim, faiss_partition_dirs, set(), use_google_cse=True)
            if google_docs:
                logging.info(f"Google CSE로 {len(google_docs)}개 문서 발견, 재팩트체크 수행")
                
                # Google CSE 결과로 재팩트체크
                google_validated_evidence = []
                for i in range(0, len(google_docs), MAX_CONCURRENT_FACTCHECKS):
                    if len(google_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                        break
                    batch = google_docs[i : i + MAX_CONCURRENT_FACTCHECKS]
                    factcheck_results = await asyncio.gather(*[limited_factcheck_doc(doc) for doc in batch])
                    for res in factcheck_results:
                        if res and isinstance(res, dict):  # None이 아니고 dict 타입인지 확인
                            google_validated_evidence.append(res)
                            logging.info(f"✅ [Google CSE] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                            if len(google_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                                break
                
                # Google CSE 결과로 신뢰도 계산
                if google_validated_evidence:
                    google_diversity_score = calculate_source_diversity_score(google_validated_evidence)
                    google_confidence = calculate_fact_check_confidence(
                        {"source_diversity": google_diversity_score, "evidence_count": min(len(google_validated_evidence), 5)}
                    )
                    logging.info(f"Google CSE 재팩트체크 완료: 신뢰도 {google_confidence}% (증거 {len(google_validated_evidence)}개)")
                    
                    # 둘 중 더 높은 신뢰도 선택
                    if google_confidence > naver_confidence:
                        confidence_score = google_confidence
                        validated_evidence = google_validated_evidence
                        logging.info(f"Google CSE 결과 선택: {google_confidence}% > 네이버 {naver_confidence}%")
                    else:
                        logging.info(f"네이버 결과 유지: {naver_confidence}% >= Google CSE {google_confidence}%")
                else:
                    logging.info("Google CSE로도 검증된 증거를 찾지 못함")
            else:
                logging.info("Google CSE로 문서를 찾지 못함")
        else:
            logging.info(f"신뢰도 {confidence_score}%로 충분하므로 Google CSE 재시도 생략")

        # 최종 신뢰도가 20% 이하이면 파티션 9로 재시도
        final_confidence = confidence_score
        final_evidence = validated_evidence
        
        if confidence_score <= 20 and len(validated_evidence) > 0:
            logging.info(f"최종 신뢰도 {confidence_score}%로 낮음 → 파티션 9로 재시도: '{claim}'")
            
            # 파티션 9만 사용하여 재검색
            partition_9_dirs = [dir for dir in faiss_partition_dirs if "9" in dir]
            logging.info(f"🔍 파티션 9 검색: 전체 파티션 {len(faiss_partition_dirs)}개 중 파티션 9 포함 {len(partition_9_dirs)}개 발견")
            if partition_9_dirs:
                logging.info(f"파티션 9 디렉토리 {len(partition_9_dirs)}개 발견, 재검색 시작")
                for dir in partition_9_dirs:
                    logging.info(f"  📁 파티션 9 경로: {dir}")
                
                # 파티션 9로 재검색
                partition_9_docs = await search_and_retrieve_docs_once(claim, partition_9_dirs, set(), use_google_cse=False)
                if partition_9_docs:
                    logging.info(f"파티션 9에서 {len(partition_9_docs)}개 문서 발견, 재팩트체크 수행")
                    
                    # 파티션 9 결과로 재팩트체크
                    partition_9_validated_evidence = []
                    for i in range(0, len(partition_9_docs), MAX_CONCURRENT_FACTCHECKS):
                        if len(partition_9_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                            break
                        batch = partition_9_docs[i : i + MAX_CONCURRENT_FACTCHECKS]
                        factcheck_results = await asyncio.gather(*[limited_factcheck_doc(doc) for doc in batch])
                        for res in factcheck_results:
                            if res and isinstance(res, dict):
                                partition_9_validated_evidence.append(res)
                                logging.info(f"✅ [파티션9] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                                if len(partition_9_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                                    break
                    
                    # 파티션 9 결과로 신뢰도 계산
                    if partition_9_validated_evidence:
                        partition_9_diversity_score = calculate_source_diversity_score(partition_9_validated_evidence)
                        partition_9_confidence = calculate_fact_check_confidence(
                            {"source_diversity": partition_9_diversity_score, "evidence_count": min(len(partition_9_validated_evidence), 5)}
                        )
                        logging.info(f"파티션 9 재팩트체크 완료: 신뢰도 {partition_9_confidence}% (증거 {len(partition_9_validated_evidence)}개)")
                        
                        # 파티션 9 결과가 더 높으면 선택
                        if partition_9_confidence > final_confidence:
                            final_confidence = partition_9_confidence
                            final_evidence = partition_9_validated_evidence
                            logging.info(f"파티션 9 결과 선택: {partition_9_confidence}% > 기존 {confidence_score}%")
                        else:
                            logging.info(f"기존 결과 유지: {confidence_score}% >= 파티션 9 {partition_9_confidence}%")
                    else:
                        logging.info("파티션 9으로도 검증된 증거를 찾지 못함")
                else:
                    logging.info("파티션 9에서 문서를 찾지 못함")
            else:
                logging.info("파티션 9 디렉토리를 찾을 수 없음")

        return {
            "claim": claim,
            "result": "likely_true" if final_evidence else "insufficient_evidence",
            "confidence_score": final_confidence,
            "evidence": final_evidence[:3],
        }

    claim_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLAIMS)

    async def limited_process_claim(idx: int, claim: str):
        async with claim_semaphore:
            return await process_claim_step(idx, claim)

    gathered = await asyncio.gather(*[limited_process_claim(i, c) for i, c in enumerate(claims_to_check)], return_exceptions=True)

    outputs: List[Dict[str, Any]] = []
    for i, res in enumerate(gathered):
        if isinstance(res, Exception):
            claim_text = claims_to_check[i] if i < len(claims_to_check) else ""
            err_type = type(res).__name__
            err_msg = str(res) or repr(res) or err_type
            logging.error(f"🛑 주장 처리 중 예외 발생: '{claim_text}' -> {err_type}: {err_msg}")
            outputs.append(
                {
                    "claim": claim_text,
                    "result": "error",
                    "confidence_score": 0,
                    "evidence": [],
                    "error": f"{err_type}: {err_msg}",
                    "error_type": err_type,
                    "error_stage": "process_claim_step",
                }
            )
        else:
            outputs.append(res)

    if outputs:
        avg_score = round(sum(o.get("confidence_score", 0) for o in outputs) / len(outputs))
        evidence_ratio = sum(1 for o in outputs if o.get("result") == "likely_true") / len(outputs)
        summary = (
            f"증거 확보된 주장 비율: {evidence_ratio*100:.1f}%"
            if len(outputs) >= 3
            else f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
        )
    else:
        avg_score = 0
        summary = "결과 없음"

    return {
        "article_url": article_url,
        "article_total_confidence_score": avg_score,
        "claims": outputs,
        "summary": summary,
        "created_at": datetime.now().isoformat(),
        "keywords": extracted_keywords,
        "three_line_summary": three_line_summary,
    }

