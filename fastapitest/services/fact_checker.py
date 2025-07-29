import os
import re
import asyncio
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse, urlunparse
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
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
from core.faiss_manager import (
    get_or_build_faiss,
    FAISS_DB_PATH,
    CHUNK_CACHE_DIR,
)

MAX_CLAIMS_TO_FACT_CHECK = 10
embed_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, max_retries=5)

def parse_channel_type(llm_output: str):
    type_match = re.search(r"유형\s*:\s*([^\n]+)", llm_output)
    reason_match = re.search(r"분류 근거\s*:\s*([^\n]+)", llm_output)
    channel_type = type_match.group(1).strip() if type_match else None
    channel_type_reason = reason_match.group(1).strip() if reason_match else None
    return channel_type, channel_type_reason

async def run_fact_check(youtube_url: str, dedup_method: str = "llm"):
    logging.info(f"유튜브 분석 시작: {youtube_url}")
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    transcript = await fetch_youtube_transcript(video_id)
    if not transcript:
        return {"error": "Failed to load transcript"}

    transcript_sentences = [t for t in re.split(r"[.?!\n]", transcript) if t.strip()]
    transcript_total_sentences = len(transcript_sentences)

    claim_extractor = build_claim_extractor()
    result = await claim_extractor.ainvoke({"transcript": transcript})
    parsed_claims = []
    claim_pattern = re.compile(r'^[-\*]\s*(?:\[\d+\]\s*)?(.+?)\s*→\s*(.+)')

    for line in result.content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = claim_pattern.match(line)
        if match:
            claim_text = match.group(1).strip()
            status_and_reason = match.group(2).strip()
            if "팩트체크 불가능" not in status_and_reason:
                if claim_text.startswith("정제: "):
                    claim_text = claim_text[len("정제: "):].strip()
                parsed_claims.append(claim_text)

    if not parsed_claims:
        return {"error": "No fact-checkable claims"}

    dedup_claims = parsed_claims
    if dedup_method == "llm":
        deduper_llm_chain = build_reduce_similar_claims_chain()
        dedup_resp = await deduper_llm_chain.ainvoke({"claims_json": json.dumps(parsed_claims, ensure_ascii=False)})
        raw_llm_output = dedup_resp.content.strip()
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_llm_output, re.DOTALL)
        if json_match:
            json_string = json_match.group(1).strip()
            dedup_claims = json.loads(json_string)
        else:
            raise ValueError("LLM deduplication output is not in expected JSON markdown format.")

    claims_to_check = dedup_claims[:MAX_CLAIMS_TO_FACT_CHECK]

    try:
        title_faiss = FAISS.load_local(FAISS_DB_PATH, embed_model, allow_dangerous_deserialization=True)
    except Exception as e:
        return {"error": f"FAISS DB load failed: {e}"}

    outputs = []
    factcheck_chain = build_factcheck_chain()

    for idx, claim in enumerate(claims_to_check):
        summarizer = build_claim_summarizer()
        summary_result = await summarizer.ainvoke({"claim": claim})
        summary = summary_result.content.strip()
        logging.info(f"\n--- [{idx+1}] 주장: {claim} | 검색어: {summary}")

        results = await search_news_google_cs(summary)
        matched_urls = []
        for item in results:
            title = clean_news_title(item.get("title", ""))
            url = urlunparse(urlparse(item.get("link", ""))._replace(query='', fragment=''))
            if not title or not url:
                continue
            docs_with_scores = title_faiss.similarity_search_with_score(title, k=3)
            for doc, score in docs_with_scores:
                if score <= 0.8 and doc.metadata.get("url"):
                    matched_urls.append((doc.metadata["url"], title))
                    break

        validated_tasks = []
        for url, title in matched_urls:
            try:
                chunk_db, article_text_from_cache = await get_or_build_faiss(url, article_text=None, embed_model=embed_model)
                if not article_text_from_cache:
                    article_text = get_article_text(url)
                    if not article_text or len(article_text) < 300:
                        continue
                    chunk_db, _ = await get_or_build_faiss(url, article_text=article_text, embed_model=embed_model)
            except Exception:
                continue

            chunks = chunk_db.similarity_search(claim, k=5)
            for chunk in chunks:
                validated_tasks.append({
                    "task": factcheck_chain.ainvoke({"claim": claim, "context": chunk.page_content}),
                    "url": url,
                    "snippet": chunk.page_content,
                    "source_title": title,
                })

        llm_responses = await asyncio.gather(*[t["task"] for t in validated_tasks], return_exceptions=True)
        validated_evidence = []
        seen_urls_for_claim = set()

        for i, response in enumerate(llm_responses):
            if isinstance(response, Exception):
                continue
            answer = response.content.strip()
            task_meta = validated_tasks[i]
            if task_meta["url"] in seen_urls_for_claim:
                continue
            if "관련성: 예" in answer and "사실 설명 여부: 예" in answer:
                validated_evidence.append({
                    "url": task_meta["url"],
                    "snippet": task_meta["snippet"],
                    "judgment": answer,
                    "source_title": task_meta.get("source_title")
                })
                seen_urls_for_claim.add(task_meta["url"])

        criteria_scores = {
            "근거의 명확성": 5 if validated_evidence else 0,
            "출처의 신뢰도": 5 if validated_evidence else 0,
            "교차 검증 여부": 5 if len(validated_evidence) >= 3 else (3 if len(validated_evidence) == 2 else (1 if len(validated_evidence) == 1 else 0)),
            "주장의 구체성": 5,
            "출처의 다양성": calculate_source_diversity_score(validated_evidence)
        }
        confidence_score = calculate_fact_check_confidence(criteria_scores)

        outputs.append({
            "claim": claim,
            "result": "likely_true" if validated_evidence else "insufficient_evidence",
            "confidence_score": confidence_score,
            "evidence": validated_evidence[:3]
        })

    avg_score = round(sum(o['confidence_score'] for o in outputs) / len(outputs)) if outputs else 0
    evidence_ratio = sum(1 for o in outputs if o["result"] == "likely_true") / len(outputs) if outputs else 0
    if len(outputs) < 3:
        summary = f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
    elif evidence_ratio < 0.3:
        summary = f"신뢰도 낮음 (증거 확보 주장 비율: {evidence_ratio*100:.1f}%)"
    else:
        summary = f"신뢰도: {avg_score}% (증거 확보된 주장 비율: {evidence_ratio*100:.1f}%)"

    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    channel_type, reason = parse_channel_type(classification.content)

    return {
        "video_id": video_id,
        "video_total_confidence_score": avg_score,
        "confidence_summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason,
        "claims": outputs
    }
