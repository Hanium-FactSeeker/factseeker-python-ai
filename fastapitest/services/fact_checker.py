# core/fact_checker.py
import os
import re
import json
import asyncio
import logging
import shutil
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# 프로젝트 구조에 맞춘 임포트
from core.lambdas import (
    extract_video_id,
    fetch_youtube_transcript,
    search_news_google_cs,
    get_article_text,
    clean_news_title,
    calculate_fact_check_confidence,
    calculate_source_diversity_score,
)
from core.llm_chains import (
    build_claim_extractor,
    build_claim_summarizer,
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier,
)
from core.faiss_manager import CHUNK_CACHE_DIR

logger = logging.getLogger(__name__)

# --- 설정값 ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
MAX_CLAIMS_TO_FACT_CHECK = int(os.environ.get("MAX_CLAIMS_TO_FACT_CHECK", 10))
MAX_ARTICLES_PER_CLAIM = int(os.environ.get("MAX_ARTICLES_PER_CLAIM", 10))
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", 0.8))
# --- 끝 ---

# 임베딩 모델
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500
)

# URL별 동시 처리 방지
_url_locks: Dict[str, asyncio.Lock] = {}

# --- URL 기반 캐시 경로 ---
def _url_to_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def _article_faiss_path(url: str) -> str:
    return os.path.join(CHUNK_CACHE_DIR, _url_to_cache_key(url))

# S3는 선택(여기선 로컬만)
def _upload_to_s3(local_dir: str, s3_key: str) -> None:
    # 외부 환경 의존 제거: 실제 운영에서만 사용하도록 주석/No-op
    return

def _download_from_s3(local_dir: str, s3_key: str) -> bool:
    return False

# --- 기사 URL 기준 FAISS 생성/로드 (Lock) ---
async def _ensure_article_faiss(url: str) -> Optional[FAISS]:
    lock = _url_locks.setdefault(url, asyncio.Lock())
    async with lock:
        local_path = _article_faiss_path(url)
        faiss_path = os.path.join(local_path, "index.faiss")
        pkl_path = os.path.join(local_path, "index.pkl")
        s3_key = f"article_faiss_cache/{_url_to_cache_key(url)}"

        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            try:
                logger.info(f"캐시 재사용 (잠금 후 확인): {url}")
                return FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
                logger.warning(f"로컬 캐시 손상, 재생성 시도: {e}")

        if _download_from_s3(local_path, s3_key):
            try:
                logger.info(f"S3 캐시 재사용 (잠금 후 확인): {url}")
                return FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
                logger.warning(f"S3 캐시 손상, 재생성 시도: {e}")

        logger.info(f"캐시 없음, 신규 크롤링 시작: {url}")
        text = await get_article_text(url)
        if not text or len(text) < 200:
            return None

        doc = Document(page_content=text, metadata={"url": url})
        faiss_db = FAISS.from_documents([doc], embed_model)
        os.makedirs(local_path, exist_ok=True)
        faiss_db.save_local(local_path)

        try:
            _upload_to_s3(local_path, s3_key)
            logger.info(f"✅ S3 업로드 성공: {s3_key}")
        except Exception as e:
            logger.warning(f"S3 업로드 실패(무시): {e}")

        return faiss_db

# --- CSE 검색 → FAISS에서 제목 근접검색으로 URL 매칭 ---
async def _search_and_retrieve_docs_once(
    claim: str,
    faiss_partition_dirs: List[str],
    seen_urls: set
) -> List[Document]:
    summarizer = build_claim_summarizer()
    try:
        summary_result = await summarizer.ainvoke({"claim": claim})
        summarized_query = (summary_result.content or "").strip() or claim
        logger.info(f"🔍 요약된 검색어: '{summarized_query}'")
    except Exception as e:
        logger.error(f"Claim 요약 실패: {e}, 원문으로 검색 진행")
        summarized_query = claim

    search_results = await search_news_google_cs(summarized_query)
    cse_titles = [clean_news_title(item.get('title', '') or "") for item in search_results[:20]]
    cse_raw_titles = [item.get('title', '') or "" for item in search_results[:20]]
    cse_urls = [item.get('link') for item in search_results[:20]]

    if not cse_titles:
        logger.warning("Google 검색 결과에서 제목을 찾을 수 없어 탐색을 종료합니다.")
        return []

    cse_title_embs = embed_model.embed_documents(cse_titles)
    search_vectors = np.array(cse_title_embs, dtype=np.float32)
    matched_urls: Dict[str, Dict[str, str]] = {}

    for faiss_dir in faiss_partition_dirs:
        try:
            title_faiss_db = FAISS.load_local(
                faiss_dir, embeddings=embed_model, allow_dangerous_deserialization=True
            )
            if getattr(title_faiss_db.index, "ntotal", 0) == 0:
                continue

            D, I = title_faiss_db.index.search(search_vectors, k=3)
            for j in range(len(cse_title_embs)):
                for i, dist in enumerate(D[j]):
                    if dist < DISTANCE_THRESHOLD:
                        faiss_idx = I[j][i]
                        docstore_id = title_faiss_db.index_to_docstore_id[faiss_idx]
                        doc = title_faiss_db.docstore._dict[docstore_id]
                        url = doc.metadata.get("url")
                        if url and url not in seen_urls and url not in matched_urls:
                            matched_urls[url] = {
                                "matched_cse_title": cse_titles[j],
                                "raw_cse_title": cse_raw_titles[j]
                            }
        except Exception as e:
            logger.error(f"FAISS 검색 실패: {faiss_dir} → {e}")

    docs: List[Document] = []
    for url in list(matched_urls.keys()):
        faiss_db = await _ensure_article_faiss(url)
        if faiss_db:
            # 단일 문서만 사용
            for d in faiss_db.docstore._dict.values():
                actual_url = d.metadata.get("url")
                if actual_url and actual_url not in seen_urls:
                    docs.append(Document(
                        page_content=d.page_content,
                        metadata={
                            "url": actual_url,
                            "matched_cse_title": matched_urls[actual_url]["matched_cse_title"],
                            "raw_cse_title": matched_urls[actual_url]["raw_cse_title"]
                        }
                    ))
                    break
    return docs


async def _search_and_retrieve_docs(claim: str, faiss_partition_dirs: List[str]) -> List[Document]:
    collected_docs: List[Document] = []
    seen_urls: set = set()
    attempt = 0

    while len(collected_docs) < MAX_ARTICLES_PER_CLAIM:
        attempt += 1
        logger.info(f"🔁 뉴스 수집 시도 {attempt}회 - 확보된 기사 수: {len(collected_docs)}")
        new_docs = await _search_and_retrieve_docs_once(claim, faiss_partition_dirs, seen_urls)
        if not new_docs:
            logger.warning(f"📭 수집된 문서 없음. 반복 진행 중... (현재 확보: {len(collected_docs)})")

        for doc in new_docs:
            url = doc.metadata.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                collected_docs.append(doc)
                logger.info(f"✅ 기사 확보: {url} ({len(collected_docs)}/{MAX_ARTICLES_PER_CLAIM})")

        if attempt > 30:
            logger.error("🚨 30회 이상 반복했지만 목표 개수 확보 실패. 중단합니다.")
            break

    logger.info(f"📰 최종 확보된 문서 수: {len(collected_docs)}개")
    return collected_docs[:MAX_ARTICLES_PER_CLAIM]


def _parse_channel_type(llm_output: str):
    channel_type_match = re.search(r"채널 유형:\s*(.+)", llm_output or "")
    reason_match = re.search(r"분류 근거:\s*(.+)", llm_output or "")
    channel_type = channel_type_match.group(1).strip() if channel_type_match else "알 수 없음"
    reason = reason_match.group(1).strip() if reason_match else "LLM 응답에서 판단 근거를 찾을 수 없습니다."
    return channel_type, reason


# ------------------------------------------------------------------------------
# 외부 진입점
# ------------------------------------------------------------------------------
async def run_fact_check(youtube_url: str, faiss_partition_dirs: List[str]) -> Dict[str, Any]:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    logger.info(f"유튜브 분석 시작: {youtube_url}")
    try:
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}

        extractor = build_claim_extractor()
        result = await extractor.ainvoke({"transcript": transcript})
        claims = [line.strip() for line in (result.content or "").strip().split("\n") if line.strip()]

        if not claims:
            return {
                "video_id": video_id, "video_url": youtube_url,
                "video_total_confidence_score": 0, "claims": []
            }

        reducer = build_reduce_similar_claims_chain()
        claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
        reduced_result = await reducer.ainvoke({"claims_json": claims_json})

        claims_to_check: List[str] = []
        try:
            json_match = re.search(r"```json\s*(\[.*?\])\s*```", reduced_result.content or "", re.DOTALL)
            if json_match:
                claims_to_check = json.loads(json_match.group(1))
            else:
                claims_to_check = [line.strip() for line in (reduced_result.content or "").strip().split("\n") if line.strip()]
        except json.JSONDecodeError:
            claims_to_check = [
                line.strip() for line in (reduced_result.content or "").strip().split("\n")
                if line.strip() and not line.strip().startswith(("```json", "```", "[", "]"))
            ]

        claims_to_check = claims_to_check[:MAX_CLAIMS_TO_FACT_CHECK]
        logger.info(f"✂️ 중복 제거 후 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")

        if not claims_to_check:
            return {
                "video_id": video_id, "video_url": youtube_url,
                "video_total_confidence_score": 0, "claims": []
            }

    except Exception as e:
        logger.exception(f"주장 추출/정제 중 오류: {e}")
        return {"error": f"Failed to extract claims: {e}"}

    fact_checker = build_factcheck_chain()

    async def process_claim_step(idx: int, claim: str) -> Dict[str, Any]:
        try:
            logger.info(f"--- 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")
            docs = await _search_and_retrieve_docs(claim, faiss_partition_dirs)
            if not docs:
                logger.info(f"근거 문서를 찾지 못함: '{claim}'")
                return {
                    "claim": claim, "result": "insufficient_evidence",
                    "confidence_score": 0, "evidence": []
                }

            url_set = set()

            async def factcheck_doc(doc: Document) -> Optional[Dict[str, Any]]:
                try:
                    check_result = await fact_checker.ainvoke({"claim": claim, "context": doc.page_content})
                    result_content = check_result.content or ""
                    relevance = re.search(r"관련성:\s*(.+)", result_content)
                    fact_check_result_match = re.search(r"사실 설명 여부:\s*(.+)", result_content)
                    justification = re.search(r"간단한 설명:\s*(.+)", result_content)
                    snippet = re.search(r"핵심 근거 문장:\s*(.+)", result_content)
                    url = doc.metadata.get("url")

                    # "관련성: 예" + URL 중복 제거만 통과
                    if (
                        relevance and "예" in relevance.group(1)
                        and fact_check_result_match and justification
                        and url and url not in url_set
                    ):
                        url_set.add(url)
                        return {
                            "url": url,
                            "relevance": "yes",
                            "fact_check_result": fact_check_result_match.group(1).strip(),
                            "justification": justification.group(1).strip(),
                            "snippet": (snippet.group(1).strip() if snippet else "")
                        }
                except Exception as e:
                    logger.error(f"    - LLM 팩트체크 체인 실행 중 오류: {e}")
                return None

            tasks = [factcheck_doc(doc) for doc in docs]
            factcheck_results = await asyncio.gather(*tasks, return_exceptions=True)

            validated: List[Dict[str, Any]] = []
            for r in factcheck_results:
                if isinstance(r, Exception):
                    logger.error(f"팩트체크 task 예외: {r}")
                    continue
                if r:
                    validated.append(r)

            # 안전한 점수 계산 (예외 방지)
            try:
                diversity_score = calculate_source_diversity_score(validated)
            except Exception as e:
                logger.exception("diversity score 계산 실패")
                diversity_score = 0

            try:
                confidence_score = calculate_fact_check_confidence({
                    "source_diversity": diversity_score,
                    "evidence_count": min(len(validated), 5)
                })
            except Exception as e:
                logger.exception("confidence 계산 실패")
                confidence_score = 0

            logger.info(f"--- 팩트체크 완료: '{claim}' -> 신뢰도: {confidence_score}%")

            return {
                "claim": claim,
                "result": "likely_true" if validated else "insufficient_evidence",
                "confidence_score": confidence_score,
                "evidence": validated[:3]
            }
        except Exception as e:
            logger.exception(f"process_claim_step 예외: {e}")
            # 절대 결과가 증발하지 않도록 기본값 반환
            return {
                "claim": claim,
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": []
            }

    claim_tasks = [process_claim_step(i, c) for i, c in enumerate(claims_to_check)]
    raw_outputs = await asyncio.gather(*claim_tasks, return_exceptions=True)

    outputs: List[Dict[str, Any]] = []
    for o in raw_outputs:
        if isinstance(o, Exception):
            logger.exception(f"claim task 예외(기본값 대체): {o}")
            outputs.append({
                "claim": "(unknown)",
                "result": "insufficient_evidence",
                "confidence_score": 0,
                "evidence": []
            })
        else:
            outputs.append(o)

    if outputs:
        try:
            avg_score = round(sum(o.get('confidence_score', 0) for o in outputs) / len(outputs))
        except Exception:
            avg_score = 0
        try:
            evidence_ratio = sum(1 for o in outputs if o.get("result") == "likely_true") / len(outputs)
            summary = f"증거 확보된 주장 비율: {evidence_ratio*100:.1f}%" if len(outputs) >= 3 else f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
        except Exception:
            summary = "결과 요약 생성 실패"
    else:
        avg_score = 0
        summary = "결과 없음"

    # 채널 유형 분류(예외 보호)
    channel_type = "알 수 없음"
    reason = "LLM 응답에서 판단 근거를 찾을 수 없습니다."
    try:
        classifier = build_channel_type_classifier()
        classification = await classifier.ainvoke({"transcript": transcript})
        channel_type, reason = _parse_channel_type(classification.content or "")
    except Exception as e:
        logger.warning(f"채널 유형 분류 실패: {e}")

    return {
        "video_id": video_id,
        "video_url": youtube_url,
        "video_total_confidence_score": avg_score,
        "claims": outputs,
        "summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason
    }
