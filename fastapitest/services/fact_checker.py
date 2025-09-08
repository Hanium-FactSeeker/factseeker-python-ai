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
from datetime import datetime

from core.lambdas import (
    extract_video_id,
    fetch_youtube_transcript,
    search_news_naver_api,
    search_news_google_cs,
    get_article_text,
    clean_news_title,
    calculate_fact_check_confidence,
    calculate_source_diversity_score,
    clean_evidence_json
)
from core.llm_chains import (
    build_claim_extractor,
    build_claim_summarizer,
    build_factcheck_chain,
    build_reduce_similar_claims_chain,
    build_channel_type_classifier,
    get_chat_llm,
    build_keyword_extractor_chain,
    build_three_line_summarizer_chain
)
from core.faiss_manager import CHUNK_CACHE_DIR

# --- 설정값 ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
MAX_CLAIMS_TO_FACT_CHECK = 10
MAX_ARTICLES_PER_CLAIM = 10  # ✨ 주장당 최대 검색 기사 수 (이 값을 조절하세요) ✨
DISTANCE_THRESHOLD = 0.8

# 동시 처리 제한 (병목 완화)
MAX_CONCURRENT_CLAIMS = int(os.environ.get("MAX_CONCURRENT_CLAIMS", "3"))
MAX_CONCURRENT_FACTCHECKS = int(os.environ.get("MAX_CONCURRENT_FACTCHECKS", "7"))
MAX_EVIDENCES_PER_CLAIM = int(os.environ.get("MAX_EVIDENCES_PER_CLAIM", "10"))
# 파티션 검색 조기 종료 임계치: 최신 파티션에서 이 개수 이상 확보되면 다음 파티션으로 가지 않음
PARTITION_STOP_HITS = int(os.environ.get("PARTITION_STOP_HITS", "1"))
# --- 설정값 끝 ---

# --- LLM 거절(정책상) 응답 감지 유틸 ---
_POLICY_REFUSAL_PATTERNS = [
    r"sorry[, ]?i can['’]t",  # sorry I can't
    r"i cannot (?:help|assist) with that request",
    r"cannot (?:help|assist) with that",
    r"not able to (?:help|assist) with that",
    r"as an ai,? i (?:cannot|can't)",
    r"i (?:cannot|can't) comply with that",
    r"policy (?:restricts|prevents) me",
    # Korean variants
    r"도와드릴 수 없습니다",
    r"지원해 드릴 수 없",
    r"요청(?:을)? 수행할 수 없",
    r"정책상 (?:제공|지원|응답)할 수 없",
]

def _is_policy_refusal(text: str) -> bool:
    if not text:
        return False
    import re as _re
    low = text.strip().lower()
    for pat in _POLICY_REFUSAL_PATTERNS:
        if _re.search(pat, low):
            return True
    return False

try:
    s3 = boto3.client('s3')
except Exception as e:
    s3 = None
    logging.error(f"S3 클라이언트 초기화 실패: {e}")

embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500
)

# URL별 동시 처리를 막기 위한 잠금(Lock) 객체
url_locks = {}


# --- URL 기반 캐시 경로 및 S3 동기화 ---
def url_to_cache_key(url):
    """Create a stable cache key from a URL-like value.

    Be defensive: metadata from external indices can sometimes contain
    non-string values (e.g., float NaN). Convert to string to avoid
    AttributeError from calling .encode() on non-strings.
    """
    if url is None:
        raw = ""
    elif not isinstance(url, str):
        raw = str(url)
    else:
        raw = url
    return hashlib.md5(raw.encode()).hexdigest()

def get_article_faiss_path(url):
    return os.path.join(CHUNK_CACHE_DIR, url_to_cache_key(url))

def upload_to_s3(local_dir, s3_key):
    if not s3:
        logging.warning("S3 클라이언트가 없어 업로드를 건너뜁니다.")
        return
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(s3_key, file)
            try:
                s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
            except ClientError as e:
                logging.error(f"S3 업로드 실패: {local_path} -> s3://{S3_BUCKET_NAME}/{s3_path} - {e}")
                raise

def download_from_s3(local_dir, s3_key):
    if not s3:
        logging.warning("S3 클라이언트가 없어 다운로드를 건너뜁니다.")
        return False
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key)
        if 'Contents' not in response:
            return False
        os.makedirs(local_dir, exist_ok=True)
        for obj in response['Contents']:
            s3_path = obj['Key']
            relative_path = os.path.relpath(s3_path, s3_key)
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(S3_BUCKET_NAME, s3_path, local_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 다운로드 중 오류 발생: s3://{S3_BUCKET_NAME}/{s3_key} - {e}")
        return False

# --- 기사 URL 기준 FAISS 생성/로드 (잠금 로직 적용) ---
async def ensure_article_faiss(url):
    """(잠금 기능 추가) 기사 본문을 벡터화하고 캐시(S3/로컬)에 저장/로드"""
    lock = url_locks.setdefault(url, asyncio.Lock())
    async with lock:
        cache_key = url_to_cache_key(url)
        local_path = get_article_faiss_path(url)
        faiss_path = os.path.join(local_path, "index.faiss")
        pkl_path = os.path.join(local_path, "index.pkl")
        s3_key = f"article_faiss_cache/{cache_key}"

        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            try:
                logging.info(f"캐시 재사용 (잠금 후 확인): {url}")
                return FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
                logging.warning(f"로컬 캐시 손상, 재생성 시도: {e}")
        
        if s3 is not None and download_from_s3(local_path, s3_key):
            try:
                logging.info(f"S3 캐시 재사용 (잠금 후 확인): {url}")
                return FAISS.load_local(local_path, embed_model, allow_dangerous_deserialization=True)
            except Exception as e:
                shutil.rmtree(local_path, ignore_errors=True)
                logging.warning(f"S3 캐시 손상, 재생성 시도: {e}")
        
        logging.info(f"캐시 없음, 신규 크롤링 시작: {url}")
        text = await get_article_text(url)
        if not text or len(text) < 200:
            return None
            
        doc = Document(page_content=text, metadata={"url": url})
        faiss_db = FAISS.from_documents([doc], embed_model)
        faiss_db.save_local(local_path)
        
        if s3 is not None:
            try:
                upload_to_s3(local_path, s3_key)
                logging.info(f"✅ S3 업로드 성공: {s3_key}")
            except Exception as e:
                logging.warning(f"S3 업로드 실패: {e}")
                
        return faiss_db

# --- CSE → FAISS에서 여러 기사, url 기준 중복 없는 문서만 수집 (한도 즉시 중단) ---
async def search_and_retrieve_docs_once(claim, faiss_partition_dirs, seen_urls, use_google_cse=False):
    def _is_valid_url(value) -> bool:
        return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))
    summarizer = build_claim_summarizer()
    try:
        summary_result = await summarizer.ainvoke({"claim": claim})
        summarized_query = summary_result.content.strip()
        logging.info(f"🔍 요약된 검색어: '{summarized_query}'")
    except Exception as e:
        logging.error(f"Claim 요약 실패: {e}, 원문으로 검색 진행")
        summarized_query = claim

    # 파티션은 호출 시 전달된 목록을 그대로 사용 (최신 → 과거)

    # 신뢰도가 0일 경우 Google CSE 사용, 그렇지 않으면 네이버 API 사용
    if use_google_cse:
        logging.info("🔄 신뢰도 0으로 인한 Google CSE 재시도")
        search_results = await search_news_google_cs(summarized_query)
        # Google CSE 결과를 네이버 API 형식으로 변환
        cse_titles = []
        cse_raw_titles = []
        for item in search_results[:10]:
            title = item.get('title', '')
            snippet = item.get('snippet', '').replace("**", "")
            cse_titles.append(clean_news_title(title))
            cse_raw_titles.append(title)
    else:
        search_results = await search_news_naver_api(summarized_query)
        # CSE 상위 10개만 사용
        cse_titles = [clean_news_title(item.get('title', '')) for item in search_results[:10]]
        cse_raw_titles = [item.get('title', '') for item in search_results[:10]]
    

    if not cse_titles:
        logging.warning("네이버 검색 결과에서 제목을 찾을 수 없어 탐색을 종료합니다.")
        return []

    # 임베딩 호출 안정화: 소규모 재시도/예외 보호
    def _embed_docs_with_retry(texts, retries=1):
        delay = 0.5
        for attempt in range(retries + 1):
            try:
                return embed_model.embed_documents(texts)
            except Exception as e:
                logging.warning(f"임베딩 실패(재시도 {attempt}/{retries}): {e}")
                if attempt < retries:
                    import time as _t
                    _t.sleep(delay)
                    delay *= 2
                else:
                    return []

    cse_title_embs = _embed_docs_with_retry(cse_titles, retries=1)
    search_vectors = np.array(cse_title_embs, dtype=np.float32)
    # 선착 글로벌 조기중단을 위해 즉시 누적
    article_urls = []
    matched_meta = {}
    chosen_for_cse = set()
    seen_tmp = set()

    # 파티션 디렉터리의 숫자가 클수록 우선 (최신)
    def partition_num(path: str) -> int:
        import os, re
        base = os.path.basename(path)
        m = re.search(r'(\d+)', base)
        return int(m.group(1)) if m else -1

    stop = False
    for faiss_dir in sorted(faiss_partition_dirs, key=partition_num, reverse=True):
        if stop:
            break
        try:
            before_count = len(article_urls)
            title_faiss_db = FAISS.load_local(
                faiss_dir, embeddings=embed_model, allow_dangerous_deserialization=True
            )
            if title_faiss_db.index.ntotal == 0:
                continue

            D, I = title_faiss_db.index.search(search_vectors, k=3)
            # CSE 순서대로, 각 제목에서 첫 유효 URL만 채택 (후보를 안정 정렬 후 선택)
            for j in range(len(cse_title_embs)):
                if j in chosen_for_cse:
                    continue
                # 후보 수집: 임계값을 통과한 것만 모음
                candidates = []
                for i, dist in enumerate(D[j]):
                    if dist < DISTANCE_THRESHOLD:
                        faiss_idx = I[j][i]
                        docstore_id = title_faiss_db.index_to_docstore_id[faiss_idx]
                        doc = title_faiss_db.docstore._dict.get(docstore_id)
                        if not doc:
                            continue
                        url = (doc.metadata or {}).get("url")
                        # 정렬을 위한 안전한 기본값
                        url_key = url if isinstance(url, str) else ""
                        candidates.append((float(dist), str(docstore_id), url_key, url))

                # 안정 정렬: 거리 오름차순 → docstore_id → url 사전순
                candidates.sort(key=lambda t: (t[0], t[1], t[2]))

                # 정렬된 후보 중 첫 유효 URL 선택
                for _dist, _doc_id, _url_key, url in candidates:
                    if _is_valid_url(url) and url not in seen_urls and url not in seen_tmp:
                        article_urls.append(url)
                        matched_meta[url] = {
                            "matched_cse_title": cse_titles[j],
                            "raw_cse_title": cse_raw_titles[j]
                        }
                        seen_tmp.add(url)
                        chosen_for_cse.add(j)
                        if len(article_urls) >= MAX_ARTICLES_PER_CLAIM:
                            stop = True
                        break
                if stop:
                    break
            # 최신 파티션에서 일정 수 이상 확보되었으면 다음 파티션으로 진행하지 않고 종료
            if len(article_urls) - before_count >= PARTITION_STOP_HITS:
                stop = True
        except Exception as e:
            logging.error(f"FAISS 검색 실패: {faiss_dir} → {e}")

    # Fallback: CSE 기반 매칭이 한 건도 없으면, 요약 키워드 자체로 제목 FAISS를 직접 검색
    if not article_urls:
        try:
            logging.info("네이버 검색 기반 매칭 결과 없음 → 키워드 직접 FAISS 검색 시도")
            # 임베딩 쿼리도 재시도/예외 보호
            def _embed_query_with_retry(q, retries=1):
                delay = 0.5
                for attempt in range(retries + 1):
                    try:
                        return embed_model.embed_query(q)
                    except Exception as e:
                        logging.warning(f"임베딩 쿼리 실패(재시도 {attempt}/{retries}): {e}")
                        if attempt < retries:
                            import time as _t
                            _t.sleep(delay)
                            delay *= 2
                        else:
                            return None

            emb = _embed_query_with_retry(summarized_query, retries=1)
            if emb is None:
                return []
            query_vec = emb
            query_np = np.array([query_vec], dtype=np.float32)
            fallback = {}

            # 최신 파티션 우선
            for faiss_dir in sorted(faiss_partition_dirs, key=partition_num, reverse=True):
                try:
                    title_faiss_db = FAISS.load_local(
                        faiss_dir, embeddings=embed_model, allow_dangerous_deserialization=True
                    )
                    if title_faiss_db.index.ntotal == 0:
                        continue
                    D, I = title_faiss_db.index.search(query_np, k=5)
                    for i, dist in enumerate(D[0]):
                        if dist < DISTANCE_THRESHOLD:
                            faiss_idx = I[0][i]
                            docstore_id = title_faiss_db.index_to_docstore_id[faiss_idx]
                            doc = title_faiss_db.docstore._dict[docstore_id]
                            url = doc.metadata.get("url")
                            if not _is_valid_url(url):
                                continue
                            cur = fallback.get(url)
                            if (cur is None) or (dist < cur["dist"]):
                                fallback[url] = {"dist": float(dist)}
                except Exception as e:
                    logging.error(f"FAISS 키워드 검색 실패: {faiss_dir} → {e}")

            if fallback:
                # 거리 오름차순, 동률 시 URL 사전순으로 정렬 후 상한 적용
                article_urls = [u for u, _ in sorted(fallback.items(), key=lambda kv: (kv[1]["dist"], kv[0]))][:MAX_ARTICLES_PER_CLAIM]
                # 메타데이터에는 요약 질의를 기록
                for u in article_urls:
                    matched_meta[u] = {
                        "matched_cse_title": summarized_query,
                        "raw_cse_title": summarized_query,
                    }
        except Exception as e:
            logging.error(f"키워드 직접 FAISS 검색 중 오류: {e}")
    docs = []
    for url in article_urls:
        faiss_db = await ensure_article_faiss(url)
        if faiss_db:
            for doc in faiss_db.docstore._dict.values():
                actual_url = doc.metadata.get("url")
                if _is_valid_url(actual_url) and actual_url not in seen_urls:
                    meta = matched_meta.get(actual_url, {"matched_cse_title": "", "raw_cse_title": ""})
                    docs.append(Document(
                        page_content=doc.page_content,
                        metadata={
                            "url": actual_url,
                            "matched_cse_title": meta["matched_cse_title"],
                            "raw_cse_title": meta["raw_cse_title"]
                        }
                    ))
                    break
    return docs

# --- 본문 확보된 뉴스 15개가 될 때까지 반복 확보 ---
async def search_and_retrieve_docs(claim, faiss_partition_dirs, use_google_cse=False):
    collected_docs = []
    seen_urls = set()
    attempt_count = 0

    while len(collected_docs) < MAX_ARTICLES_PER_CLAIM:
        attempt_count += 1
        logging.info(f"🔁 뉴스 수집 시도 {attempt_count}회 - 확보된 기사 수: {len(collected_docs)}")

        new_docs = await search_and_retrieve_docs_once(claim, faiss_partition_dirs, seen_urls, use_google_cse)

        if not new_docs:
            logging.warning(f"📭 수집된 문서 없음. 반복 진행 중... (현재 확보: {len(collected_docs)})")

        for doc in new_docs:
            url = doc.metadata.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                collected_docs.append(doc)
                logging.info(f"✅ 기사 확보: {url} ({len(collected_docs)}/{MAX_ARTICLES_PER_CLAIM})")

        if attempt_count > 30 and len(collected_docs) < MAX_ARTICLES_PER_CLAIM:
            logging.error("🚨 30회 이상 반복했지만 15개 확보 실패. 중단합니다.")
            break

    logging.info(f"📰 최종 확보된 문서 수: {len(collected_docs)}개")
    return collected_docs[:MAX_ARTICLES_PER_CLAIM]

async def run_fact_check(youtube_url, faiss_partition_dirs):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    logging.info(f"유튜브 분석 시작: {youtube_url}")
    try:
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}

        # Extract keywords and summary
        keyword_extractor = build_keyword_extractor_chain()
        summarizer = build_three_line_summarizer_chain()

        keywords_task = keyword_extractor.ainvoke({"text": transcript})
        summary_task = summarizer.ainvoke({"text": transcript})

        keywords_result, summary_result = await asyncio.gather(keywords_task, summary_task)

        # 키워드는 JSON 배열로 반환되도록 체인을 수정했으므로 배열 파싱을 우선 시도합니다.
        extracted_keywords_raw = (keywords_result.content or "").strip()
        extracted_keywords_list = []
        try:
            m = re.search(r"\[.*\]", extracted_keywords_raw, re.DOTALL)
            json_text = m.group(0) if m else extracted_keywords_raw
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                extracted_keywords_list = [str(k).strip() for k in parsed if str(k).strip()]
        except Exception:
            extracted_keywords_list = [s.strip() for s in extracted_keywords_raw.split(',') if s.strip()]

        three_line_summary = summary_result.content.strip() if summary_result.content else ""

        extractor = build_claim_extractor()
        result = await extractor.ainvoke({"transcript": transcript})
        raw_extract = (result.content or "").strip()
        if _is_policy_refusal(raw_extract):
            # 추출 단계에서 정책상 거절이 발생한 경우: 안내 문구를 단일 주장으로 표시
            claims = ["AI 정책상 넘어갑니다."]
        else:
            claims = [line.strip() for line in raw_extract.split('\n') if line.strip()]

        if not claims:
            return {
                "video_id": video_id, "video_url": youtube_url,
                "video_total_confidence_score": 0, "claims": []
            }

        reducer = build_reduce_similar_claims_chain()
        claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
        reduced_result = await reducer.ainvoke({"claims_json": claims_json})

        claims_to_check = []
        try:
            json_match = re.search(r"```json\s*(\[.*?\])\s*```", reduced_result.content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
                claims_to_check = json.loads(json_content)
            else:
                claims_to_check = [line.strip() for line in reduced_result.content.strip().split('\n') if line.strip()]
        except json.JSONDecodeError:
            claims_to_check = [
                line.strip() for line in reduced_result.content.strip().split('\n')
                if line.strip() and not line.strip().startswith(('```json', '```', '[', ']'))
            ]

        # 모든 주장을 그대로 팩트체크 대상으로 사용 (상한 제거)
        logging.info(f"✂️ 중복 제거 후 최종 팩트체크 대상 주장 {len(claims_to_check)}개: {claims_to_check}")

        if not claims_to_check:
            return {
                "video_id": video_id, "video_url": youtube_url,
                "video_total_confidence_score": 0, "claims": []
            }

    except Exception as e:
        logging.exception(f"주장 추출/정제 중 오류: {e}")
        return {"error": f"Failed to extract claims: {e}"}

    # 주장을 처리하는 단계를 정의하고, 주장 단위 동시성은 외부 세마포어로 제어합니다.
    async def process_claim_step(idx, claim):
        logging.info(f"--- 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")

        url_set = set()  # 같은 URL에서 중복 증거 방지
        validated_evidence = []
        policy_skipped_count = 0
        fact_checker = build_factcheck_chain()

        # 개별 팩트체크 함수 (이전 방식으로 복원)
        async def factcheck_doc(doc):
            try:
                check_result = await fact_checker.ainvoke({"claim": claim, "context": doc.page_content})
                result_content = check_result.content or ""
                # 정책상 거절 응답 감지 → 사용자 공지용 플래그 반환
                if _is_policy_refusal(result_content):
                    return {"policy_skipped": True, "policy_notice": "AI 정책상 넘어갑니다."}
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
                        "url": url, "relevance": "yes",
                        "fact_check_result": fact_check_result_match.group(1).strip(),
                        "justification": justification.group(1).strip(),
                        "snippet": snippet.group(1).strip() if snippet else ""
                    }
            except Exception as e:
                logging.error(f"    - LLM 팩트체크 체인 실행 중 오류: {e}")
            return None

        # per-claim 문서 팩트체크 동시성 제한
        factcheck_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FACTCHECKS)

        async def limited_factcheck_doc(doc):
            async with factcheck_semaphore:
                return await factcheck_doc(doc)

        # CSE 검색은 한 번만 수행하고, 해당 결과에서 매칭된 문서만 배치로 팩트체크
        new_docs = await search_and_retrieve_docs_once(claim, faiss_partition_dirs, set())
        if not new_docs:
            logging.info(f"근거 문서를 찾지 못함: '{claim}'")
            return {
                "claim": claim, "result": "insufficient_evidence",
                "confidence_score": 0, "evidence": []
            }

        # 배치 처리로 조기 종료 가능하게 함
        for i in range(0, len(new_docs), MAX_CONCURRENT_FACTCHECKS):
            if len(validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                break
            batch = new_docs[i:i+MAX_CONCURRENT_FACTCHECKS]
            factcheck_tasks = [limited_factcheck_doc(doc) for doc in batch]
            factcheck_results = await asyncio.gather(*factcheck_tasks)
            for res in factcheck_results:
                if res and isinstance(res, dict):  # None이 아니고 dict 타입인지 확인
                    if res.get("policy_skipped"):
                        policy_skipped_count += 1
                        logging.info(f"⛔ 정책상 스킵됨: '{claim}'")
                        continue
                    validated_evidence.append(res)
                    logging.info(f"✅ [네이버] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                    if len(validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                        break

        # 최종 결과 계산
        diversity_score = calculate_source_diversity_score(validated_evidence)
        confidence_score = calculate_fact_check_confidence({
            "source_diversity": diversity_score,
            "evidence_count": min(len(validated_evidence), 5)
        })

        logging.info(f"--- 팩트체크 완료: '{claim}' -> 신뢰도: {confidence_score}% (증거 {len(validated_evidence)}개)")

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
                    batch = google_docs[i:i+MAX_CONCURRENT_FACTCHECKS]
                    factcheck_tasks = [limited_factcheck_doc(doc) for doc in batch]
                    factcheck_results = await asyncio.gather(*factcheck_tasks)
                    for res in factcheck_results:
                        if res and isinstance(res, dict):  # None이 아니고 dict 타입인지 확인
                            if res.get("policy_skipped"):
                                policy_skipped_count += 1
                                logging.info(f"⛔ 정책상 스킵됨: '{claim}' [Google CSE]")
                                continue
                            google_validated_evidence.append(res)
                            logging.info(f"✅ [Google CSE] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                            if len(google_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                                break
                
                # Google CSE 결과로 신뢰도 계산
                if google_validated_evidence:
                    google_diversity_score = calculate_source_diversity_score(google_validated_evidence)
                    google_confidence = calculate_fact_check_confidence({
                        "source_diversity": google_diversity_score,
                        "evidence_count": min(len(google_validated_evidence), 5)
                    })
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
        
        if confidence_score <= 20:
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
                        batch = partition_9_docs[i:i+MAX_CONCURRENT_FACTCHECKS]
                        factcheck_tasks = [limited_factcheck_doc(doc) for doc in batch]
                        factcheck_results = await asyncio.gather(*factcheck_tasks)
                        for res in factcheck_results:
                            if res and isinstance(res, dict):
                                if res.get("policy_skipped"):
                                    policy_skipped_count += 1
                                    logging.info(f"⛔ 정책상 스킵됨: '{claim}' [파티션9]")
                                    continue
                                partition_9_validated_evidence.append(res)
                                logging.info(f"✅ [파티션9] 주장 '{claim}' → 증거 URL: {res.get('url', 'N/A')}")
                                if len(partition_9_validated_evidence) >= MAX_EVIDENCES_PER_CLAIM:
                                    break
                    
                    # 파티션 9 결과로 신뢰도 계산
                    if partition_9_validated_evidence:
                        partition_9_diversity_score = calculate_source_diversity_score(partition_9_validated_evidence)
                        partition_9_confidence = calculate_fact_check_confidence({
                            "source_diversity": partition_9_diversity_score,
                            "evidence_count": min(len(partition_9_validated_evidence), 5)
                        })
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

        # 증거 전처리 적용
        cleaned_evidence = clean_evidence_json(final_evidence[:3])
        
        claim_out = f"{claim} (AI 정책상 넘어갑니다.)" if policy_skipped_count > 0 else claim
        result_obj = {
            "claim": claim_out,
            "result": "likely_true" if final_evidence else "insufficient_evidence",
            "confidence_score": final_confidence,
            "evidence": cleaned_evidence
        }
        return result_obj

    # 주장 단위 동시성 제한 (최대 3개 동시 처리)
    claim_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CLAIMS)

    async def limited_process_claim(idx, claim):
        async with claim_semaphore:
            return await process_claim_step(idx, claim)

    claim_tasks = [limited_process_claim(idx, claim) for idx, claim in enumerate(claims_to_check)]
    gathered = await asyncio.gather(*claim_tasks, return_exceptions=True)

    # 예외가 발생해도 주장을 누락하지 않도록 에러 항목으로 기록
    outputs = []
    for i, result in enumerate(gathered):
        if isinstance(result, Exception):
            claim_text = claims_to_check[i] if i < len(claims_to_check) else ""
            # 예외 타입과 상세 메시지를 함께 기록해 공백 메시지 문제 방지
            err_type = type(result).__name__
            err_msg = str(result) or repr(result) or err_type
            logging.error(f"🛑 주장 처리 중 예외 발생: '{claim_text}' -> {err_type}: {err_msg}")
            outputs.append({
                "claim": claim_text,
                "result": "error",
                "confidence_score": 0,
                "evidence": [],
                "error": f"{err_type}: {err_msg}",
                "error_type": err_type,
                "error_stage": "process_claim_step"
            })
        else:
            outputs.append(result)

    if outputs:
        # 가중 평균 계산
        total_weighted_score = 0
        total_weight = 0
        
        for output in outputs:
            confidence = output['confidence_score']
            evidence_count = len(output.get('evidence', []))

            # Strict B: 신뢰도 가중 하한 제거, 상한 3.0 적용. 0% 보정 삭제.
            evidence_weight = min(evidence_count + 1, 5)  # 증거 0개=1, 1개=2, ..., 최대 5
            confidence_weight = min(confidence / 20, 3.0)  # 0.0 ~ 3.0

            weight = evidence_weight * confidence_weight
            total_weighted_score += confidence * weight
            total_weight += weight
        
        # 가중 평균 계산
        if total_weight > 0:
            avg_score = round(total_weighted_score / total_weight)
        else:
            avg_score = 0
            
        evidence_ratio = sum(1 for o in outputs if o["result"] == "likely_true") / len(outputs)
        # 주장 수와 무관하게 항상 비율을 표시
        summary = f"증거 확보된 주장 비율: {evidence_ratio*100:.1f}%"
    else:
        avg_score = 0
        summary = "결과 없음"
    
    classifier = build_channel_type_classifier()
    classification = await classifier.ainvoke({"transcript": transcript})
    channel_type, reason = parse_channel_type(classification.content)

    return {
        "video_id": video_id,
        "video_url": youtube_url,
        "video_total_confidence_score": avg_score,
        "claims": outputs,
        "summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason,
        "created_at": datetime.now().isoformat(),
        "keywords": ", ".join(extracted_keywords_list),
        "three_line_summary": three_line_summary
    }

def parse_channel_type(llm_output: str):
    channel_type_match = re.search(r"채널 유형:\s*(.+)", llm_output)
    reason_match = re.search(r"분류 근거:\s*(.+)", llm_output)
    channel_type = channel_type_match.group(1).strip() if channel_type_match else "알 수 없음"
    reason = reason_match.group(1).strip() if reason_match else "LLM 응답에서 판단 근거를 찾을 수 없습니다."
    return channel_type, reason
