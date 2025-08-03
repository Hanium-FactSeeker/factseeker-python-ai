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

# S3 설정 (FAISS 파티션 로드용)
S3_BUCKET_NAME = "factseeker-faiss-db" # FAISS 인덱스가 저장된 S3 버킷 이름
S3_KEY_PREFIX_BASE_TITLES = "feature_faiss_db_openai_partition/" # 분할된 인덱스를 위한 S3 경로 접두사

# 매니페스트 파일 경로 (build_partitioned_faiss_db.py와 동일하게 설정되어야 함)
S3_MANIFEST_KEY = f"{S3_KEY_PREFIX_BASE_TITLES}faiss_index_manifest.json"

s3_client = boto3.client('s3') # S3 클라이언트 초기화

async def download_faiss_index_from_s3_for_title_partition(s3_partition_prefix: str, local_dir: str):
    """S3에서 특정 파티션의 FAISS 인덱스 파일들을 다운로드합니다."""
    os.makedirs(local_dir, exist_ok=True)
    try:
        # S3_KEY_PREFIX_BASE_TITLES 대신, 매니페스트에서 받은 s3_partition_prefix를 직접 사용
        response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=S3_BUCKET_NAME, Prefix=s3_partition_prefix)
        if 'Contents' not in response:
            logging.warning(f"⚠️ S3 경로 '{s3_partition_prefix}'에 FAISS 인덱스 파일이 없습니다.")
            return False

        for obj in response['Contents']:
            s3_file_key = obj['Key']
            local_file_name = os.path.join(local_dir, os.path.basename(s3_file_key))
            # logging.info(f"    └─ 다운로드: s3://{S3_BUCKET_NAME}/{s3_file_key} -> {local_file_name}") # 너무 많은 로그 방지
            await asyncio.to_thread(s3_client.download_file, S3_BUCKET_NAME, s3_file_key, local_file_name)
        return True
    except Exception as e:
        logging.error(f"❌ FAISS 인덱스 다운로드 중 오류 발생 ('{s3_partition_prefix}'): {e}")
        return False

async def query_partitioned_faiss_titles(query_vector: list, k: int = 3):
    """
    S3 매니페스트 파일에 나열된 모든 FAISS 인덱스 파티션을 순회하며 쿼리를 수행하고 결과를 병합합니다.
    """
    all_results = []
    
    active_partitions = []
    try:
        # 매니페스트 파일 다운로드 및 파티션 목록 읽기
        manifest_obj = await asyncio.to_thread(s3_client.get_object, Bucket=S3_BUCKET_NAME, Key=S3_MANIFEST_KEY)
        manifest_content = manifest_obj['Body'].read().decode('utf-8')
        manifest_data = json.loads(manifest_content)
        active_partitions = manifest_data.get("active_partitions", [])
        logging.info(f"✅ S3 매니페스트에서 {len(active_partitions)}개의 활성 FAISS 파티션 로드.")
    except s3_client.exceptions.NoSuchKey:
        logging.error(f"❌ FAISS 매니페스트 파일 S3://{S3_BUCKET_NAME}/{S3_MANIFEST_KEY}를 찾을 수 없습니다. 파티션이 올바르게 생성되었는지 확인하세요.")
        return []
    except Exception as e:
        logging.error(f"❌ FAISS 매니페스트 파일 로드 또는 파싱 중 오류 발생: {e}")
        return []

    if not active_partitions:
        logging.warning("⚠️ 활성화된 FAISS 파티션이 매니페스트 파일에 없습니다. 검색을 수행할 수 없습니다.")
        return []

    for p_prefix in active_partitions:
        # S3 객체 접두사에서 로컬 디렉토리 이름으로 사용하기 위해 슬래시를 언더스코어로 변경
        # 예를 들어, "feature_faiss_db_openai_partition/partition_0/" -> "partition_0_"
        # 또는 "feature_faiss_db_openai_partition/merged_july_2025/" -> "merged_july_2025_"
        # 단일 폴더 이름만 추출하기 위해 strip('/') 후 basename 사용
        safe_dir_name = os.path.basename(p_prefix.strip('/')) 
        
        # 로컬 임시 디렉토리 이름에 PID를 추가하여 동시성 문제 방지 (추가로 해시값도 추가)
        local_partition_dir = f"temp_faiss_title_{safe_dir_name}_{os.getpid()}_{hash(p_prefix) % 10000}" 

        s3_download_success = await download_faiss_index_from_s3_for_title_partition(p_prefix, local_partition_dir)
        
        if s3_download_success and os.path.exists(os.path.join(local_partition_dir, "index.faiss")):
            try:
                logging.info(f"    └─ 파티션 '{p_prefix}' 로드 중...")
                # allow_dangerous_deserialization=True는 필수입니다.
                partition_index = await asyncio.to_thread(
                    FAISS.load_local,
                    local_partition_dir,
                    embed_model,
                    allow_dangerous_deserialization=True
                )
                
                docs_and_scores = await asyncio.to_thread(
                    partition_index.similarity_search_by_vector_with_relevance_scores,
                    query_vector, k=k
                )
                
                for doc, score in docs_and_scores:
                    all_results.append({"doc": doc, "score": score, "partition_prefix": p_prefix})
                
            except Exception as e:
                logging.error(f"❌ 파티션 '{p_prefix}' 인덱스 로드 또는 쿼리 중 오류 발생: {e}")
            finally:
                # 임시 다운로드 디렉토리 삭제 (항상 시도)
                try:
                    # shutil.rmtree는 동기 함수이므로 asyncio.to_thread 사용
                    await asyncio.to_thread(lambda: os.path.exists(local_partition_dir) and shutil.rmtree(local_partition_dir))
                except OSError as e:
                    logging.warning(f"⚠️ 임시 디렉토리 '{local_partition_dir}' 삭제 중 오류 발생: {e}")
        else:
            logging.warning(f"⚠️ 파티션 '{p_prefix}' 인덱스 파일을 로드할 수 없어 검색에서 제외합니다.")

    # 모든 파티션의 결과를 합쳐서 최종 상위 k개만 반환
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
    
    return [(res["doc"], res["score"]) for res in sorted_results[:k]]


# 나머지 run_fact_check 함수는 이전과 동일합니다.
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
    
    transcript = fetch_youtube_transcript(youtube_url)
    if not transcript:
        return {"error": "Failed to load transcript"}

    claim_extractor = build_claim_extractor()
    result = await claim_extractor.ainvoke({"transcript": transcript})
    parsed_claims = []
    claim_pattern = re.compile(r'^[-\*]\s*(?:\[\d+\]\s*)?(.+?)\s*→\s*(.+)')

    for line in result.strip().split('\n'): # result.content -> result (ChatCompletion은 직접 content를 반환하지 않을 수 있음)
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
            logging.warning("LLM deduplication output is not in expected JSON markdown format. Using original parsed claims.")
            dedup_claims = parsed_claims

    claims_to_check = dedup_claims[:MAX_CLAIMS_TO_FACT_CHECK]

    # ===== 1. 모든 주장을 한 번에 배치 임베딩하고 캐싱 ===
    logging.info(f"--- {len(claims_to_check)}개의 주장에 대한 임베딩을 배치 처리합니다. ---")
    claim_embeddings = {}
    
    if claims_to_check:
        try:
            embeddings_list = await asyncio.to_thread(embed_model.embed_documents, claims_to_check)
            
            for i, text in enumerate(claims_to_check):
                claim_embeddings[text] = embeddings_list[i]
            logging.info("주장 임베딩 배치 처리 완료.")
        except Exception as e:
            logging.error(f"주장 임베딩 배치 처리 중 오류 발생: {e}")
            claim_embeddings = {} 

    outputs = []
    factcheck_chain = build_factcheck_chain()
    summarizer = build_claim_summarizer() 
    
    local_article_faiss_dbs = {} 

    async def process_claim_step(claim_data):
        """단일 클레임에 대한 팩트체크 로직을 캡슐화한 비동기 함수"""
        claim_idx, claim = claim_data
        
        summary_result = await summarizer.ainvoke({"claim": claim, "claim_context": ""}) # claim_context 추가
        summary = summary_result.content.strip()
        logging.info(f"\n--- [{claim_idx+1}] 주장: {claim} | 검색어: {summary}")

        # 1. 뉴스 검색 (비동기)
        results = await search_news_google_cs(summary)
        
        # ===== 2. 검색된 뉴스 제목들을 배치 임베딩하고 캐싱 ===
        logging.info(f"--- {len(results)}개의 뉴스 제목에 대한 임베딩을 배치 처리합니다. ---")
        news_title_embeddings = {}
        titles_to_embed_now = []

        for item in results:
            original_title = item.get("title", "")
            if not original_title:
                continue
            cleaned_title = clean_news_title(original_title)
            if cleaned_title and cleaned_title not in titles_to_embed_now:
                titles_to_embed_now.append(cleaned_title)

        if titles_to_embed_now:
            try:
                embeddings_list = await asyncio.to_thread(embed_model.embed_documents, titles_to_embed_now)
                
                for i, text in enumerate(titles_to_embed_now):
                    news_title_embeddings[text] = embeddings_list[i]
                logging.info("뉴스 제목 임베딩 배치 처리 완료.")
            except Exception as e:
                logging.error(f"뉴스 제목 임베딩 배치 처리 중 오류 발생: {e}")
                news_title_embeddings = {}

        # 2. 제목 유사도 검색 및 매칭 URL 수집
        matched_urls_meta = []
        for item in results:
            original_title = item.get("title", "")
            url = urlunparse(urlparse(item.get("link", ""))._replace(query='', fragment=''))
            if not original_title or not url:
                continue
            
            cleaned_title = clean_news_title(original_title)
            if not cleaned_title:
                continue

            query_vector = news_title_embeddings.get(cleaned_title)
            if query_vector is None:
                logging.warning(f"뉴스 제목 '{cleaned_title}'에 대한 임베딩이 캐시에 없습니다. 개별 임베딩을 시도합니다.")
                try:
                    query_vector = await asyncio.to_thread(embed_model.embed_query, cleaned_title)
                    news_title_embeddings[cleaned_title] = query_vector
                except Exception as e:
                    logging.error(f"뉴스 제목 '{cleaned_title}' 개별 임베딩 실패: {e}. 이 제목 건너뜀.")
                    continue
            
            # --- 수정된 부분: 분할된 FAISS 인덱스를 쿼리하는 함수 호출 ---
            docs_with_scores = await query_partitioned_faiss_titles(query_vector, k=3)
            # --- 수정 끝 ---

            for doc, score in docs_with_scores:
                if score <= 0.8 and doc.metadata.get("url"): 
                    matched_urls_meta.append((doc.metadata["url"], original_title))
                    break
        # ===================================================
        
        # 3. 각 matched_url에 대해 FAISS DB 조회/생성 작업 병렬화
        unique_urls_to_fetch = list(set(url for url, _ in matched_urls_meta))
        
        fetch_db_tasks = []
        for url in unique_urls_to_fetch:
            if url in local_article_faiss_dbs:
                continue

            async def _fetch_and_build_for_url(u):
                try:
                    chunk_db, article_text_from_cache = await get_or_build_faiss(u, article_text=None, embed_model=embed_model)
                    
                    if chunk_db and article_text_from_cache:
                        logging.info(f"FAISS DB 캐시 로드 및 본문 텍스트 복원 완료 (URL: {u})")
                        return chunk_db
                    else:
                        logging.info(f"캐시에 없어 기사 텍스트를 새로 가져옵니다 (URL: {u})")
                        article_text = await asyncio.to_thread(get_article_text, u)
                        if not article_text or len(article_text) < 300:
                            logging.warning(f"기사 텍스트가 짧거나 없습니다 (URL: {u}). FAISS DB 생성 건너뜀.")
                            return None
                        
                        chunk_db, _ = await get_or_build_faiss(u, article_text=article_text, embed_model=embed_model)
                        return chunk_db
                except Exception as e:
                    logging.warning(f"기사 처리 중 오류 발생 (URL: {u}): {e}")
                    return None
            
            fetch_db_tasks.append(_fetch_and_build_for_url(url))
        
        fetched_dbs_results = await asyncio.gather(*fetch_db_tasks, return_exceptions=True)
        
        for i, url in enumerate(unique_urls_to_fetch):
            if not isinstance(fetched_dbs_results[i], Exception) and fetched_dbs_results[i] is not None:
                local_article_faiss_dbs[url] = fetched_dbs_results[i]


        # 4. 청크 유사도 검색 및 LLM 팩트체크 작업 병렬화
        validated_tasks = []
        seen_urls_for_claim = set()

        claim_vec = claim_embeddings.get(claim)
        if claim_vec is None:
            logging.warning(f"'{claim}'에 대한 임베딩이 캐시에 없습니다. 개별 임베딩을 시도합니다.")
            claim_vec = await asyncio.to_thread(embed_model.embed_query, claim)
            claim_embeddings[claim] = claim_vec

        for url, title in matched_urls_meta:
            if url in local_article_faiss_dbs:
                chunk_db = local_article_faiss_dbs[url]
                chunks = chunk_db.similarity_search_by_vector(claim_vec, k=5)
                for chunk in chunks:
                    validated_tasks.append({
                        "task": factcheck_chain.ainvoke({"claim": claim, "context": chunk.page_content}),
                        "url": url,
                        "snippet": chunk.page_content,
                        "source_title": title,
                    })
        
        llm_responses = await asyncio.gather(*[t["task"] for t in validated_tasks], return_exceptions=True)
        validated_evidence = []

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
        "video_url": youtube_url,
        "video_total_confidence_score": avg_score,
        "confidence_summary": summary,
        "channel_type": channel_type,
        "channel_type_reason": reason,
        "claims": outputs
    }