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

# --- ✨✨✨ S3 설정 추가 ✨✨✨ ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "factseeker-faiss-db")
try:
    s3 = boto3.client('s3')
except Exception as e:
    s3 = None
    logging.error(f"S3 클라이언트 초기화 실패: {e}")
# --- ✨✨✨ S3 설정 종료 ✨✨✨ ---


MAX_CLAIMS_TO_FACT_CHECK = 10
DISTANCE_THRESHOLD = 0.8  # L2 거리 임계값

embed_model = OpenAIEmbeddings(model="text-embedding-3-small", request_timeout=60, max_retries=5, chunk_size=500)

# --- ✨✨✨ S3 업로드/다운로드 함수 추가 ✨✨✨ ---
def upload_to_s3(local_dir, s3_key):
    if not s3:
        logging.warning("S3 클라이언트가 없어 업로드를 건너뜁니다.")
        return
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # s3_key는 디렉토리처럼 작동하므로 파일 이름을 추가합니다.
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
        # S3 키(폴더) 아래의 모든 객체를 나열합니다.
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=s3_key)
        if 'Contents' not in response:
            return False # S3에 파일이 없음

        os.makedirs(local_dir, exist_ok=True)
        for obj in response['Contents']:
            s3_path = obj['Key']
            # s3_key prefix를 제거하여 로컬 파일 경로를 만듭니다.
            relative_path = os.path.relpath(s3_path, s3_key)
            local_path = os.path.join(local_dir, relative_path)
            
            # 파일이 위치할 로컬 디렉토리가 없으면 생성합니다.
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            s3.download_file(S3_BUCKET_NAME, s3_path, local_path)
        return True
    except ClientError as e:
        # 404 (Not Found)는 파일이 없는 것이므로 오류가 아닙니다.
        if e.response['Error']['Code'] == '404':
            return False
        logging.error(f"S3 다운로드 중 오류 발생: s3://{S3_BUCKET_NAME}/{s3_key} - {e}")
        return False
# --- ✨✨✨ S3 함수 종료 ✨✨✨ ---


async def get_article_text_safe(url):
    """기사 본문 병렬 크롤링 시 예외 안전 래퍼"""
    try:
        text = await get_article_text(url)
        return url, text
    except Exception as e:
        logging.warning(f"❌ 기사 본문 추출 실패: {url} - {e}")
        return url, None

async def search_and_retrieve_docs(claim, faiss_partition_dirs):
    summarizer = build_claim_summarizer()
    try:
        summary_result = await summarizer.ainvoke({"claim": claim})
        summarized_query = summary_result.content.strip()
        logging.info(f"🔍 요약된 검색어: '{summarized_query}'")
    except Exception as e:
        logging.error(f"Claim 요약 실패: {e}, 원문으로 검색 진행")
        summarized_query = claim

    search_results = await search_news_google_cs(summarized_query)

    cse_titles = [clean_news_title(item.get('title', '')) for item in search_results[:10]]
    cse_raw_titles = [item.get('title', '') for item in search_results[:10]]
    cse_urls = [item.get('link') for item in search_results[:10]]
    cse_title_embs = embed_model.embed_documents(cse_titles)

    matched_urls = {}
    for idx, faiss_dir in enumerate(faiss_partition_dirs):
        faiss_index_path = os.path.join(faiss_dir, "index.faiss")
        faiss_pkl_path = os.path.join(faiss_dir, "index.pkl")
        if not (os.path.exists(faiss_index_path) and os.path.exists(faiss_pkl_path)):
            continue
        try:
            title_faiss_db = FAISS.load_local(
                faiss_dir,
                embeddings=embed_model,
                allow_dangerous_deserialization=True
            )
            D, I = title_faiss_db.index.search(np.array(cse_title_embs, dtype=np.float32), k=1)
            for j, dist in enumerate(D.flatten()):
                if dist < DISTANCE_THRESHOLD:
                    faiss_idx = I.flatten()[j]
                    docstore_id = title_faiss_db.index_to_docstore_id[faiss_idx]
                    doc = title_faiss_db.docstore._dict[docstore_id]
                    url = doc.metadata.get("url")
                    if url and url not in matched_urls:
                        matched_urls[url] = {
                            "matched_cse_title": cse_titles[j],
                            "raw_cse_title": cse_raw_titles[j]
                        }
        except Exception as e:
            logging.error(f"FAISS 파티션 {faiss_dir} 검색 실패: {e}")

    article_urls = list(matched_urls.keys())
    coros = [get_article_text_safe(url) for url in article_urls]
    article_results = await asyncio.gather(*coros)
    docs = []
    for url, article_text in article_results:
        meta = matched_urls[url]
        if article_text and len(article_text) > 200:
            docs.append(Document(
                page_content=article_text,
                metadata={
                    "url": url,
                    "matched_cse_title": meta["matched_cse_title"],
                    "raw_cse_title": meta["raw_cse_title"]
                }
            ))
    logging.info(f"📰 최종 크롤링 성공 문서 수: {len(docs)}")
    return docs

async def run_fact_check(youtube_url, faiss_partition_dirs):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    logging.info(f"유튜브 분석 시작: {youtube_url}")
    try:
        transcript = fetch_youtube_transcript(youtube_url)
        if not transcript:
            return {"error": "Failed to load transcript"}

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
                claims_to_check = [
                    line.strip() for line in reduced_result.content.strip().split('\n')
                    if line.strip()
                ]
        except json.JSONDecodeError:
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

    async def process_claim_step(idx, claim):
        logging.info(f"--- 팩트체크 시작: ({idx + 1}/{len(claims_to_check)}) '{claim}'")

        # --- ✨✨✨ FAISS 생성/로드 로직 수정 ✨✨✨ ---
        claim_hash = hashlib.md5(claim.encode()).hexdigest()
        s3_key = f"claim_faiss_cache/{claim_hash}"
        local_faiss_path = os.path.join(CHUNK_CACHE_DIR, s3_key)
        
        faiss_db = None
        docs = None

        # 1. 로컬 캐시 확인
        if os.path.exists(local_faiss_path):
            try:
                faiss_db = FAISS.load_local(local_faiss_path, embed_model, allow_dangerous_deserialization=True)
                docs = [doc for doc in faiss_db.docstore._dict.values()]
                logging.info(f"✅ 로컬 캐시에서 FAISS DB 로드 성공: {local_faiss_path}")
            except Exception as e:
                logging.warning(f"⚠️ 로컬 캐시 로드 실패, 재생성 시도: {e}")
                shutil.rmtree(local_faiss_path)

        # 2. S3 캐시 확인
        if not faiss_db:
            logging.info(f"S3 캐시 확인 중: s3://{S3_BUCKET_NAME}/{s3_key}")
            if download_from_s3(local_faiss_path, s3_key):
                try:
                    faiss_db = FAISS.load_local(local_faiss_path, embed_model, allow_dangerous_deserialization=True)
                    docs = [doc for doc in faiss_db.docstore._dict.values()]
                    logging.info(f"✅ S3 캐시에서 FAISS DB 다운로드 및 로드 성공")
                except Exception as e:
                    logging.warning(f"⚠️ S3 캐시 로드 실패, 재생성 시도: {e}")
                    shutil.rmtree(local_faiss_path)
            else:
                logging.info(f"S3에 캐시 없음. 새로 생성합니다.")

        # 3. 새로 생성 및 S3에 업로드
        if not faiss_db:
            docs = await search_and_retrieve_docs(claim, faiss_partition_dirs)
            if not docs:
                logging.info(f"근거 문서를 찾지 못함: '{claim}'")
                return {
                    "claim": claim, "result": "insufficient_evidence",
                    "confidence_score": 0, "evidence": []
                }
            
            os.makedirs(local_faiss_path, exist_ok=True)
            faiss_db = FAISS.from_documents(docs, embed_model)
            faiss_db.save_local(local_faiss_path)
            logging.info(f"✅ FAISS DB 새로 생성 및 로컬 저장 완료: {local_faiss_path}")
            
            try:
                logging.info(f"🚀 S3에 FAISS 인덱스 업로드 시도: s3://{S3_BUCKET_NAME}/{s3_key}")
                upload_to_s3(local_faiss_path, s3_key)
                logging.info(f"✅ S3 업로드 성공.")
            except Exception as e:
                logging.error(f"❌ S3 업로드 실패: {e}")

        if not docs: # 캐시에서 로드한 경우 docs가 None이므로 새로 채워야 함
            docs = [doc for doc_id, doc in faiss_db.docstore._dict.items()]
        # --- ✨✨✨ 로직 수정 종료 ✨✨✨ ---

        validated_evidence = []
        fact_checker = build_factcheck_chain()

        async def factcheck_doc(doc):
            try:
                check_result = await fact_checker.ainvoke({
                    "claim": claim, "context": doc.page_content
                })
                result_content = check_result.content
                relevance = re.search(r"관련성: (.+)", result_content)
                fact_check_result_match = re.search(r"사실 설명 여부: (.+)", result_content)
                justification = re.search(r"간단한 설명: (.+)", result_content)
                snippet = re.search(r"핵심 근거 문장: (.+)", result_content)

                if relevance and fact_check_result_match and justification:
                    fact_check_result_text = fact_check_result_match.group(1).strip()
                    if "예" in relevance.group(1) and "아니오" not in fact_check_result_text:
                        return {
                            "url": doc.metadata.get("url"),
                            "relevance": "yes",
                            "fact_check_result": fact_check_result_text,
                            "justification": justification.group(1).strip(),
                            "snippet": snippet.group(1).strip() if snippet else ""
                        }
            except Exception as e:
                logging.error(f"    - LLM 팩트체크 체인 실행 중 오류: {e}")
            return None

        factcheck_tasks = [factcheck_doc(doc) for doc in docs]
        factcheck_results = await asyncio.gather(*factcheck_tasks)
        validated_evidence = [res for res in factcheck_results if res]

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
            "evidence": validated_evidence[:3]
        }

    claim_tasks = [
        process_claim_step(idx, claim)
        for idx, claim in enumerate(claims_to_check)
    ]
    outputs = await asyncio.gather(*claim_tasks, return_exceptions=True)
    outputs = [output for output in outputs if not isinstance(output, Exception)]

    avg_score = round(sum(o['confidence_score'] for o in outputs) / len(outputs)) if outputs else 0
    evidence_ratio = sum(1 for o in outputs if o["result"] == "likely_true") / len(outputs) if outputs else 0
    summary = f"증거 확보된 주장 비율: {evidence_ratio*100:.1f}%" if len(outputs) >= 3 else f"신뢰도 평가 불가 (팩트체크 주장 수 부족: {len(outputs)}개)"
    
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
        "channel_type_reason": reason
    }


def parse_channel_type(llm_output: str):
    channel_type_match = re.search(r"채널 유형:\s*(.+)", llm_output)
    reason_match = re.search(r"분류 근거:\s*(.+)", llm_output)
    channel_type = channel_type_match.group(1).strip() if channel_type_match else "알 수 없음"
    reason = reason_match.group(1).strip() if reason_match else "LLM 응답에서 판단 근거를 찾을 수 없습니다."
    return channel_type, reason