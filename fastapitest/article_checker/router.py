from typing import Callable, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, AnyUrl

from fastapitest.article_checker.article_fact_checker import run_article_fact_check


class ArticleFactCheckRequest(BaseModel):
    article_url: AnyUrl


def create_router(get_faiss_partition_dirs: Callable[[], List[str]]) -> APIRouter:
    router = APIRouter()

    @router.post("/article-fact-check")
    async def article_fact_check_endpoint(req: ArticleFactCheckRequest):
        faiss_partition_dirs = get_faiss_partition_dirs() or []
        if not faiss_partition_dirs:
            raise HTTPException(
                status_code=503,
                detail="서버가 아직 준비되지 않았습니다. FAISS 파티션이 로드되지 않았습니다.",
            )

        try:
            result = await run_article_fact_check(str(req.article_url), faiss_partition_dirs)
            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")

    return router

