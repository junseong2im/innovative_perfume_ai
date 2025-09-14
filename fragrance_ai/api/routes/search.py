from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
import time

from ..schemas import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SearchResult,
    FragranceNote
)
from ...services.search_service import SearchService
from ..dependencies import get_search_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(
    request: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service)
):
    """의미 기반 검색 엔드포인트"""
    try:
        start_time = time.time()
        
        # 실제 검색 서비스 호출
        search_request = {
            "query": request.query,
            "collection_name": getattr(request, 'collection_name', 'fragrance_notes'),
            "top_k": getattr(request, 'top_k', 10),
            "min_similarity": getattr(request, 'min_similarity', 0.5),
            "filters": getattr(request, 'filters', None)
        }
        
        # 검색 서비스를 통해 의미 기반 검색 수행
        search_results = await search_service.semantic_search(**search_request)
        
        # SearchResult 객체들로 변환
        results = []
        for result in search_results.get("results", []):
            search_result = SearchResult(
                id=result.get("id", ""),
                document=result.get("document", result.get("content", "")),
                metadata=result.get("metadata", {}),
                distance=result.get("distance", 1.0),
                similarity=1.0 - result.get("distance", 1.0)
            )
            results.append(search_result)
        
        return SemanticSearchResponse(
            results=results,
            total_results=search_results.get("total_results", len(results)),
            query=request.query,
            search_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{item_id}")
async def get_similar_items(
    item_id: str,
    collection_name: str = Query(default="fragrance_notes"),
    top_k: int = Query(default=5, ge=1, le=20),
    search_service: SearchService = Depends(get_search_service)
):
    """특정 아이템과 유사한 아이템 검색"""
    try:
        # 검색 서비스를 통해 유사 아이템 검색
        similar_results = await search_service.find_similar_items(
            item_id=item_id,
            collection_name=collection_name,
            top_k=top_k
        )
        
        return {
            "item_id": item_id,
            "similar_items": similar_results.get("similar_items", []),
            "collection": collection_name,
            "total_found": len(similar_results.get("similar_items", []))
        }
        
    except Exception as e:
        logger.error(f"Similar items search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(
    collection_name: str,
    search_service: SearchService = Depends(get_search_service)
):
    """컬렉션 통계 정보"""
    try:
        # 검색 서비스를 통해 컬렉션 통계 조회
        stats = await search_service.get_collection_stats(collection_name)
        
        return {
            "collection_name": collection_name,
            "document_count": stats.get("document_count", 0),
            "last_updated": stats.get("last_updated"),
            "embedding_dimension": stats.get("embedding_dimension"),
            "index_size": stats.get("index_size")
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections/{collection_name}/search")
async def search_collection(
    collection_name: str,
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=50),
    min_similarity: float = Query(default=0.5, ge=0.0, le=1.0),
    search_service: SearchService = Depends(get_search_service)
):
    """특정 컬렉션에서 검색"""
    try:
        # 검색 서비스를 통해 특정 컬렉션 검색
        search_results = await search_service.search_collection(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        return {
            "collection": collection_name,
            "query": query,
            "results": search_results.get("results", []),
            "total_results": search_results.get("total_results", 0),
            "search_time": search_results.get("search_time", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Collection search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))