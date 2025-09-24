"""
향수 검색 도구
- 하이브리드 검색 (벡터 + 메타데이터)
- 시맨틱 검색 및 필터링 기능
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from datetime import datetime

# 향수 데이터베이스 및 벡터 스토어 임포트
from ..services.search_service import SearchService
from ..core.vector_store import VectorStore
from ..database.models import Recipe, FragranceNote

logger = logging.getLogger(__name__)

# Pydantic 스키마
class SearchQuery(BaseModel):
    """검색 쿼리 스키마"""
    text_query: str = Field(..., description="검색 텍스트 쿼리")
    search_type: str = Field(default="similarity", description="검색 타입: similarity, hybrid, filter")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="메타데이터 필터")
    top_k: int = Field(default=10, description="반환할 결과 수")
    similarity_threshold: float = Field(default=0.7, description="유사도 임계값")

class SearchResultItem(BaseModel):
    """검색 결과 항목"""
    perfume_id: str
    name: str
    brand: Optional[str] = None
    description: str
    similarity_score: float
    fragrance_family: str
    top_notes: List[str]
    heart_notes: List[str]
    base_notes: List[str]
    metadata: Dict[str, Any]

class SearchResult(BaseModel):
    """검색 결과 전체"""
    query: str
    total_results: int
    results: List[SearchResultItem]
    search_time_ms: float
    filters_applied: Dict[str, Any]

# 검색 서비스 인스턴스 (싱글톤)
search_service = None

def get_search_service():
    """검색 서비스 인스턴스 가져오기"""
    global search_service
    if search_service is None:
        search_service = SearchService()
    return search_service

async def hybrid_search(
    text_query: str,
    metadata_filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    similarity_threshold: float = 0.7
) -> SearchResult:
    """
    # LLM TOOL DESCRIPTION (FOR ORCHESTRATOR)
    # Use this tool to search for existing perfumes or fragrance notes.
    # It performs hybrid search combining semantic similarity and metadata filtering.
    # Returns detailed perfume information including notes, family, and descriptions.

    Args:
        text_query: 검색할 텍스트 쿼리 (예: "fresh citrus perfume for summer")
        metadata_filters: 필터링 조건 (예: {"season": "summer", "gender": "unisex"})
        top_k: 반환할 최대 결과 수
        similarity_threshold: 최소 유사도 점수

    Returns:
        SearchResult: 검색 결과와 메타데이터
    """
    start_time = datetime.now()

    try:
        service = get_search_service()

        # 하이브리드 검색 수행
        raw_results = await service.hybrid_search(
            query=text_query,
            collection_name="perfumes",
            filters=metadata_filters,
            top_k=top_k,
            threshold=similarity_threshold
        )

        # 결과 변환
        search_items = []
        for result in raw_results.get("results", []):
            item = SearchResultItem(
                perfume_id=result.get("id"),
                name=result.get("name"),
                brand=result.get("brand"),
                description=result.get("description", ""),
                similarity_score=result.get("score", 0.0),
                fragrance_family=result.get("fragrance_family", "unknown"),
                top_notes=result.get("top_notes", []),
                heart_notes=result.get("heart_notes", []),
                base_notes=result.get("base_notes", []),
                metadata=result.get("metadata", {})
            )
            search_items.append(item)

        # 검색 시간 계산
        search_time = (datetime.now() - start_time).total_seconds() * 1000

        return SearchResult(
            query=text_query,
            total_results=len(search_items),
            results=search_items,
            search_time_ms=search_time,
            filters_applied=metadata_filters or {}
        )

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        # 실패 시 빈 결과 반환
        return SearchResult(
            query=text_query,
            total_results=0,
            results=[],
            search_time_ms=0,
            filters_applied=metadata_filters or {}
        )

async def search_fragrance_notes(
    note_query: str,
    note_type: Optional[str] = None,
    fragrance_family: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    향료 노트 검색

    Args:
        note_query: 노트 검색 쿼리
        note_type: 노트 타입 (top, middle, base)
        fragrance_family: 향수 패밀리
        top_k: 반환할 노트 수

    Returns:
        검색된 노트 리스트
    """
    try:
        service = get_search_service()

        filters = {}
        if note_type:
            filters["note_type"] = note_type
        if fragrance_family:
            filters["fragrance_family"] = fragrance_family

        results = await service.search_notes(
            query=note_query,
            filters=filters,
            top_k=top_k
        )

        return results

    except Exception as e:
        logger.error(f"Note search failed: {e}")
        return []

async def find_similar_perfumes(
    perfume_id: str,
    top_k: int = 5
) -> List[SearchResultItem]:
    """
    유사한 향수 찾기

    Args:
        perfume_id: 기준 향수 ID
        top_k: 반환할 유사 향수 수

    Returns:
        유사한 향수 리스트
    """
    try:
        service = get_search_service()

        # 기준 향수의 임베딩 가져오기
        base_perfume = await service.get_perfume_by_id(perfume_id)
        if not base_perfume:
            return []

        # 유사 검색 수행
        similar = await service.find_similar(
            reference_id=perfume_id,
            collection_name="perfumes",
            top_k=top_k
        )

        # 결과 변환
        results = []
        for item in similar:
            results.append(SearchResultItem(
                perfume_id=item.get("id"),
                name=item.get("name"),
                brand=item.get("brand"),
                description=item.get("description", ""),
                similarity_score=item.get("score", 0.0),
                fragrance_family=item.get("fragrance_family", "unknown"),
                top_notes=item.get("top_notes", []),
                heart_notes=item.get("heart_notes", []),
                base_notes=item.get("base_notes", []),
                metadata=item.get("metadata", {})
            ))

        return results

    except Exception as e:
        logger.error(f"Similar perfume search failed: {e}")
        return []