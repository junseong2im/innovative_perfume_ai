"""
향상된 검색 서비스

Repository 패턴을 사용하여 데이터 계층과 분리된 검색 비즈니스 로직을 제공합니다.
비동기 처리와 캐싱을 통해 성능을 최적화합니다.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ..repositories import FragranceNoteRepository, RecipeRepository
from ..database.connection import get_db_session
from ..database.models import FragranceNote, Recipe
from ..models.embedding import AdvancedKoreanFragranceEmbedding
from ..core.production_logging import get_logger
from ..core.intelligent_cache import FragranceCacheManager
from ..core.exceptions import FragranceAIException, ErrorCode

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    id: str
    name: str
    name_korean: Optional[str]
    description: Optional[str]
    score: float
    similarity: float
    metadata: Dict[str, Any]
    result_type: str  # 'note' or 'recipe'


@dataclass
class SearchRequest:
    """검색 요청 데이터 클래스"""
    query: str
    search_type: str = "hybrid"  # semantic, keyword, hybrid
    result_types: List[str] = None  # ['note', 'recipe']
    filters: Optional[Dict[str, Any]] = None
    limit: int = 20
    offset: int = 0
    include_metadata: bool = True
    use_cache: bool = True


@dataclass
class SearchResponse:
    """검색 응답 데이터 클래스"""
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    cached: bool
    query_embedding: Optional[List[float]] = None
    suggestions: List[str] = None


class EnhancedSearchService:
    """향상된 검색 서비스"""

    def __init__(self,
                 embedding_model: Optional[AdvancedKoreanFragranceEmbedding] = None,
                 cache_manager: Optional[FragranceCacheManager] = None):
        self.embedding_model = embedding_model
        self.cache_manager = cache_manager

        # 기본 가중치
        self.search_weights = {
            'semantic_similarity': 0.6,
            'keyword_match': 0.3,
            'metadata_match': 0.1
        }

    # ==========================================
    # 메인 검색 인터페이스
    # ==========================================

    async def search(self, request: SearchRequest) -> SearchResponse:
        """통합 검색 수행"""
        start_time = datetime.now()

        try:
            # 캐시 확인
            cached_result = None
            if request.use_cache and self.cache_manager:
                cache_key = self._generate_cache_key(request)
                cached_result = await self.cache_manager.get_cached_search_result(cache_key)

                if cached_result:
                    logger.debug(f"Cache hit for search: {request.query}")
                    return SearchResponse(
                        results=cached_result['results'],
                        total_count=cached_result['total_count'],
                        search_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        cached=True
                    )

            # 검색 실행
            results = await self._perform_search(request)

            # 결과 정렬 및 제한
            sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
            limited_results = sorted_results[request.offset:request.offset + request.limit]

            # 응답 생성
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            response = SearchResponse(
                results=limited_results,
                total_count=len(results),
                search_time_ms=search_time,
                cached=False,
                suggestions=await self._generate_suggestions(request.query)
            )

            # 캐시 저장
            if request.use_cache and self.cache_manager:
                await self.cache_manager.cache_search_result(
                    cache_key,
                    {
                        'results': limited_results,
                        'total_count': len(results)
                    },
                    ttl=3600  # 1시간
                )

            logger.info(f"Search completed: {len(limited_results)} results in {search_time:.2f}ms")
            return response

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise FragranceAIException(
                message=f"Search operation failed: {str(e)}",
                error_code=ErrorCode.SEARCH_ERROR,
                cause=e
            )

    async def semantic_search(self,
                            query: str,
                            result_types: List[str] = None,
                            limit: int = 20,
                            similarity_threshold: float = 0.7) -> List[SearchResult]:
        """시맨틱 검색"""
        if not self.embedding_model:
            raise FragranceAIException(
                message="Embedding model not available",
                error_code=ErrorCode.MODEL_ERROR
            )

        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_model.encode_async([query])
            embedding_vector = query_embedding.embeddings[0]

            # 결과 타입별 검색
            all_results = []

            if not result_types:
                result_types = ['note', 'recipe']

            if 'note' in result_types:
                note_results = await self._semantic_search_notes(
                    embedding_vector, limit, similarity_threshold
                )
                all_results.extend(note_results)

            if 'recipe' in result_types:
                recipe_results = await self._semantic_search_recipes(
                    embedding_vector, limit, similarity_threshold
                )
                all_results.extend(recipe_results)

            # 유사도 기준 정렬
            all_results.sort(key=lambda x: x.similarity, reverse=True)
            return all_results[:limit]

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise

    async def keyword_search(self,
                           query: str,
                           result_types: List[str] = None,
                           filters: Optional[Dict[str, Any]] = None,
                           limit: int = 20) -> List[SearchResult]:
        """키워드 검색"""
        try:
            all_results = []

            if not result_types:
                result_types = ['note', 'recipe']

            # 데이터베이스 세션 사용
            with get_db_session() as session:
                if 'note' in result_types:
                    note_repo = FragranceNoteRepository(session)
                    note_results = await self._keyword_search_notes(
                        note_repo, query, filters, limit
                    )
                    all_results.extend(note_results)

                if 'recipe' in result_types:
                    recipe_repo = RecipeRepository(session)
                    recipe_results = await self._keyword_search_recipes(
                        recipe_repo, query, filters, limit
                    )
                    all_results.extend(recipe_results)

            return all_results[:limit]

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            raise

    async def hybrid_search(self,
                          query: str,
                          result_types: List[str] = None,
                          filters: Optional[Dict[str, Any]] = None,
                          limit: int = 20) -> List[SearchResult]:
        """하이브리드 검색 (시맨틱 + 키워드)"""
        try:
            # 병렬로 시맨틱 검색과 키워드 검색 실행
            semantic_task = asyncio.create_task(
                self.semantic_search(query, result_types, limit * 2)
            )
            keyword_task = asyncio.create_task(
                self.keyword_search(query, result_types, filters, limit * 2)
            )

            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task, return_exceptions=True
            )

            # 예외 처리
            if isinstance(semantic_results, Exception):
                logger.warning(f"Semantic search failed: {semantic_results}")
                semantic_results = []

            if isinstance(keyword_results, Exception):
                logger.warning(f"Keyword search failed: {keyword_results}")
                keyword_results = []

            # 결과 병합 및 점수 계산
            merged_results = self._merge_search_results(
                semantic_results, keyword_results, query
            )

            return merged_results[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise

    # ==========================================
    # 전문 검색 메서드
    # ==========================================

    async def search_notes_by_characteristics(self,
                                            intensity_range: Optional[Tuple[float, float]] = None,
                                            longevity_range: Optional[Tuple[float, float]] = None,
                                            sillage_range: Optional[Tuple[float, float]] = None,
                                            fragrance_families: Optional[List[str]] = None,
                                            note_types: Optional[List[str]] = None,
                                            limit: int = 50) -> List[SearchResult]:
        """특성 기반 노트 검색"""
        try:
            with get_db_session() as session:
                note_repo = FragranceNoteRepository(session)

                # 특성 필터 적용
                notes = note_repo.find_by_characteristics(
                    intensity_min=intensity_range[0] if intensity_range else None,
                    intensity_max=intensity_range[1] if intensity_range else None,
                    longevity_min=longevity_range[0] if longevity_range else None,
                    longevity_max=longevity_range[1] if longevity_range else None,
                    sillage_min=sillage_range[0] if sillage_range else None,
                    sillage_max=sillage_range[1] if sillage_range else None,
                )

                # 추가 필터 적용
                if fragrance_families:
                    notes = [n for n in notes if n.fragrance_family in fragrance_families]

                if note_types:
                    notes = [n for n in notes if n.note_type in note_types]

                # SearchResult로 변환
                results = []
                for note in notes[:limit]:
                    score = self._calculate_characteristic_score(
                        note, intensity_range, longevity_range, sillage_range
                    )

                    results.append(SearchResult(
                        id=note.id,
                        name=note.name,
                        name_korean=note.name_korean,
                        description=note.description,
                        score=score,
                        similarity=score,
                        metadata=self._build_note_metadata(note),
                        result_type='note'
                    ))

                return results

        except Exception as e:
            logger.error(f"Characteristic-based note search failed: {str(e)}")
            raise

    async def search_recipes_by_composition(self,
                                          required_notes: List[str],
                                          optional_notes: Optional[List[str]] = None,
                                          excluded_notes: Optional[List[str]] = None,
                                          complexity_range: Optional[Tuple[int, int]] = None,
                                          fragrance_families: Optional[List[str]] = None,
                                          limit: int = 50) -> List[SearchResult]:
        """구성 기반 레시피 검색"""
        try:
            with get_db_session() as session:
                recipe_repo = RecipeRepository(session)

                # 기본 필터로 레시피 조회
                recipes = recipe_repo.get_all()

                # 필수 노트 포함 필터
                if required_notes:
                    filtered_recipes = []
                    for recipe in recipes:
                        recipe_note_ids = [ing.note_id for ing in recipe.ingredients]
                        if all(note_id in recipe_note_ids for note_id in required_notes):
                            filtered_recipes.append(recipe)
                    recipes = filtered_recipes

                # 제외 노트 필터
                if excluded_notes:
                    filtered_recipes = []
                    for recipe in recipes:
                        recipe_note_ids = [ing.note_id for ing in recipe.ingredients]
                        if not any(note_id in recipe_note_ids for note_id in excluded_notes):
                            filtered_recipes.append(recipe)
                    recipes = filtered_recipes

                # 복잡도 필터
                if complexity_range:
                    recipes = [r for r in recipes
                             if complexity_range[0] <= r.complexity <= complexity_range[1]]

                # 향족 필터
                if fragrance_families:
                    recipes = [r for r in recipes if r.fragrance_family in fragrance_families]

                # SearchResult로 변환
                results = []
                for recipe in recipes[:limit]:
                    score = self._calculate_composition_score(
                        recipe, required_notes, optional_notes, excluded_notes
                    )

                    results.append(SearchResult(
                        id=recipe.id,
                        name=recipe.name,
                        name_korean=recipe.name_korean,
                        description=recipe.description,
                        score=score,
                        similarity=score,
                        metadata=self._build_recipe_metadata(recipe),
                        result_type='recipe'
                    ))

                # 점수순 정렬
                results.sort(key=lambda x: x.score, reverse=True)
                return results

        except Exception as e:
            logger.error(f"Composition-based recipe search failed: {str(e)}")
            raise

    async def find_similar_items(self,
                               item_id: str,
                               item_type: str,
                               limit: int = 10) -> List[SearchResult]:
        """유사 아이템 찾기"""
        try:
            with get_db_session() as session:
                if item_type == 'note':
                    note_repo = FragranceNoteRepository(session)
                    similar_notes = note_repo.find_similar_notes(item_id, limit)

                    return [
                        SearchResult(
                            id=note.id,
                            name=note.name,
                            name_korean=note.name_korean,
                            description=note.description,
                            score=0.8,  # 기본 유사도 점수
                            similarity=0.8,
                            metadata=self._build_note_metadata(note),
                            result_type='note'
                        )
                        for note in similar_notes
                    ]

                elif item_type == 'recipe':
                    recipe_repo = RecipeRepository(session)
                    similar_recipes = recipe_repo.find_similar_recipes(item_id, limit)

                    return [
                        SearchResult(
                            id=recipe.id,
                            name=recipe.name,
                            name_korean=recipe.name_korean,
                            description=recipe.description,
                            score=0.8,
                            similarity=0.8,
                            metadata=self._build_recipe_metadata(recipe),
                            result_type='recipe'
                        )
                        for recipe in similar_recipes
                    ]

                else:
                    raise ValueError(f"Unsupported item type: {item_type}")

        except Exception as e:
            logger.error(f"Similar items search failed: {str(e)}")
            raise

    # ==========================================
    # 내부 헬퍼 메서드
    # ==========================================

    async def _perform_search(self, request: SearchRequest) -> List[SearchResult]:
        """검색 타입에 따른 검색 수행"""
        if request.search_type == "semantic":
            return await self.semantic_search(
                request.query,
                request.result_types,
                request.limit * 2  # 더 많은 결과를 가져와서 필터링
            )
        elif request.search_type == "keyword":
            return await self.keyword_search(
                request.query,
                request.result_types,
                request.filters,
                request.limit * 2
            )
        elif request.search_type == "hybrid":
            return await self.hybrid_search(
                request.query,
                request.result_types,
                request.filters,
                request.limit * 2
            )
        else:
            raise ValueError(f"Unsupported search type: {request.search_type}")

    async def _semantic_search_notes(self,
                                   embedding_vector: List[float],
                                   limit: int,
                                   similarity_threshold: float) -> List[SearchResult]:
        """노트 시맨틱 검색"""
        # 실제 구현에서는 벡터 데이터베이스(ChromaDB 등) 사용
        # 현재는 임시 구현
        with get_db_session() as session:
            note_repo = FragranceNoteRepository(session)
            notes = note_repo.get_all(limit=limit * 2)

            results = []
            for note in notes:
                # 벡터 유사도 계산
                note_embedding = await self._get_or_create_embedding(
                    f"{note.name} {note.description}"
                )

                similarity = self._calculate_cosine_similarity(
                    embedding_vector, note_embedding
                )

                if similarity >= similarity_threshold:
                    results.append(SearchResult(
                        id=note.id,
                        name=note.name,
                        name_korean=note.name_korean,
                        description=note.description,
                        score=similarity,
                        similarity=similarity,
                        metadata=self._build_note_metadata(note),
                        result_type='note'
                    ))

            return results[:limit]

    async def _semantic_search_recipes(self,
                                     embedding_vector: List[float],
                                     limit: int,
                                     similarity_threshold: float) -> List[SearchResult]:
        """레시피 시맨틱 검색"""
        # 실제 구현에서는 벡터 데이터베이스 사용
        with get_db_session() as session:
            recipe_repo = RecipeRepository(session)
            recipes = recipe_repo.get_all(limit=limit * 2)

            results = []
            for recipe in recipes:
                # 벡터 유사도 계산
                recipe_embedding = await self._get_or_create_embedding(
                    f"{recipe.name} {recipe.description}"
                )

                similarity = self._calculate_cosine_similarity(
                    embedding_vector, recipe_embedding
                )

                if similarity >= similarity_threshold:
                    results.append(SearchResult(
                        id=recipe.id,
                        name=recipe.name,
                        name_korean=recipe.name_korean,
                        description=recipe.description,
                        score=similarity,
                        similarity=similarity,
                        metadata=self._build_recipe_metadata(recipe),
                        result_type='recipe'
                    ))

            return results[:limit]

    async def _keyword_search_notes(self,
                                  note_repo: FragranceNoteRepository,
                                  query: str,
                                  filters: Optional[Dict[str, Any]],
                                  limit: int) -> List[SearchResult]:
        """노트 키워드 검색"""
        notes = note_repo.find_by_name(query, exact=False)

        results = []
        for note in notes[:limit]:
            score = self._calculate_keyword_score(query, note.name, note.description)

            results.append(SearchResult(
                id=note.id,
                name=note.name,
                name_korean=note.name_korean,
                description=note.description,
                score=score,
                similarity=score,
                metadata=self._build_note_metadata(note),
                result_type='note'
            ))

        return results

    async def _keyword_search_recipes(self,
                                    recipe_repo: RecipeRepository,
                                    query: str,
                                    filters: Optional[Dict[str, Any]],
                                    limit: int) -> List[SearchResult]:
        """레시피 키워드 검색"""
        recipes = recipe_repo.search_recipes_by_name(query, exact=False)

        results = []
        for recipe in recipes[:limit]:
            score = self._calculate_keyword_score(query, recipe.name, recipe.description)

            results.append(SearchResult(
                id=recipe.id,
                name=recipe.name,
                name_korean=recipe.name_korean,
                description=recipe.description,
                score=score,
                similarity=score,
                metadata=self._build_recipe_metadata(recipe),
                result_type='recipe'
            ))

        return results

    def _merge_search_results(self,
                            semantic_results: List[SearchResult],
                            keyword_results: List[SearchResult],
                            query: str) -> List[SearchResult]:
        """검색 결과 병합"""
        # ID 기반으로 결과 병합
        merged_dict = {}

        # 시맨틱 검색 결과
        for result in semantic_results:
            result.score = (
                result.similarity * self.search_weights['semantic_similarity']
            )
            merged_dict[result.id] = result

        # 키워드 검색 결과
        for result in keyword_results:
            if result.id in merged_dict:
                # 기존 결과와 병합
                existing = merged_dict[result.id]
                existing.score += (
                    result.score * self.search_weights['keyword_match']
                )
            else:
                # 새로운 결과 추가
                result.score = (
                    result.score * self.search_weights['keyword_match']
                )
                merged_dict[result.id] = result

        # 점수순 정렬
        merged_results = list(merged_dict.values())
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results

    def _calculate_keyword_score(self, query: str, name: str, description: Optional[str]) -> float:
        """키워드 매칭 점수 계산"""
        query_lower = query.lower()
        score = 0.0

        # 이름 매칭
        if name and query_lower in name.lower():
            score += 0.8

        # 설명 매칭
        if description and query_lower in description.lower():
            score += 0.2

        return min(score, 1.0)

    def _calculate_characteristic_score(self,
                                      note: FragranceNote,
                                      intensity_range: Optional[Tuple[float, float]],
                                      longevity_range: Optional[Tuple[float, float]],
                                      sillage_range: Optional[Tuple[float, float]]) -> float:
        """특성 매칭 점수 계산"""
        score = 0.0
        criteria_count = 0

        if intensity_range:
            if intensity_range[0] <= note.intensity <= intensity_range[1]:
                score += 1.0
            criteria_count += 1

        if longevity_range:
            if longevity_range[0] <= note.longevity <= longevity_range[1]:
                score += 1.0
            criteria_count += 1

        if sillage_range:
            if sillage_range[0] <= note.sillage <= sillage_range[1]:
                score += 1.0
            criteria_count += 1

        return score / max(criteria_count, 1)

    def _calculate_composition_score(self,
                                   recipe: Recipe,
                                   required_notes: List[str],
                                   optional_notes: Optional[List[str]],
                                   excluded_notes: Optional[List[str]]) -> float:
        """구성 매칭 점수 계산"""
        score = 0.0
        recipe_note_ids = [ing.note_id for ing in recipe.ingredients]

        # 필수 노트 점수
        if required_notes:
            matched_required = sum(1 for note_id in required_notes if note_id in recipe_note_ids)
            score += (matched_required / len(required_notes)) * 0.7

        # 선택 노트 점수
        if optional_notes:
            matched_optional = sum(1 for note_id in optional_notes if note_id in recipe_note_ids)
            score += (matched_optional / len(optional_notes)) * 0.2

        # 품질 점수 추가
        if recipe.quality_score:
            score += (recipe.quality_score / 10.0) * 0.1

        return min(score, 1.0)

    def _build_note_metadata(self, note: FragranceNote) -> Dict[str, Any]:
        """노트 메타데이터 구성"""
        return {
            'fragrance_family': note.fragrance_family,
            'note_type': note.note_type,
            'intensity': note.intensity,
            'longevity': note.longevity,
            'sillage': note.sillage,
            'mood_tags': note.mood_tags,
            'season_tags': note.season_tags,
            'gender_tags': note.gender_tags,
            'origin': note.origin,
            'grade': note.grade
        }

    def _build_recipe_metadata(self, recipe: Recipe) -> Dict[str, Any]:
        """레시피 메타데이터 구성"""
        return {
            'fragrance_family': recipe.fragrance_family,
            'recipe_type': recipe.recipe_type,
            'complexity': recipe.complexity,
            'quality_score': recipe.quality_score,
            'sillage': recipe.sillage,
            'longevity': recipe.longevity,
            'mood_tags': recipe.mood_tags,
            'season_tags': recipe.season_tags,
            'gender_tags': recipe.gender_tags,
            'status': recipe.status,
            'is_public': recipe.is_public,
            'ingredient_count': len(recipe.ingredients) if hasattr(recipe, 'ingredients') else 0
        }

    def _generate_cache_key(self, request: SearchRequest) -> str:
        """캐시 키 생성"""
        key_parts = [
            request.query,
            request.search_type,
            str(sorted(request.result_types or [])),
            str(sorted(request.filters.items()) if request.filters else ""),
            str(request.limit),
            str(request.offset)
        ]
        return "|".join(key_parts)

    async def _generate_suggestions(self, query: str) -> List[str]:
        """검색 제안 생성"""
        # 간단한 제안 생성 로직
        # 실제로는 더 정교한 제안 시스템 구현
        suggestions = []

        # 인기 검색어나 유사한 검색어 기반 제안
        common_suggestions = [
            "시트러스 향수", "플로럴 향수", "우디 향수", "오리엔탈 향수",
            "여름 향수", "겨울 향수", "로맨틱한 향수", "프레시한 향수"
        ]

        for suggestion in common_suggestions:
            if query.lower() in suggestion.lower() or suggestion.lower() in query.lower():
                suggestions.append(suggestion)

        return suggestions[:5]

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        import numpy as np

        # numpy 배열로 변환
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # 코사인 유사도 계산
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)
        return float(similarity)

    async def _get_or_create_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터를 가져오거나 생성"""
        # 캐시 확인
        cache_key = f"embedding:{text[:100]}"  # 텍스트 앞부분으로 캐시 키 생성

        if self.cache:
            cached_embedding = await self.cache.get(cache_key)
            if cached_embedding:
                return cached_embedding

        # 임베딩 생성
        if self.embedding_model:
            try:
                embedding = await self.embedding_model.encode_text(text)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                # 폴백: 간단한 해시 기반 벡터 생성
                embedding = self._generate_fallback_embedding(text)
        else:
            # 임베딩 모델이 없을 때 폴백
            embedding = self._generate_fallback_embedding(text)

        # 캐시 저장
        if self.cache:
            await self.cache.set(cache_key, embedding, ttl=3600)  # 1시간 캐시

        return embedding

    def _generate_fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """폴백 임베딩 생성 (해시 기반)"""
        import hashlib
        import numpy as np

        # 텍스트를 해시하여 시드 생성
        hash_object = hashlib.md5(text.encode())
        seed = int(hash_object.hexdigest()[:8], 16)

        # 시드 기반 랜덤 벡터 생성
        np.random.seed(seed)
        embedding = np.random.randn(dim)

        # 정규화
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()


# 전역 서비스 인스턴스 (의존성 주입으로 대체 가능)
_search_service_instance = None


def get_search_service() -> EnhancedSearchService:
    """검색 서비스 인스턴스 반환"""
    global _search_service_instance
    if _search_service_instance is None:
        _search_service_instance = EnhancedSearchService()
    return _search_service_instance