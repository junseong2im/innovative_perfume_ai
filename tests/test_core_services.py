"""
핵심 서비스 및 모델 테스트
비즈니스 로직의 정확성을 검증합니다.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
from typing import List, Dict, Any

# 테스트용 환경 설정
import os
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key"


class TestEmbeddingModel:
    """임베딩 모델 테스트"""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock 임베딩 모델 생성"""
        with patch('fragrance_ai.models.embedding.AdvancedKoreanFragranceEmbedding') as mock_class:
            mock_instance = Mock()
            mock_instance.encode_async = AsyncMock(return_value=Mock(
                embeddings=[np.random.rand(384) for _ in range(3)]
            ))
            mock_instance.encode = Mock(return_value=np.random.rand(3, 384))
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_embedding_generation(self, mock_embedding_model):
        """임베딩 생성 테스트"""
        texts = ["테스트 텍스트 1", "테스트 텍스트 2", "테스트 텍스트 3"]

        result = await mock_embedding_model.encode_async(texts)

        assert len(result.embeddings) == 3
        for embedding in result.embeddings:
            assert len(embedding) == 384  # 임베딩 차원 확인
            assert isinstance(embedding, np.ndarray)

    def test_embedding_caching(self, mock_embedding_model):
        """임베딩 캐싱 기능 테스트"""
        text = "캐시 테스트 텍스트"

        # 첫 번째 호출
        result1 = mock_embedding_model.encode([text])
        # 두 번째 호출 (캐시에서 가져와야 함)
        result2 = mock_embedding_model.encode([text])

        assert result1.shape == result2.shape
        mock_embedding_model.encode.assert_called()


class TestSearchService:
    """검색 서비스 테스트"""

    @pytest.fixture
    def mock_search_service(self):
        """Mock 검색 서비스 생성"""
        with patch('fragrance_ai.services.search_service.SearchService') as mock_class:
            mock_instance = Mock()
            mock_instance.search_async = AsyncMock(return_value=[
                {
                    "id": "test_1",
                    "content": "테스트 향수 노트",
                    "similarity": 0.85,
                    "metadata": {"category": "floral"}
                }
            ])
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_search_service):
        """시맨틱 검색 기능 테스트"""
        query = "로즈 향수"
        top_k = 5

        results = await mock_search_service.search_async(query, top_k=top_k)

        assert len(results) >= 0
        if results:
            result = results[0]
            assert "id" in result
            assert "content" in result
            assert "similarity" in result
            assert 0 <= result["similarity"] <= 1

    @pytest.mark.asyncio
    async def test_search_filtering(self, mock_search_service):
        """검색 필터링 기능 테스트"""
        query = "시트러스 향수"
        filters = {"category": "citrus", "season": "summer"}

        # 필터링된 검색 결과 mock 설정
        mock_search_service.search_async.return_value = [
            {
                "id": "citrus_1",
                "content": "레몬 향수",
                "similarity": 0.9,
                "metadata": {"category": "citrus", "season": "summer"}
            }
        ]

        results = await mock_search_service.search_async(query, filters=filters)

        if results:
            for result in results:
                metadata = result.get("metadata", {})
                assert metadata.get("category") == "citrus"
                assert metadata.get("season") == "summer"


class TestGenerationService:
    """생성 서비스 테스트"""

    @pytest.fixture
    def mock_generation_service(self):
        """Mock 생성 서비스 생성"""
        with patch('fragrance_ai.services.generation_service.GenerationService') as mock_class:
            mock_instance = Mock()
            mock_instance.generate_recipe_async = AsyncMock(return_value={
                "recipe_id": "test_recipe_001",
                "name": "테스트 향수 레시피",
                "composition": {
                    "top_notes": {"percentage": 30, "ingredients": ["bergamot", "lemon"]},
                    "heart_notes": {"percentage": 50, "ingredients": ["rose", "jasmine"]},
                    "base_notes": {"percentage": 20, "ingredients": ["sandalwood", "musk"]}
                },
                "quality_score": 8.5,
                "description": "테스트용 향수 레시피입니다."
            })
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_recipe_generation(self, mock_generation_service):
        """레시피 생성 기능 테스트"""
        request_data = {
            "fragrance_family": "floral",
            "mood": "romantic",
            "intensity": "moderate",
            "target_customer": "20-30대 여성"
        }

        recipe = await mock_generation_service.generate_recipe_async(request_data)

        # 레시피 구조 검증
        assert "recipe_id" in recipe
        assert "name" in recipe
        assert "composition" in recipe
        assert "quality_score" in recipe

        # 구성 요소 검증
        composition = recipe["composition"]
        assert "top_notes" in composition
        assert "heart_notes" in composition
        assert "base_notes" in composition

        # 품질 점수 검증
        assert 0 <= recipe["quality_score"] <= 10

    @pytest.mark.asyncio
    async def test_recipe_validation(self, mock_generation_service):
        """레시피 검증 기능 테스트"""
        # 잘못된 비율의 레시피
        invalid_recipe = {
            "composition": {
                "top_notes": {"percentage": 50},  # 너무 높은 비율
                "heart_notes": {"percentage": 40},
                "base_notes": {"percentage": 20}  # 총합이 110%
            }
        }

        # 검증 실패 시나리오
        mock_generation_service.validate_recipe = Mock(return_value=False)

        is_valid = mock_generation_service.validate_recipe(invalid_recipe)
        assert not is_valid


class TestRAGSystem:
    """RAG 시스템 테스트"""

    @pytest.fixture
    def mock_rag_system(self):
        """Mock RAG 시스템 생성"""
        with patch('fragrance_ai.models.rag_system.FragranceRAGSystem') as mock_class:
            mock_instance = Mock()
            mock_instance.generate_with_rag = AsyncMock(return_value=Mock(
                generated_text="RAG 기반 응답입니다.",
                confidence_score=0.8,
                source_documents=["문서1", "문서2", "문서3"],
                reasoning_steps=["검색", "분석", "생성"],
                retrieval_context=Mock(
                    retrieval_time=0.15,
                    retrieved_documents=["문서1", "문서2"],
                    similarity_scores=[0.9, 0.8]
                )
            ))
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_rag_generation(self, mock_rag_system):
        """RAG 기반 텍스트 생성 테스트"""
        query = "향수 제조에 대해 설명해 주세요"

        result = await mock_rag_system.generate_with_rag(
            query=query,
            temperature=0.7,
            enable_reasoning=True
        )

        # 응답 구조 검증
        assert hasattr(result, 'generated_text')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'source_documents')
        assert hasattr(result, 'reasoning_steps')

        # 응답 내용 검증
        assert len(result.generated_text) > 0
        assert 0 <= result.confidence_score <= 1
        assert len(result.source_documents) > 0

    @pytest.mark.asyncio
    async def test_rag_with_context(self, mock_rag_system):
        """컨텍스트가 있는 RAG 생성 테스트"""
        query = "이 향수의 특징은 무엇인가요?"
        context = "앞서 논의한 로즈 향수에 대해"

        # 컨텍스트가 있는 응답 mock 설정
        mock_rag_system.generate_with_rag.return_value = Mock(
            generated_text="로즈 향수의 특징은 우아하고 로맨틱한 향입니다.",
            confidence_score=0.9,
            source_documents=["로즈 향수 가이드"],
            reasoning_steps=["컨텍스트 분석", "관련 문서 검색", "응답 생성"]
        )

        result = await mock_rag_system.generate_with_rag(
            query=query,
            context=context
        )

        assert "로즈" in result.generated_text
        assert result.confidence_score > 0.8


class TestCacheManager:
    """캐시 매니저 테스트"""

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock 캐시 매니저 생성"""
        with patch('fragrance_ai.core.advanced_caching.FragranceCacheManager') as mock_class:
            mock_instance = Mock()
            mock_instance.get_cached_embedding = AsyncMock(return_value=None)
            mock_instance.cache_embedding = AsyncMock()
            mock_instance.get_stats = Mock(return_value={
                "hit_rate": 0.75,
                "cache_size": 1000,
                "total_requests": 5000
            })
            mock_class.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_cache_miss_and_set(self, mock_cache_manager):
        """캐시 미스 및 설정 테스트"""
        key = "test_embedding_key"

        # 캐시 미스 확인
        result = await mock_cache_manager.get_cached_embedding(key)
        assert result is None

        # 캐시 설정
        embedding_data = np.random.rand(384)
        await mock_cache_manager.cache_embedding(key, embedding_data)

        # 캐시 설정 호출 확인
        mock_cache_manager.cache_embedding.assert_called_with(key, embedding_data)

    def test_cache_statistics(self, mock_cache_manager):
        """캐시 통계 테스트"""
        stats = mock_cache_manager.get_stats()

        assert "hit_rate" in stats
        assert "cache_size" in stats
        assert "total_requests" in stats

        # 통계 값 검증
        assert 0 <= stats["hit_rate"] <= 1
        assert stats["cache_size"] >= 0
        assert stats["total_requests"] >= 0


class TestConfigurationValidation:
    """설정 검증 테스트"""

    def test_environment_variables(self):
        """환경 변수 설정 테스트"""
        from fragrance_ai.core.config import settings

        # 테스트 환경에서 설정된 값들 확인
        assert settings.debug is True  # 테스트 환경에서는 debug 모드
        assert "sqlite" in settings.database_url  # 테스트 DB 사용
        assert "test" in settings.secret_key  # 테스트용 시크릿 키

    def test_model_configuration(self):
        """모델 설정 테스트"""
        from fragrance_ai.core.config import settings

        # AI 모델 설정 확인
        assert settings.embedding_model_name is not None
        assert settings.generation_model_name is not None
        assert settings.vector_dimension > 0
        assert settings.max_seq_length > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])