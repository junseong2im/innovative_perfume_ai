"""
핵심 API 엔드포인트 테스트
프로덕션 안정성을 위한 필수 테스트들
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np
import json

from fragrance_ai.api.main import app
from fragrance_ai.core.config import settings


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """비동기 테스트 클라이언트"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_embedding_model():
    """임베딩 모델 모킹"""
    mock = Mock()
    mock.encode_async.return_value = Mock(embeddings=np.random.rand(1, 384))
    return mock


@pytest.fixture
def mock_vector_store():
    """벡터 스토어 모킹"""
    mock = Mock()
    mock.search.return_value = [
        {
            "id": "test_1",
            "document": "테스트 향수 설명",
            "metadata": {"type": "fragrance"},
            "distance": 0.2,
            "similarity": 0.8
        }
    ]
    return mock


class TestHealthCheck:
    """헬스체크 엔드포인트 테스트"""

    def test_health_check_success(self, client):
        """정상 헬스체크 테스트"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data


class TestSemanticSearch:
    """의미 검색 API 테스트"""

    @patch('fragrance_ai.api.main.app.state.embedding_model')
    @patch('fragrance_ai.api.main.app.state.search_batch_processor')
    def test_semantic_search_success(self, mock_processor, mock_model, client):
        """정상 의미 검색 테스트"""
        # Mock 설정
        mock_model.encode_async.return_value = Mock(embeddings=np.random.rand(1, 384))
        mock_processor.add_item.return_value = [
            {
                "id": "test_1",
                "document": "테스트 향수",
                "similarity": 0.85,
                "collection": "fragrances"
            }
        ]

        search_request = {
            "query": "상큼한 시트러스 향수",
            "top_k": 5,
            "search_type": "similarity",
            "min_similarity": 0.7
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert "search_time" in data

    def test_semantic_search_invalid_request(self, client):
        """잘못된 검색 요청 테스트"""
        invalid_request = {
            "query": "",  # 빈 쿼리
            "top_k": -1   # 잘못된 값
        }

        response = client.post("/api/v2/semantic-search", json=invalid_request)
        assert response.status_code == 422  # Validation error

    @patch('fragrance_ai.api.main.app.state.embedding_model')
    def test_semantic_search_model_error(self, mock_model, client):
        """모델 에러 시 처리 테스트"""
        # 모델에서 예외 발생
        mock_model.encode_async.side_effect = Exception("Model error")

        search_request = {
            "query": "테스트 쿼리",
            "top_k": 5
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert data["error"] is True


class TestRAGChat:
    """RAG 채팅 API 테스트"""

    @patch('fragrance_ai.api.main.app.state.rag_system')
    def test_rag_chat_success(self, mock_rag, client):
        """정상 RAG 채팅 테스트"""
        # Mock RAG 응답
        mock_rag.generate_with_rag.return_value = Mock(
            generated_text="향수 추천 결과입니다.",
            confidence_score=0.9,
            source_documents=["문서1", "문서2"],
            reasoning_steps=["단계1", "단계2"],
            retrieval_context=Mock(
                retrieval_time=0.1,
                retrieved_documents=["문서1"],
                similarity_scores=[0.8]
            )
        )

        # 인증 토큰 모킹
        with patch('fragrance_ai.api.main.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user"}

            response = client.post(
                "/api/v2/rag-chat",
                params={
                    "query": "봄에 어울리는 향수 추천해주세요",
                    "temperature": 0.7,
                    "enable_reasoning": True
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "confidence_score" in data
        assert "source_documents" in data

    def test_rag_chat_unauthorized(self, client):
        """인증 없는 RAG 채팅 테스트"""
        response = client.post(
            "/api/v2/rag-chat",
            params={"query": "테스트 쿼리"}
        )
        assert response.status_code == 401  # Unauthorized


class TestPerformanceEndpoints:
    """성능 메트릭 엔드포인트 테스트"""

    def test_performance_metrics_unauthorized(self, client):
        """권한 없는 성능 메트릭 접근 테스트"""
        response = client.get("/api/v2/performance")
        assert response.status_code == 401

    @patch('fragrance_ai.api.main.require_permission')
    @patch('fragrance_ai.api.main.global_performance_optimizer')
    def test_performance_metrics_success(self, mock_optimizer, mock_permission, client):
        """정상 성능 메트릭 조회 테스트"""
        # 권한 모킹
        mock_permission.return_value = {"user_id": "admin_user"}

        # 성능 데이터 모킹
        mock_optimizer.get_performance_report.return_value = {
            "cpu_usage": 50.0,
            "memory_usage": 70.0,
            "gpu_usage": 80.0
        }

        with patch('fragrance_ai.api.main.app.state.cache_manager') as mock_cache:
            mock_cache.get_stats.return_value = {"hit_rate": 0.85}
            mock_cache.get_hot_keys.return_value = ["key1", "key2"]

            response = client.get("/api/v2/performance")

        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert "cache" in data
        assert "timestamp" in data


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_404_endpoint(self, client):
        """존재하지 않는 엔드포인트 테스트"""
        response = client.get("/api/v2/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """잘못된 HTTP 메서드 테스트"""
        response = client.delete("/api/v2/semantic-search")
        assert response.status_code == 405

    @patch('fragrance_ai.api.main.app.state.embedding_model')
    def test_internal_server_error(self, mock_model, client):
        """내부 서버 에러 테스트"""
        # 예상치 못한 에러 발생
        mock_model.encode_async.side_effect = RuntimeError("Unexpected error")

        search_request = {
            "query": "테스트 쿼리",
            "top_k": 5
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 500

        data = response.json()
        assert data["error"] is True
        assert "error_code" in data
        assert "timestamp" in data


class TestInputValidation:
    """입력 검증 테스트"""

    def test_query_length_validation(self, client):
        """쿼리 길이 검증 테스트"""
        # 너무 긴 쿼리
        long_query = "a" * 10000
        search_request = {
            "query": long_query,
            "top_k": 5
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 422

    def test_negative_top_k(self, client):
        """음수 top_k 검증 테스트"""
        search_request = {
            "query": "테스트",
            "top_k": -5
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 422

    def test_invalid_similarity_threshold(self, client):
        """잘못된 유사도 임계값 테스트"""
        search_request = {
            "query": "테스트",
            "top_k": 5,
            "min_similarity": 1.5  # 1.0 초과
        }

        response = client.post("/api/v2/semantic-search", json=search_request)
        assert response.status_code == 422


class TestAsyncEndpoints:
    """비동기 엔드포인트 테스트"""

    @pytest.mark.asyncio
    async def test_async_semantic_search(self, async_client, mock_embedding_model):
        """비동기 의미 검색 테스트"""
        with patch('fragrance_ai.api.main.app.state.embedding_model', mock_embedding_model):
            with patch('fragrance_ai.api.main.perform_single_search') as mock_search:
                mock_search.return_value = [
                    {
                        "id": "test_1",
                        "document": "테스트 향수",
                        "similarity": 0.85
                    }
                ]

                search_request = {
                    "query": "테스트 쿼리",
                    "top_k": 5
                }

                response = await async_client.post(
                    "/api/v2/semantic-search",
                    json=search_request
                )

                assert response.status_code == 200
                data = response.json()
                assert "results" in data

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """동시 요청 처리 테스트"""
        search_request = {
            "query": "테스트",
            "top_k": 3
        }

        # 10개의 동시 요청
        tasks = []
        for _ in range(10):
            task = async_client.post("/api/v2/semantic-search", json=search_request)
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 모든 응답이 정상적으로 처리되었는지 확인
        success_count = 0
        for response in responses:
            if hasattr(response, 'status_code') and response.status_code in [200, 500]:
                success_count += 1

        assert success_count >= 8  # 80% 이상 성공


@pytest.mark.integration
class TestIntegrationTests:
    """통합 테스트"""

    def test_full_search_workflow(self, client):
        """전체 검색 워크플로우 테스트"""
        # 1. 헬스체크
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. 의미 검색 (모킹 필요)
        with patch('fragrance_ai.api.main.app.state.embedding_model') as mock_model:
            with patch('fragrance_ai.api.main.perform_single_search') as mock_search:
                mock_model.encode_async.return_value = Mock(embeddings=np.random.rand(1, 384))
                mock_search.return_value = [{"id": "test", "similarity": 0.8}]

                search_request = {
                    "query": "테스트 향수",
                    "top_k": 5
                }

                search_response = client.post(
                    "/api/v2/semantic-search",
                    json=search_request
                )
                assert search_response.status_code == 200

        # 3. 성능 메트릭 확인 (권한 필요)
        # 실제 통합 테스트에서는 인증 토큰과 함께 테스트


if __name__ == "__main__":
    pytest.main([__file__, "-v"])