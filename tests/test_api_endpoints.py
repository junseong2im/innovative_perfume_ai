"""
FastAPI 엔드포인트 테스트
핵심 API 기능들의 동작을 검증합니다.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import status
import json
from unittest.mock import Mock, patch, AsyncMock

# Test client 설정을 위한 환경변수
import os
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["DEBUG"] = "true"

from fragrance_ai.api.main import app

# Test client 생성
client = TestClient(app)


class TestHealthEndpoint:
    """헬스 체크 엔드포인트 테스트"""

    def test_health_check_success(self):
        """헬스 체크가 정상적으로 응답하는지 확인"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime" in data
        assert "version" in data
        assert data["status"] == "healthy"

    def test_health_check_components(self):
        """헬스 체크에서 컴포넌트 상태를 확인"""
        response = client.get("/health")
        data = response.json()

        assert "components" in data
        components = data["components"]

        # 주요 컴포넌트들이 포함되어 있는지 확인
        expected_components = [
            "embedding_model",
            "rag_system",
            "cache_manager",
            "performance_optimizer"
        ]

        for component in expected_components:
            assert component in components


class TestRootEndpoint:
    """루트 엔드포인트 테스트"""

    def test_root_endpoint(self):
        """루트 엔드포인트가 올바른 정보를 반환하는지 확인"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "features" in data
        assert "endpoints" in data

        # 버전 정보 확인
        assert data["version"] == "2.0.0"

        # 주요 엔드포인트들이 포함되어 있는지 확인
        endpoints = data["endpoints"]
        expected_endpoints = [
            "docs",
            "semantic_search",
            "rag_chat",
            "generate_recipe",
            "performance"
        ]

        for endpoint in expected_endpoints:
            assert endpoint in endpoints


class TestSemanticSearchEndpoint:
    """시맨틱 검색 엔드포인트 테스트"""

    @patch('fragrance_ai.api.main.perform_single_search')
    def test_semantic_search_basic(self, mock_search):
        """기본 시맨틱 검색 기능 테스트"""
        # Mock 검색 결과 설정
        mock_results = [
            {
                "id": "test_1",
                "document": "테스트 향수 문서",
                "metadata": {"source": "test"},
                "distance": 0.1,
                "similarity": 0.9,
                "collection": "test_collection",
                "rank": 1
            }
        ]
        mock_search.return_value = mock_results

        # 검색 요청
        search_data = {
            "query": "테스트 향수",
            "search_type": "similarity",
            "top_k": 5,
            "min_similarity": 0.5
        }

        response = client.post("/api/v2/semantic-search", json=search_data)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert "query" in data
        assert "search_time" in data

        # 검색 결과 검증
        assert data["query"] == "테스트 향수"
        assert data["total_results"] >= 0
        assert isinstance(data["search_time"], (int, float))

    def test_semantic_search_validation_error(self):
        """잘못된 검색 요청에 대한 validation 에러 테스트"""
        # 빈 쿼리로 요청
        search_data = {
            "query": "",  # 빈 쿼리 - validation 에러 예상
            "search_type": "similarity"
        }

        response = client.post("/api/v2/semantic-search", json=search_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_semantic_search_invalid_type(self):
        """잘못된 검색 타입에 대한 에러 테스트"""
        search_data = {
            "query": "테스트",
            "search_type": "invalid_type"  # 잘못된 검색 타입
        }

        response = client.post("/api/v2/semantic-search", json=search_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRAGChatEndpoint:
    """RAG 채팅 엔드포인트 테스트"""

    @patch('fragrance_ai.api.main.app.state.rag_system')
    def test_rag_chat_basic(self, mock_rag_system):
        """기본 RAG 채팅 기능 테스트"""
        # Mock RAG 응답 설정
        mock_result = Mock()
        mock_result.generated_text = "테스트 RAG 응답입니다."
        mock_result.confidence_score = 0.85
        mock_result.source_documents = ["문서1", "문서2"]
        mock_result.reasoning_steps = ["단계1", "단계2"]
        mock_result.retrieval_context = Mock()
        mock_result.retrieval_context.retrieval_time = 0.1
        mock_result.retrieval_context.retrieved_documents = ["문서1", "문서2"]
        mock_result.retrieval_context.similarity_scores = [0.9, 0.8]

        mock_rag_system.generate_with_rag = AsyncMock(return_value=mock_result)

        # RAG 채팅 요청
        params = {
            "query": "향수에 대해 알려주세요",
            "temperature": 0.7,
            "enable_reasoning": True
        }

        response = client.post("/api/v2/rag-chat", params=params)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "response" in data
        assert "confidence_score" in data
        assert "source_documents" in data
        assert "reasoning_steps" in data
        assert "retrieval_info" in data
        assert "response_time" in data

        # 응답 내용 검증
        assert data["response"] == "테스트 RAG 응답입니다."
        assert data["confidence_score"] == 0.85
        assert len(data["source_documents"]) <= 3  # Top 3만 반환


class TestErrorHandling:
    """에러 처리 테스트"""

    def test_404_error(self):
        """존재하지 않는 엔드포인트 접근 테스트"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self):
        """허용되지 않는 HTTP 메서드 테스트"""
        response = client.put("/health")  # PUT은 허용되지 않음
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


class TestCORSConfiguration:
    """CORS 설정 테스트"""

    def test_cors_preflight(self):
        """CORS preflight 요청 테스트"""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }

        response = client.options("/api/v2/semantic-search", headers=headers)
        assert response.status_code == status.HTTP_200_OK

        # CORS 헤더 확인
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """테스트 환경 설정"""
    # 테스트 시작 전 설정
    yield
    # 테스트 완료 후 정리
    # 테스트 DB 파일 삭제 등
    import os
    if os.path.exists("./test.db"):
        os.remove("./test.db")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])