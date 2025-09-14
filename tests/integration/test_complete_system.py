#!/usr/bin/env python3
"""
Fragrance AI 시스템 통합 테스트
전체 시스템의 End-to-End 기능 검증
"""

import asyncio
import pytest
import httpx
import json
import logging
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
from pathlib import Path

# 테스트 대상 모듈들
from fragrance_ai.api.main import app
from fragrance_ai.core.config import settings
from fragrance_ai.models.embedding import AdvancedKoreanFragranceEmbedding
from fragrance_ai.models.generator import FragranceRecipeGenerator
from fragrance_ai.core.vector_store import VectorStore
from fragrance_ai.database.connection import get_db_session
from fragrance_ai.services.search_service import SearchService
from fragrance_ai.services.generation_service import GenerationService
from fragrance_ai.evaluation.advanced_evaluator import AdvancedModelEvaluator
from fragrance_ai.utils.web_scraper import FragranceDataScraper
from fragrance_ai.utils.data_cleaner import FragranceDataCleaner

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
class TestCompleteSystemIntegration:
    """완전한 시스템 통합 테스트 클래스"""

    @pytest.fixture(autouse=True)
    async def setup_system(self):
        """시스템 초기화"""
        self.base_url = f"http://localhost:{settings.api_port}"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True
        )

        # 컴포넌트 초기화
        self.embedding_model = AdvancedKoreanFragranceEmbedding()
        self.generator = FragranceRecipeGenerator()
        self.vector_store = VectorStore()
        self.search_service = SearchService()
        self.generation_service = GenerationService()
        self.evaluator = AdvancedModelEvaluator()

        # 테스트 데이터 준비
        self.test_data = await self._prepare_test_data()

        yield

        # 정리
        await self.client.aclose()

    async def _prepare_test_data(self) -> Dict[str, Any]:
        """테스트 데이터 준비"""
        return {
            'test_queries': [
                '따뜻하고 달콤한 겨울 향수',
                '상쾌한 시트러스 여름 향수',
                '우아한 플로럴 데이트 향수',
                '깊고 신비로운 밤 향수',
                '가벼운 일상 향수'
            ],
            'test_generation_prompts': [
                {
                    'prompt': '로맨틱한 데이트를 위한 향수',
                    'mood': 'romantic',
                    'season': 'spring'
                },
                {
                    'prompt': '비즈니스 미팅에 적합한 향수',
                    'mood': 'professional',
                    'season': 'all'
                },
                {
                    'prompt': '파티에서 돋보이는 향수',
                    'mood': 'energetic',
                    'season': 'winter'
                }
            ],
            'expected_response_fields': [
                'results', 'total_results', 'query', 'search_time'
            ]
        }

    async def test_system_health_check(self):
        """시스템 헬스체크 테스트"""
        logger.info("Testing system health check...")

        response = await self.client.get(f"{self.base_url}/health")

        assert response.status_code == 200
        health_data = response.json()

        assert health_data['status'] == 'healthy'
        assert 'models' in health_data
        assert health_data['models']['vector_store'] == True
        assert health_data['models']['embedding_model'] == True
        assert health_data['models']['generator'] == True

        logger.info("✅ System health check passed")

    async def test_embedding_model_functionality(self):
        """임베딩 모델 기능 테스트"""
        logger.info("Testing embedding model functionality...")

        start_time = time.time()

        # 단일 텍스트 임베딩
        test_text = "톰포드 블랙 오키드는 관능적이고 신비로운 향수입니다"
        embedding_result = await self.embedding_model.encode_async([test_text])

        assert embedding_result.embeddings.shape[0] == 1
        assert embedding_result.embeddings.shape[1] > 0  # 차원 확인
        assert embedding_result.processing_time > 0

        # 배치 임베딩
        test_texts = self.test_data['test_queries']
        batch_result = await self.embedding_model.encode_async(test_texts)

        assert batch_result.embeddings.shape[0] == len(test_texts)
        assert batch_result.embeddings.shape[1] > 0

        # 다면적 임베딩
        multi_aspect_result = self.embedding_model.encode_multi_aspect(test_texts[:2])

        assert 'base' in multi_aspect_result
        assert len(multi_aspect_result['base']) == 2

        processing_time = time.time() - start_time
        logger.info(f"✅ Embedding model test passed (took {processing_time:.2f}s)")

    async def test_generation_model_functionality(self):
        """생성 모델 기능 테스트"""
        logger.info("Testing generation model functionality...")

        start_time = time.time()

        for prompt_data in self.test_data['test_generation_prompts']:
            # 레시피 생성
            recipe = self.generator.generate_recipe(
                prompt=prompt_data['prompt'],
                recipe_type='detailed',
                mood=prompt_data['mood'],
                season=prompt_data['season']
            )

            # 기본 구조 검증
            assert 'name' in recipe
            assert 'description' in recipe
            assert 'top_notes' in recipe
            assert 'heart_notes' in recipe
            assert 'base_notes' in recipe
            assert 'instructions' in recipe

            # 내용 검증
            assert len(recipe['name']) > 0
            assert len(recipe['description']) > 10
            assert len(recipe['top_notes']) > 0
            assert len(recipe['heart_notes']) > 0
            assert len(recipe['base_notes']) > 0

            # 품질 평가
            quality_scores = self.generator.evaluate_recipe_quality(recipe)
            assert 'overall' in quality_scores
            assert 0 <= quality_scores['overall'] <= 1

            logger.info(f"Generated recipe: {recipe['name']} (quality: {quality_scores['overall']:.2f})")

        processing_time = time.time() - start_time
        logger.info(f"✅ Generation model test passed (took {processing_time:.2f}s)")

    async def test_vector_store_functionality(self):
        """벡터 스토어 기능 테스트"""
        logger.info("Testing vector store functionality...")

        # 테스트 데이터 준비
        test_documents = []
        for i, query in enumerate(self.test_data['test_queries']):
            embedding = await self.embedding_model.encode_async([query])
            test_documents.append({
                'id': f'test_doc_{i}',
                'embedding': embedding.embeddings[0].tolist(),
                'metadata': {
                    'text': query,
                    'type': 'test',
                    'index': i
                }
            })

        # 문서 추가
        self.vector_store.batch_add_documents(
            collection_name="test_collection",
            documents=test_documents
        )

        # 검색 테스트
        for query in self.test_data['test_queries'][:2]:
            results = self.vector_store.semantic_search(
                collection_name="test_collection",
                query=query,
                top_k=3
            )

            assert len(results) > 0
            assert len(results) <= 3

            for result in results:
                assert 'id' in result
                assert 'score' in result
                assert 'metadata' in result
                assert 0 <= result['score'] <= 1

        # 컬렉션 통계
        stats = self.vector_store.get_collection_stats("test_collection")
        assert stats['count'] == len(test_documents)

        logger.info("✅ Vector store test passed")

    async def test_search_api_endpoint(self):
        """검색 API 엔드포인트 테스트"""
        logger.info("Testing search API endpoint...")

        for query in self.test_data['test_queries']:
            # 의미 기반 검색 API 호출
            response = await self.client.post(
                f"{self.base_url}/api/v1/semantic-search",
                json={
                    'query': query,
                    'search_type': 'hybrid',
                    'top_k': 5,
                    'collections': ['fragrance_notes', 'recipes']
                }
            )

            assert response.status_code == 200

            result = response.json()

            # 응답 구조 검증
            for field in self.test_data['expected_response_fields']:
                assert field in result

            assert isinstance(result['results'], list)
            assert isinstance(result['total_results'], int)
            assert result['query'] == query
            assert result['search_time'] > 0

            # 결과 품질 검증
            if len(result['results']) > 0:
                for item in result['results']:
                    assert 'id' in item
                    assert 'score' in item
                    assert 'metadata' in item
                    assert 0 <= item['score'] <= 1

            logger.info(f"Search query '{query}' returned {len(result['results'])} results")

        logger.info("✅ Search API test passed")

    async def test_generation_api_endpoint(self):
        """생성 API 엔드포인트 테스트"""
        logger.info("Testing generation API endpoint...")

        for prompt_data in self.test_data['test_generation_prompts']:
            response = await self.client.post(
                f"{self.base_url}/api/v1/generate-recipe",
                json={
                    'prompt': prompt_data['prompt'],
                    'recipe_type': 'detailed',
                    'mood': prompt_data['mood'],
                    'season': prompt_data['season'],
                    'include_story': True
                }
            )

            assert response.status_code == 200

            result = response.json()

            # 응답 구조 검증
            assert 'recipe' in result
            assert 'quality_scores' in result
            assert 'generation_time' in result
            assert 'prompt' in result

            recipe = result['recipe']
            assert 'name' in recipe
            assert 'description' in recipe
            assert 'top_notes' in recipe
            assert 'heart_notes' in recipe
            assert 'base_notes' in recipe

            quality_scores = result['quality_scores']
            assert 'overall' in quality_scores
            assert 0 <= quality_scores['overall'] <= 1

            logger.info(f"Generated recipe API: {recipe['name']} (quality: {quality_scores['overall']:.2f})")

        logger.info("✅ Generation API test passed")

    async def test_batch_generation_api(self):
        """배치 생성 API 테스트"""
        logger.info("Testing batch generation API...")

        prompts = [data['prompt'] for data in self.test_data['test_generation_prompts']]

        response = await self.client.post(
            f"{self.base_url}/api/v1/batch-generate",
            json={
                'prompts': prompts,
                'batch_size': 2
            }
        )

        assert response.status_code == 200

        result = response.json()

        assert 'recipes' in result
        assert 'total_recipes' in result
        assert 'average_quality' in result
        assert 'generation_time' in result

        assert len(result['recipes']) == len(prompts)
        assert result['total_recipes'] == len(prompts)
        assert 0 <= result['average_quality'] <= 1

        for recipe in result['recipes']:
            assert 'name' in recipe
            assert 'description' in recipe
            assert len(recipe['name']) > 0
            assert len(recipe['description']) > 10

        logger.info(f"✅ Batch generation API test passed ({len(result['recipes'])} recipes)")

    async def test_database_integration(self):
        """데이터베이스 통합 테스트"""
        logger.info("Testing database integration...")

        try:
            async for session in get_db_session():
                # 테스트 쿼리 실행
                result = await session.execute("SELECT COUNT(*) as count FROM fragrance_notes")
                count_result = result.fetchone()

                assert count_result is not None

                # 샘플 데이터 조회
                result = await session.execute("SELECT * FROM fragrance_notes LIMIT 5")
                sample_data = result.fetchall()

                logger.info(f"Database contains {count_result.count if count_result else 0} fragrance records")
                logger.info(f"Sample data retrieved: {len(sample_data)} records")

                break
        except Exception as e:
            pytest.fail(f"Database integration test failed: {e}")

        logger.info("✅ Database integration test passed")

    async def test_end_to_end_workflow(self):
        """End-to-End 워크플로우 테스트"""
        logger.info("Testing end-to-end workflow...")

        start_time = time.time()

        # 1. 검색 수행
        search_query = "로맨틱한 장미 향수"
        search_response = await self.client.post(
            f"{self.base_url}/api/v1/semantic-search",
            json={
                'query': search_query,
                'search_type': 'hybrid',
                'top_k': 3
            }
        )

        assert search_response.status_code == 200
        search_results = search_response.json()

        # 2. 검색 결과 기반 레시피 생성
        generation_prompt = f"{search_query}와 유사한 향수 레시피를 만들어주세요"
        generation_response = await self.client.post(
            f"{self.base_url}/api/v1/generate-recipe",
            json={
                'prompt': generation_prompt,
                'recipe_type': 'premium',
                'mood': 'romantic',
                'include_story': True
            }
        )

        assert generation_response.status_code == 200
        generation_results = generation_response.json()

        # 3. 생성된 레시피 품질 검증
        recipe = generation_results['recipe']
        quality_scores = generation_results['quality_scores']

        assert quality_scores['overall'] > 0.5  # 최소 품질 기준

        # 4. 시스템 메트릭 확인
        metrics_response = await self.client.get(
            f"{self.base_url}/metrics",
            headers={'Authorization': 'Bearer test-token'}
        )

        # 인증이 없어도 계속 진행 (선택적)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            logger.info(f"System metrics: {metrics.get('system', {})}")

        total_time = time.time() - start_time

        logger.info(f"✅ End-to-end workflow completed in {total_time:.2f}s")
        logger.info(f"Search found {len(search_results.get('results', []))} results")
        logger.info(f"Generated recipe: '{recipe['name']}' with quality {quality_scores['overall']:.2f}")

    async def test_performance_benchmarks(self):
        """성능 벤치마크 테스트"""
        logger.info("Testing performance benchmarks...")

        # 검색 성능 테스트
        search_times = []
        for query in self.test_data['test_queries']:
            start_time = time.time()

            response = await self.client.post(
                f"{self.base_url}/api/v1/semantic-search",
                json={'query': query, 'top_k': 5}
            )

            end_time = time.time()
            search_time = end_time - start_time
            search_times.append(search_time)

            assert response.status_code == 200
            assert search_time < 3.0  # 3초 이내

        avg_search_time = sum(search_times) / len(search_times)

        # 생성 성능 테스트
        generation_times = []
        for prompt_data in self.test_data['test_generation_prompts']:
            start_time = time.time()

            response = await self.client.post(
                f"{self.base_url}/api/v1/generate-recipe",
                json={
                    'prompt': prompt_data['prompt'],
                    'recipe_type': 'basic'
                }
            )

            end_time = time.time()
            generation_time = end_time - start_time
            generation_times.append(generation_time)

            assert response.status_code == 200
            assert generation_time < 10.0  # 10초 이내

        avg_generation_time = sum(generation_times) / len(generation_times)

        logger.info(f"✅ Performance benchmarks:")
        logger.info(f"  Average search time: {avg_search_time:.2f}s")
        logger.info(f"  Average generation time: {avg_generation_time:.2f}s")

        # 성능 기준 검증
        assert avg_search_time < 2.0  # 평균 검색 시간 2초 이내
        assert avg_generation_time < 8.0  # 평균 생성 시간 8초 이내

    async def test_error_handling(self):
        """에러 핸들링 테스트"""
        logger.info("Testing error handling...")

        # 잘못된 검색 요청
        response = await self.client.post(
            f"{self.base_url}/api/v1/semantic-search",
            json={'invalid_field': 'test'}
        )
        assert response.status_code in [400, 422]

        # 빈 쿼리 테스트
        response = await self.client.post(
            f"{self.base_url}/api/v1/semantic-search",
            json={'query': '', 'top_k': 5}
        )
        # 빈 쿼리는 처리될 수 있음
        assert response.status_code in [200, 400]

        # 너무 긴 프롬프트 테스트
        very_long_prompt = "향수 " * 1000
        response = await self.client.post(
            f"{self.base_url}/api/v1/generate-recipe",
            json={'prompt': very_long_prompt}
        )
        # 서버가 적절히 처리하는지 확인
        assert response.status_code in [200, 400, 413]

        # 존재하지 않는 엔드포인트
        response = await self.client.get(f"{self.base_url}/api/v1/nonexistent")
        assert response.status_code == 404

        logger.info("✅ Error handling test passed")

    async def test_data_consistency(self):
        """데이터 일관성 테스트"""
        logger.info("Testing data consistency...")

        # 동일한 쿼리로 여러 번 검색하여 일관성 확인
        test_query = self.test_data['test_queries'][0]
        results = []

        for _ in range(3):
            response = await self.client.post(
                f"{self.base_url}/api/v1/semantic-search",
                json={'query': test_query, 'top_k': 5}
            )
            assert response.status_code == 200
            results.append(response.json())

            await asyncio.sleep(0.1)  # 짧은 대기

        # 결과 일관성 검증
        for i in range(1, len(results)):
            assert results[0]['total_results'] == results[i]['total_results']

            # 상위 결과 순서 일관성 (약간의 변동은 허용)
            if len(results[0]['results']) > 0 and len(results[i]['results']) > 0:
                top_result_0 = results[0]['results'][0]['id']
                top_result_i = results[i]['results'][0]['id']
                # 동일하거나 매우 유사한 점수여야 함
                assert abs(results[0]['results'][0]['score'] - results[i]['results'][0]['score']) < 0.01

        logger.info("✅ Data consistency test passed")

    async def test_concurrent_requests(self):
        """동시 요청 처리 테스트"""
        logger.info("Testing concurrent request handling...")

        async def make_search_request(query: str):
            response = await self.client.post(
                f"{self.base_url}/api/v1/semantic-search",
                json={'query': query, 'top_k': 3}
            )
            return response.status_code, response.json()

        # 동시에 여러 검색 요청 실행
        tasks = [
            make_search_request(query)
            for query in self.test_data['test_queries']
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # 모든 요청이 성공했는지 확인
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent request failed: {result}")

            status_code, response_data = result
            assert status_code == 200
            assert 'results' in response_data

        total_time = end_time - start_time
        avg_time_per_request = total_time / len(tasks)

        logger.info(f"✅ Concurrent requests test passed")
        logger.info(f"  {len(tasks)} requests completed in {total_time:.2f}s")
        logger.info(f"  Average time per request: {avg_time_per_request:.2f}s")

        # 동시 처리가 순차 처리보다 효율적인지 확인
        assert total_time < len(tasks) * 2.0  # 순차 처리 예상 시간보다 짧아야 함

    async def test_system_scalability(self):
        """시스템 확장성 테스트 (부하 테스트)"""
        logger.info("Testing system scalability...")

        # 점진적으로 부하 증가
        load_levels = [5, 10, 15]

        for load_level in load_levels:
            logger.info(f"Testing with {load_level} concurrent requests...")

            async def make_request():
                query = self.test_data['test_queries'][0]
                response = await self.client.post(
                    f"{self.base_url}/api/v1/semantic-search",
                    json={'query': query, 'top_k': 3}
                )
                return response.status_code == 200, response.elapsed.total_seconds()

            # 동시 요청 실행
            start_time = time.time()
            tasks = [make_request() for _ in range(load_level)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            # 결과 분석
            successful_requests = 0
            total_response_time = 0

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Request failed with exception: {result}")
                    continue

                success, response_time = result
                if success:
                    successful_requests += 1
                    total_response_time += response_time

            success_rate = successful_requests / load_level
            avg_response_time = total_response_time / max(successful_requests, 1)
            throughput = successful_requests / (end_time - start_time)

            logger.info(f"  Load level {load_level}: {success_rate:.1%} success rate")
            logger.info(f"  Average response time: {avg_response_time:.2f}s")
            logger.info(f"  Throughput: {throughput:.1f} requests/sec")

            # 성능 기준 검증
            assert success_rate >= 0.9  # 90% 이상 성공률
            assert avg_response_time < 5.0  # 5초 이내 응답

        logger.info("✅ System scalability test passed")


async def run_integration_tests():
    """통합 테스트 실행 함수"""

    print("\n" + "="*60)
    print("🚀 FRAGRANCE AI 시스템 통합 테스트 시작")
    print("="*60)

    test_instance = TestCompleteSystemIntegration()

    # 시스템 초기화
    await test_instance.setup_system()

    tests = [
        ("시스템 헬스체크", test_instance.test_system_health_check),
        ("임베딩 모델 기능", test_instance.test_embedding_model_functionality),
        ("생성 모델 기능", test_instance.test_generation_model_functionality),
        ("벡터 스토어 기능", test_instance.test_vector_store_functionality),
        ("검색 API 엔드포인트", test_instance.test_search_api_endpoint),
        ("생성 API 엔드포인트", test_instance.test_generation_api_endpoint),
        ("배치 생성 API", test_instance.test_batch_generation_api),
        ("데이터베이스 통합", test_instance.test_database_integration),
        ("End-to-End 워크플로우", test_instance.test_end_to_end_workflow),
        ("성능 벤치마크", test_instance.test_performance_benchmarks),
        ("에러 핸들링", test_instance.test_error_handling),
        ("데이터 일관성", test_instance.test_data_consistency),
        ("동시 요청 처리", test_instance.test_concurrent_requests),
        ("시스템 확장성", test_instance.test_system_scalability),
    ]

    passed_tests = 0
    failed_tests = 0
    start_time = time.time()

    for test_name, test_func in tests:
        try:
            print(f"\n🔍 {test_name} 테스트 중...")
            await test_func()
            passed_tests += 1

        except Exception as e:
            print(f"❌ {test_name} 테스트 실패: {e}")
            failed_tests += 1
            logger.error(f"Test {test_name} failed", exc_info=True)

    total_time = time.time() - start_time

    print(f"\n" + "="*60)
    print("📊 통합 테스트 결과 요약")
    print("="*60)
    print(f"총 테스트 수: {len(tests)}")
    print(f"통과: {passed_tests} ✅")
    print(f"실패: {failed_tests} ❌")
    print(f"성공률: {passed_tests/len(tests)*100:.1f}%")
    print(f"총 소요 시간: {total_time:.1f}초")

    if failed_tests == 0:
        print("\n🎉 모든 통합 테스트가 성공적으로 완료되었습니다!")
        print("✨ Fragrance AI 시스템이 정상적으로 작동하고 있습니다.")
    else:
        print(f"\n⚠️  {failed_tests}개의 테스트가 실패했습니다.")
        print("🔧 실패한 테스트를 확인하고 문제를 해결해주세요.")

    print("="*60)

    return failed_tests == 0


if __name__ == "__main__":
    # 통합 테스트 실행
    asyncio.run(run_integration_tests())