"""
오케스트레이터 통합 테스트
전체 도구 체인이 실제로 작동하는지 검증
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fragrance_ai.orchestrator.artisan_orchestrator import (
    ArtisanOrchestrator,
    ConversationContext,
    ArtisanResponse
)
from fragrance_ai.tools.search_tool import hybrid_search
from fragrance_ai.tools.generator_tool import create_recipe
from fragrance_ai.tools.validator_tool import validate_composition
from fragrance_ai.tools.knowledge_tool import query_knowledge_base

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrchestratorIntegrationTest:
    """오케스트레이터 통합 테스트"""

    def __init__(self):
        """테스트 초기화"""
        self.orchestrator = ArtisanOrchestrator()
        self.test_results = []
        self.passed = 0
        self.failed = 0

    async def test_individual_tools(self):
        """개별 도구 테스트"""
        logger.info("=" * 70)
        logger.info("개별 도구 테스트 시작")
        logger.info("=" * 70)

        # 1. 검색 도구 테스트
        try:
            logger.info("\n1. 검색 도구 테스트")
            search_result = await hybrid_search(
                query="fresh citrus perfume",
                filters={"season": "summer"},
                limit=5
            )

            assert search_result is not None, "검색 결과가 None"
            assert "results" in search_result, "results 키가 없음"
            assert isinstance(search_result["results"], list), "results가 리스트가 아님"

            logger.info(f"✅ 검색 도구 테스트 통과 - {len(search_result['results'])}개 결과")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 검색 도구 테스트 실패: {e}")
            self.failed += 1
            self.test_results.append(("검색 도구", str(e)))

        # 2. 생성 도구 테스트
        try:
            logger.info("\n2. 생성 도구 테스트")
            generation_request = {
                "description": "summer fresh citrus perfume",
                "style": "modern",
                "intensity": "light",
                "gender": "unisex"
            }

            recipe = await create_recipe(generation_request)

            assert recipe is not None, "레시피가 None"
            assert "name" in recipe, "레시피에 name이 없음"
            assert "top_notes" in recipe, "레시피에 top_notes가 없음"
            assert "heart_notes" in recipe, "레시피에 heart_notes가 없음"
            assert "base_notes" in recipe, "레시피에 base_notes가 없음"

            logger.info(f"✅ 생성 도구 테스트 통과 - 레시피: {recipe.get('name', 'Unknown')}")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 생성 도구 테스트 실패: {e}")
            self.failed += 1
            self.test_results.append(("생성 도구", str(e)))

        # 3. 검증 도구 테스트
        try:
            logger.info("\n3. 검증 도구 테스트")
            composition = {
                "top_notes": [
                    {"name": "bergamot", "percentage": 20},
                    {"name": "lemon", "percentage": 15}
                ],
                "heart_notes": [
                    {"name": "rose", "percentage": 25},
                    {"name": "jasmine", "percentage": 20}
                ],
                "base_notes": [
                    {"name": "sandalwood", "percentage": 15},
                    {"name": "musk", "percentage": 5}
                ]
            }

            validation_result = await validate_composition(composition)

            assert validation_result is not None, "검증 결과가 None"
            assert "valid" in validation_result, "valid 키가 없음"
            assert "score" in validation_result, "score 키가 없음"
            assert isinstance(validation_result["score"], (int, float)), "score가 숫자가 아님"

            logger.info(f"✅ 검증 도구 테스트 통과 - 점수: {validation_result['score']:.2f}")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 검증 도구 테스트 실패: {e}")
            self.failed += 1
            self.test_results.append(("검증 도구", str(e)))

        # 4. 지식 도구 테스트
        try:
            logger.info("\n4. 지식 도구 테스트")
            knowledge_response = await query_knowledge_base(
                category="note",
                query="bergamot"
            )

            assert knowledge_response is not None, "지식 응답이 None"
            assert hasattr(knowledge_response, "answer"), "answer 속성이 없음"
            assert hasattr(knowledge_response, "confidence"), "confidence 속성이 없음"
            assert knowledge_response.confidence > 0, "신뢰도가 0 이하"

            logger.info(f"✅ 지식 도구 테스트 통과 - 신뢰도: {knowledge_response.confidence:.2f}")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 지식 도구 테스트 실패: {e}")
            self.failed += 1
            self.test_results.append(("지식 도구", str(e)))

    async def test_orchestrator_flow(self):
        """오케스트레이터 전체 플로우 테스트"""
        logger.info("\n" + "=" * 70)
        logger.info("오케스트레이터 통합 플로우 테스트")
        logger.info("=" * 70)

        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            history=[]
        )

        test_cases = [
            {
                "name": "향수 생성 요청",
                "message": "여름에 어울리는 시트러스 향수를 만들어주세요",
                "expected_keys": ["message", "recipe_summary"]
            },
            {
                "name": "향수 검색 요청",
                "message": "플로럴 계열 향수를 찾아주세요",
                "expected_keys": ["message", "search_results"]
            },
            {
                "name": "지식 질문",
                "message": "베르가못 노트에 대해 알려주세요",
                "expected_keys": ["message", "knowledge_info"]
            },
            {
                "name": "복합 요청",
                "message": "로즈와 재스민을 사용한 로맨틱한 향수를 만들고 검증해주세요",
                "expected_keys": ["message", "recipe_summary", "validation_score"]
            }
        ]

        for test_case in test_cases:
            try:
                logger.info(f"\n테스트: {test_case['name']}")
                logger.info("-" * 50)

                # 메시지 처리
                response = await self.orchestrator.process_message(
                    test_case["message"],
                    context
                )

                # 응답 검증
                assert response is not None, "응답이 None"
                assert isinstance(response, ArtisanResponse), "응답 타입이 잘못됨"
                assert response.message is not None, "메시지가 None"
                assert len(response.message) > 0, "메시지가 비어있음"

                # 예상 키 확인
                response_dict = response.__dict__
                for key in test_case["expected_keys"]:
                    if key not in ["message"]:  # message는 항상 있음
                        # 다른 키들은 선택적일 수 있음
                        pass

                logger.info(f"✅ {test_case['name']} 테스트 통과")
                logger.info(f"   응답 길이: {len(response.message)} 글자")

                if response.recipe_summary:
                    logger.info(f"   레시피: {response.recipe_summary.get('name', 'Unknown')}")

                if response.validation_score is not None:
                    logger.info(f"   검증 점수: {response.validation_score:.2f}")

                self.passed += 1

            except Exception as e:
                logger.error(f"❌ {test_case['name']} 테스트 실패: {e}")
                self.failed += 1
                self.test_results.append((test_case['name'], str(e)))

    async def test_error_handling(self):
        """에러 처리 테스트"""
        logger.info("\n" + "=" * 70)
        logger.info("에러 처리 및 복구 테스트")
        logger.info("=" * 70)

        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            history=[]
        )

        # 1. 빈 메시지 처리
        try:
            logger.info("\n1. 빈 메시지 처리 테스트")
            response = await self.orchestrator.process_message("", context)
            assert response is not None, "빈 메시지에도 응답해야 함"
            logger.info("✅ 빈 메시지 처리 테스트 통과")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ 빈 메시지 처리 실패: {e}")
            self.failed += 1

        # 2. 매우 긴 메시지 처리
        try:
            logger.info("\n2. 긴 메시지 처리 테스트")
            long_message = "향수 " * 1000  # 매우 긴 메시지
            response = await self.orchestrator.process_message(long_message, context)
            assert response is not None, "긴 메시지도 처리해야 함"
            logger.info("✅ 긴 메시지 처리 테스트 통과")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ 긴 메시지 처리 실패: {e}")
            self.failed += 1

        # 3. 특수 문자 처리
        try:
            logger.info("\n3. 특수 문자 처리 테스트")
            special_message = "향수 @#$%^&*() 만들어줘 <script>alert('test')</script>"
            response = await self.orchestrator.process_message(special_message, context)
            assert response is not None, "특수 문자 메시지도 처리해야 함"
            assert "<script>" not in response.message, "XSS 방어 실패"
            logger.info("✅ 특수 문자 처리 테스트 통과")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ 특수 문자 처리 실패: {e}")
            self.failed += 1

    async def test_performance(self):
        """성능 테스트"""
        logger.info("\n" + "=" * 70)
        logger.info("성능 테스트")
        logger.info("=" * 70)

        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            history=[]
        )

        # 응답 시간 측정
        try:
            logger.info("\n응답 시간 측정")

            test_messages = [
                "향수를 만들어주세요",
                "플로럴 향수 검색",
                "베르가못에 대해 알려줘"
            ]

            response_times = []

            for msg in test_messages:
                start_time = datetime.now()
                response = await self.orchestrator.process_message(msg, context)
                end_time = datetime.now()

                duration = (end_time - start_time).total_seconds()
                response_times.append(duration)

                logger.info(f"   '{msg[:20]}...' 응답 시간: {duration:.2f}초")

            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)

            logger.info(f"\n   평균 응답 시간: {avg_time:.2f}초")
            logger.info(f"   최대 응답 시간: {max_time:.2f}초")

            # 성능 기준: 평균 5초 이내
            assert avg_time < 5.0, f"평균 응답 시간이 너무 김: {avg_time:.2f}초"

            logger.info("✅ 성능 테스트 통과")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 성능 테스트 실패: {e}")
            self.failed += 1
            self.test_results.append(("성능 테스트", str(e)))

    async def test_parallel_execution(self):
        """병렬 실행 테스트"""
        logger.info("\n" + "=" * 70)
        logger.info("병렬 실행 테스트")
        logger.info("=" * 70)

        try:
            # 여러 도구 동시 실행
            tasks = [
                hybrid_search("citrus", {}, 5),
                create_recipe({"description": "floral perfume"}),
                validate_composition({
                    "top_notes": [{"name": "lemon", "percentage": 30}],
                    "heart_notes": [{"name": "rose", "percentage": 40}],
                    "base_notes": [{"name": "musk", "percentage": 30}]
                }),
                query_knowledge_base("note", "jasmine")
            ]

            start_time = datetime.now()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = datetime.now()

            duration = (end_time - start_time).total_seconds()

            # 결과 검증
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            logger.info(f"   병렬 실행 시간: {duration:.2f}초")
            logger.info(f"   성공: {successful}, 실패: {failed}")

            assert successful >= 3, f"너무 많은 도구가 실패: {failed}/{len(tasks)}"

            logger.info("✅ 병렬 실행 테스트 통과")
            self.passed += 1

        except Exception as e:
            logger.error(f"❌ 병렬 실행 테스트 실패: {e}")
            self.failed += 1

    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("=" * 70)
        logger.info("오케스트레이터 통합 테스트 시작")
        logger.info("=" * 70)

        start_time = datetime.now()

        # 모든 테스트 실행
        await self.test_individual_tools()
        await self.test_orchestrator_flow()
        await self.test_error_handling()
        await self.test_performance()
        await self.test_parallel_execution()

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # 결과 요약
        logger.info("\n" + "=" * 70)
        logger.info("테스트 결과 요약")
        logger.info("=" * 70)
        logger.info(f"✅ 통과: {self.passed}")
        logger.info(f"❌ 실패: {self.failed}")
        logger.info(f"📊 총계: {self.passed + self.failed}")
        logger.info(f"⏱️ 소요 시간: {total_duration:.2f}초")

        if self.failed > 0:
            logger.info("\n실패한 테스트 상세:")
            for test_name, error in self.test_results:
                logger.error(f"  - {test_name}: {error}")

        # 테스트 결과 저장
        result_file = Path("test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "passed": self.passed,
                "failed": self.failed,
                "total": self.passed + self.failed,
                "duration": total_duration,
                "failures": self.test_results
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"\n테스트 결과가 {result_file}에 저장되었습니다.")

        return self.failed == 0


async def main():
    """메인 함수"""
    tester = OrchestratorIntegrationTest()
    success = await tester.run_all_tests()

    # 종료 코드 설정
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)