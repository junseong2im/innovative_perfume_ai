"""
오케스트레이터 예외 처리 테스트 실행기
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.unit.test_orchestrator_exception_handling import (
    TestOrchestratorExceptionHandling,
    TestExceptionPropagation
)


async def run_tests():
    """테스트 실행"""
    print("=" * 70)
    print("오케스트레이터 예외 처리 테스트 시작")
    print("=" * 70)

    test_suite = TestOrchestratorExceptionHandling()
    exception_suite = TestExceptionPropagation()

    # 테스트 케이스 목록
    test_cases = [
        ("검색 도구 실패 시 폴백", test_suite.test_search_tool_failure_with_fallback),
        ("생성기 재시도 메커니즘", test_suite.test_generator_failure_with_retry),
        ("검증기 타임아웃 처리", test_suite.test_validator_timeout_handling),
        ("지식 도구 완전 실패", test_suite.test_knowledge_tool_complete_failure),
        ("여러 도구 연속 실패", test_suite.test_multiple_tool_failures_in_sequence),
        ("Circuit Breaker 활성화", test_suite.test_circuit_breaker_activation),
        ("모든 도구 실패 시 응답", test_suite.test_process_message_with_all_tools_failing),
        ("부분 성공 집계", test_suite.test_partial_success_aggregation),
        ("사용자 친화적 에러 메시지", test_suite.test_graceful_error_message_formatting),
        ("예외 로깅", exception_suite.test_exception_logging),
        ("예외 컨텍스트 보존", exception_suite.test_exception_context_preservation),
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in test_cases:
        try:
            print(f"\n테스트: {test_name}")
            print("-" * 50)

            # 테스트에 필요한 픽스처 준비
            orchestrator = test_suite.orchestrator()
            context = test_suite.context() if hasattr(test_suite, 'context') else None

            # 테스트 실행
            if asyncio.iscoroutinefunction(test_func):
                if context and test_func.__code__.co_argcount > 2:
                    await test_func(test_suite, orchestrator, context)
                else:
                    await test_func(test_suite, orchestrator)
            else:
                if context and test_func.__code__.co_argcount > 2:
                    test_func(test_suite, orchestrator, context)
                else:
                    test_func(test_suite, orchestrator)

            print(f"✅ {test_name} - 통과")
            passed += 1

        except AssertionError as e:
            print(f"❌ {test_name} - 실패: {e}")
            failed += 1
            errors.append((test_name, str(e)))
        except Exception as e:
            print(f"⚠️ {test_name} - 에러: {e}")
            failed += 1
            errors.append((test_name, f"예외 발생: {e}"))

    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)
    print(f"✅ 통과: {passed}")
    print(f"❌ 실패: {failed}")
    print(f"📊 총계: {passed + failed}")

    if errors:
        print("\n실패한 테스트 상세:")
        for test_name, error in errors:
            print(f"\n- {test_name}:")
            print(f"  {error}")

    return passed, failed


async def verify_exception_handling():
    """예외 처리가 실제로 작동하는지 간단한 검증"""
    print("\n" + "=" * 70)
    print("실제 예외 처리 동작 검증")
    print("=" * 70)

    from fragrance_ai.orchestrator.artisan_orchestrator import (
        ArtisanOrchestrator,
        ConversationContext,
        ToolExecutionResult,
        ToolStatus
    )

    orchestrator = ArtisanOrchestrator()
    context = ConversationContext(
        user_id="test",
        conversation_id="test",
        history=[]
    )

    # 도구를 강제로 실패시키는 간단한 테스트
    print("\n1. 도구 실패 시뮬레이션...")

    async def failing_tool(*args, **kwargs):
        raise Exception("의도적인 실패")

    orchestrator.tools["hybrid_search"] = failing_tool

    # 폴백 설정
    original_fallback = orchestrator._execute_fallback

    async def mock_fallback(tool, params):
        print(f"   - 폴백 호출됨: {tool}")
        return ToolExecutionResult(
            tool=tool,
            status=ToolStatus.PARTIAL,
            result={"fallback": True, "message": "폴백 응답"},
            fallback_used=True
        )

    orchestrator._execute_fallback = mock_fallback

    # 도구 실행
    result = await orchestrator._execute_single_tool_with_resilience({
        "tool": "hybrid_search",
        "params": {"query": "test"}
    })

    print(f"   - 결과 상태: {result.status}")
    print(f"   - 폴백 사용: {result.fallback_used}")
    print(f"   - 에러: {result.error}")

    assert result.status in [ToolStatus.PARTIAL, ToolStatus.FAILED], "예외 처리 실패"
    assert result.fallback_used or result.error, "폴백 또는 에러 처리 실패"

    print("\n✅ 예외 처리가 올바르게 작동합니다!")

    # Circuit Breaker 테스트
    print("\n2. Circuit Breaker 테스트...")

    # 원래 폴백 복원
    orchestrator._execute_fallback = original_fallback

    # 여러 번 실패시켜 Circuit Breaker 활성화
    for i in range(5):
        result = await orchestrator._execute_single_tool_with_resilience({
            "tool": "hybrid_search",
            "params": {"query": f"test_{i}"}
        })

    breaker = orchestrator.circuit_breakers.get("hybrid_search")
    if breaker:
        print(f"   - Circuit Breaker 상태: {'OPEN' if breaker.is_open else 'CLOSED'}")
        print(f"   - 실패 횟수: {breaker.failures}")
        assert breaker.failures >= 3, "Circuit Breaker가 실패를 추적하지 않음"
        print("\n✅ Circuit Breaker가 올바르게 작동합니다!")
    else:
        print("   ⚠️ Circuit Breaker가 생성되지 않음")


if __name__ == "__main__":
    # 이벤트 루프 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 실제 동작 검증
        loop.run_until_complete(verify_exception_handling())

        # 전체 테스트 실행
        passed, failed = loop.run_until_complete(run_tests())

        # 종료 코드 설정
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        loop.close()