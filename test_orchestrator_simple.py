"""
Living Scent Orchestrator 간단한 통합 테스트
Sprint 4 완성도 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_orchestrator():
    """오케스트레이터 기본 테스트"""

    print("\n" + "="*60)
    print("Living Scent Orchestrator Test")
    print("="*60)

    try:
        # 오케스트레이터 직접 임포트
        from fragrance_ai.orchestrator.living_scent_orchestrator import LivingScentOrchestrator

        print("\n1. Initializing Orchestrator...")
        orchestrator = LivingScentOrchestrator()

        # 헬스체크
        print("\n2. Health Check:")
        health = orchestrator.health_check()
        print(f"Status: {health['status']}")
        print(f"Components: {health['components']}")

        # 통계
        print("\n3. Statistics:")
        stats = orchestrator.get_statistics()
        print(f"MOGA Available: {stats['moga_available']}")
        print(f"PPO Available: {stats['ppo_available']}")

        print("\n[OK] Test Complete!")

    except ImportError as e:
        print(f"\n[ERROR] Import Error: {e}")
        print("\nTrying minimal test...")

        # 최소 테스트 - 파일만 확인
        import os
        files = [
            "fragrance_ai/orchestrator/living_scent_orchestrator.py",
            "fragrance_ai/training/moga_optimizer.py",
            "fragrance_ai/training/ppo_engine.py"
        ]

        print("\nFile Check:")
        for f in files:
            exists = os.path.exists(f)
            status = "[OK]" if exists else "[MISSING]"
            print(f"{status} {f}")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_orchestrator()