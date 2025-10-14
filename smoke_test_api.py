"""
API 스모크 테스트
3가지 모드(fast/balanced/creative)로 API 호출 테스트
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_mode(mode: str, brief: Dict[str, Any]) -> None:
    """
    특정 모드로 API 테스트

    Args:
        mode: fast/balanced/creative
        brief: Creative brief
    """
    print(f"\n{'='*60}")
    print(f"테스트 모드: {mode.upper()}")
    print(f"{'='*60}\n")

    # 1. DNA 생성
    print(f"[{mode}] 1. DNA 생성 중...")
    dna_request = {
        "brief": brief,
        "name": f"Test {mode.capitalize()} Formula",
        "description": f"Testing {mode} mode",
        "product_category": "eau_de_parfum"
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/dna/create",
            json=dna_request,
            timeout=30
        )
        response.raise_for_status()
        dna_data = response.json()
        dna_id = dna_data["dna_id"]

        print(f"   [OK] DNA 생성 완료: {dna_id}")
        print(f"   - 성분 수: {len(dna_data['ingredients'])}")
        print(f"   - 비용: ${dna_data.get('total_cost_per_kg', 0):.2f}/kg")
        print(f"   - IFRA 준수: {dna_data['compliance']['ifra_compliant']}")

        # 2. Evolution 옵션 생성
        print(f"\n[{mode}] 2. Evolution 옵션 생성 중...")
        options_request = {
            "dna_id": dna_id,
            "brief": brief,
            "num_options": 3,
            "optimization_profile": "commercial",
            "algorithm": "PPO"
        }

        response = requests.post(
            f"{API_BASE_URL}/evolve/options",
            json=options_request,
            timeout=30
        )
        response.raise_for_status()
        options_data = response.json()
        experiment_id = options_data["experiment_id"]

        print(f"   [OK] Evolution 옵션 생성 완료: {experiment_id}")
        print(f"   - 옵션 수: {len(options_data['options'])}")

        for i, option in enumerate(options_data['options'], 1):
            print(f"   - Option {i}: {option['action']}")
            print(f"     {option['description']}")

        if options_data.get('optimization_scores'):
            scores = options_data['optimization_scores']
            print(f"\n   [SCORES] 최적화 점수:")
            print(f"   - Creativity: {scores.get('creativity', 0):.3f}")
            print(f"   - Fitness: {scores.get('fitness', 0):.3f}")
            print(f"   - Stability: {scores.get('stability', 0):.3f}")
            print(f"   - Total: {scores.get('total', 0):.3f}")

        # 3. 피드백 전송 (RL 업데이트)
        print(f"\n[{mode}] 3. 피드백 전송 중...")
        chosen_option = options_data['options'][0]
        feedback_request = {
            "experiment_id": experiment_id,
            "chosen_id": chosen_option['id'],
            "rating": 4.0,
            "notes": f"Testing {mode} mode feedback"
        }

        response = requests.post(
            f"{API_BASE_URL}/evolve/feedback",
            json=feedback_request,
            timeout=30
        )
        response.raise_for_status()
        feedback_data = response.json()

        print(f"   [OK] 피드백 처리 완료")
        print(f"   - 상태: {feedback_data['status']}")
        print(f"   - Iteration: {feedback_data['iteration']}")

        if feedback_data.get('metrics'):
            metrics = feedback_data['metrics']
            print(f"\n   [METRICS] RL Update 메트릭:")
            print(f"   - Loss: {metrics.get('loss', 'N/A')}")
            print(f"   - Reward: {metrics.get('reward', 'N/A')}")
            print(f"   - Entropy: {metrics.get('entropy', 'N/A')}")
            print(f"   - Clip Fraction: {metrics.get('clip_fraction', 'N/A')}")

        # 4. Experiment 상태 확인
        print(f"\n[{mode}] 4. Experiment 상태 확인...")
        response = requests.get(
            f"{API_BASE_URL}/experiments/{experiment_id}",
            timeout=30
        )
        response.raise_for_status()
        status_data = response.json()

        print(f"   [OK] Experiment 상태")
        print(f"   - 상태: {status_data['status']}")
        print(f"   - Iterations: {status_data['iterations']}")
        print(f"   - Algorithm: {status_data['algorithm']}")

        print(f"\n[OK] {mode.upper()} 모드 테스트 완료!")

    except requests.exceptions.RequestException as e:
        print(f"\n   [ERROR] 오류 발생: {e}")
        if hasattr(e.response, 'text'):
            print(f"   응답: {e.response.text}")


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("API 스모크 테스트 시작")
    print("="*60)

    # Health check
    print("\n0. Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print(f"   [OK] API 서버 정상: {health['status']}")
    except Exception as e:
        print(f"   [ERROR] API 서버 연결 실패: {e}")
        print(f"   서버를 먼저 시작하세요: python -m uvicorn app.main:app --reload")
        return

    # 테스트 Brief (각 모드별)
    briefs = {
        "fast": {
            "style": "fresh",
            "intensity": 0.6,
            "complexity": 0.4,
            "masculinity": 0.5,
            "notes": ["citrus", "aquatic"]
        },
        "balanced": {
            "style": "floral",
            "intensity": 0.7,
            "complexity": 0.6,
            "masculinity": 0.4,
            "notes": ["floral", "fruity", "woody"]
        },
        "creative": {
            "style": "oriental",
            "intensity": 0.8,
            "complexity": 0.8,
            "masculinity": 0.6,
            "notes": ["oriental", "spicy", "woody", "amber"]
        }
    }

    # 각 모드별 테스트
    for mode in ["fast", "balanced", "creative"]:
        test_mode(mode, briefs[mode])
        time.sleep(1)  # Rate limiting

    print("\n" + "="*60)
    print("[OK] 전체 스모크 테스트 완료!")
    print("="*60)
    print("\n[LOG] 로그 확인:")
    print("   - LLM Brief JSON 로그 확인")
    print("   - RL Update 메트릭 로그 확인")
    print("   - Prometheus 메트릭: http://localhost:8000/metrics")


if __name__ == "__main__":
    main()
