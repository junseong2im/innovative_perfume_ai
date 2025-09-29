"""
진짜 AI 시스템 검증 테스트
실제 도메인 지식과 의미있는 최적화가 일어나는지 확인
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict, List, Tuple


def test_real_moga():
    """진짜 MOGA 시스템 테스트"""
    print("\n" + "="*70)
    print("진짜 MOGA 시스템 검증")
    print("="*70)

    from fragrance_ai.training.real_moga_optimizer import RealFragranceMOGA

    # MOGA 옵티마이저 생성
    moga = RealFragranceMOGA(
        population_size=50,
        max_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.3
    )

    print("\n1. 초기 개체군 생성...")
    moga.initialize_population()

    # 초기 개체 분석
    initial_best = max(moga.population, key=lambda x: x.fitness)
    print(f"   초기 최고 적합도: {initial_best.fitness:.4f}")
    print(f"   초기 최고 조화도: {initial_best.objectives['harmony']:.4f}")
    print(f"   초기 최고 지속력: {initial_best.objectives['longevity']:.4f}")
    print(f"   초기 최고 확산력: {initial_best.objectives['sillage']:.4f}")

    # 초기 향수 구성
    print("\n   초기 최고 향수 구성:")
    print(f"   Top Notes: {list(initial_best.top_genes.keys())[:3]}")
    print(f"   Middle Notes: {list(initial_best.middle_genes.keys())[:3]}")
    print(f"   Base Notes: {list(initial_best.base_genes.keys())[:3]}")

    print("\n2. 진화 실행 (20세대)...")
    for gen in range(20):
        moga.evolve_generation()
        if gen % 5 == 0:
            best = max(moga.population, key=lambda x: x.fitness)
            print(f"   세대 {gen}: 적합도={best.fitness:.4f}, "
                  f"조화도={best.objectives['harmony']:.4f}, "
                  f"Pareto front={len(moga.pareto_front)}")

    # 최종 결과
    final_best = moga.get_best_solution()
    print(f"\n3. 최종 결과:")
    print(f"   최종 적합도: {final_best.fitness:.4f}")
    print(f"   최종 조화도: {final_best.objectives['harmony']:.4f}")
    print(f"   최종 지속력: {final_best.objectives['longevity']:.4f}")
    print(f"   최종 확산력: {final_best.objectives['sillage']:.4f}")
    print(f"   최종 균형감: {final_best.objectives['balance']:.4f}")

    # 최종 향수 구성
    print("\n   최종 향수 구성:")
    print(f"   Top Notes:")
    for ing, conc in list(final_best.top_genes.items())[:5]:
        print(f"      - {ing}: {conc:.1f}%")
    print(f"   Middle Notes:")
    for ing, conc in list(final_best.middle_genes.items())[:5]:
        print(f"      - {ing}: {conc:.1f}%")
    print(f"   Base Notes:")
    for ing, conc in list(final_best.base_genes.items())[:5]:
        print(f"      - {ing}: {conc:.1f}%")

    # 개선 확인
    improvement = final_best.fitness - initial_best.fitness
    if improvement > 0.05:
        print(f"\n   [SUCCESS] 실제 최적화 확인! 개선율: {improvement:.4f}")
        return True
    else:
        print(f"\n   [FAIL] 최적화 미미함. 개선율: {improvement:.4f}")
        return False


def test_real_rlhf():
    """진짜 RLHF 시스템 테스트"""
    print("\n" + "="*70)
    print("진짜 RLHF 시스템 검증")
    print("="*70)

    from fragrance_ai.training.real_rlhf import (
        RealFragranceRLHF, FragranceState, FragranceAction, Experience
    )
    import torch

    # RLHF 시스템 생성
    rlhf = RealFragranceRLHF(state_dim=46)  # 실제 차원

    print("\n1. 초기 향수 상태 생성...")
    initial_state = FragranceState(
        top_notes={'bergamot': 15, 'lemon': 10, 'orange': 8},
        middle_notes={'rose': 20, 'jasmine': 15, 'ylang_ylang': 12},
        base_notes={'sandalwood': 15, 'musk': 10, 'vanilla': 15},
        current_metrics={
            'harmony': 0.6,
            'longevity': 0.5,
            'sillage': 0.4,
            'balance': 0.65
        },
        user_preferences={
            'fresh': 0.7,
            'floral': 0.8,
            'woody': 0.5,
            'oriental': 0.3
        },
        season='spring',
        occasion='daily'
    )

    print(f"   초기 조화도: {initial_state.current_metrics['harmony']:.3f}")
    print(f"   초기 지속력: {initial_state.current_metrics['longevity']:.3f}")

    print("\n2. 경험 수집 (100개 에피소드)...")
    total_reward = 0
    improvements = []

    for episode in range(100):
        # 현재 상태
        current_state = initial_state

        # 행동 선택
        action = rlhf.select_action(current_state, epsilon=0.2)

        # 행동 적용
        next_state = rlhf.apply_action(current_state, action)

        # 사용자 피드백 시뮬레이션
        user_rating = np.random.uniform(3, 9)  # 3-9점
        user_feedback = "better balance" if np.random.random() > 0.5 else "too strong"

        # 보상 계산
        reward = rlhf.calculate_real_reward(
            current_state, action, next_state,
            user_rating, user_feedback
        )
        total_reward += reward

        # 개선 측정
        harmony_improvement = next_state.current_metrics['harmony'] - current_state.current_metrics['harmony']
        improvements.append(harmony_improvement)

        # 경험 저장
        experience = Experience(
            state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=(episode % 10 == 9),
            user_feedback=user_feedback
        )
        rlhf.store_experience(experience)

        if episode % 25 == 0:
            print(f"   에피소드 {episode}: 평균 보상={total_reward/(episode+1):.3f}, "
                  f"행동={action.action_type} on {action.target_note}")

    print(f"\n3. PPO 학습 실행...")
    # 학습 전 가중치
    initial_weights = rlhf.policy_net.fc1.weight.data.clone()

    # PPO 학습
    rlhf.train_ppo(batch_size=32, epochs=20)

    # 학습 후 가중치
    final_weights = rlhf.policy_net.fc1.weight.data
    weight_change = torch.mean(torch.abs(final_weights - initial_weights)).item()

    print(f"   가중치 변화량: {weight_change:.6f}")
    print(f"   평균 개선율: {np.mean(improvements):.4f}")
    print(f"   최종 평균 보상: {total_reward/100:.3f}")

    # 학습 통계
    if rlhf.training_stats['policy_loss']:
        print(f"   정책 손실: {np.mean(rlhf.training_stats['policy_loss']):.4f}")
        print(f"   가치 손실: {np.mean(rlhf.training_stats['value_loss']):.4f}")

    if weight_change > 0.001 and total_reward/100 > 0:
        print(f"\n   [SUCCESS] 실제 학습 확인! 가중치 변화: {weight_change:.6f}")
        return True
    else:
        print(f"\n   [FAIL] 학습 미미함. 가중치 변화: {weight_change:.6f}")
        return False


def test_fragrance_chemistry():
    """향수 화학 도메인 지식 테스트"""
    print("\n" + "="*70)
    print("향수 화학 도메인 지식 검증")
    print("="*70)

    from fragrance_ai.domain.fragrance_chemistry import FragranceChemistry

    chemistry = FragranceChemistry()

    print("\n1. 실제 향수 레시피 평가...")

    # 유명한 향수 구조 모방
    # Chanel No.5 스타일 (Aldehydic Floral)
    chanel_style = {
        'top': [('aldehydes', 15), ('bergamot', 10), ('lemon', 8)],
        'middle': [('rose', 25), ('jasmine', 20), ('ylang_ylang', 15)],
        'base': [('sandalwood', 10), ('vanilla', 5), ('musk', 2)]
    }

    evaluation = chemistry.evaluate_fragrance_complete(
        chanel_style['top'],
        chanel_style['middle'],
        chanel_style['base']
    )

    print(f"   Chanel No.5 스타일:")
    print(f"   - 조화도: {evaluation['harmony']:.3f}")
    print(f"   - 지속력: {evaluation['longevity']:.3f}")
    print(f"   - 확산력: {evaluation['sillage']:.3f}")
    print(f"   - 균형감: {evaluation['balance']:.3f}")
    print(f"   - 전체 점수: {evaluation['overall']:.3f}")

    print("\n2. 조화도 매트릭스 테스트...")
    # 잘 어울리는 조합
    good_combo = [('bergamot', 20), ('lavender', 15), ('sandalwood', 25)]
    good_harmony = chemistry.calculate_harmony(good_combo)

    # 안 어울리는 조합
    bad_combo = [('aldehydes', 30), ('patchouli', 30), ('cinnamon', 30)]
    bad_harmony = chemistry.calculate_harmony(bad_combo)

    print(f"   좋은 조합 (Citrus-Aromatic-Woody): {good_harmony:.3f}")
    print(f"   나쁜 조합 (Metallic-Earthy-Spicy): {bad_harmony:.3f}")

    if good_harmony > bad_harmony:
        print(f"\n   [SUCCESS] 조화도 계산이 실제 조향 지식을 반영함!")
        return True
    else:
        print(f"\n   [FAIL] 조화도 계산이 비현실적")
        return False


def main():
    print("\n" + "="*70)
    print("진짜 AI 시스템 종합 검증")
    print("="*70)

    results = {
        'MOGA': test_real_moga(),
        'RLHF': test_real_rlhf(),
        'Chemistry': test_fragrance_chemistry()
    }

    print("\n" + "="*70)
    print("최종 검증 결과")
    print("="*70)

    for system, result in results.items():
        status = "[REAL]" if result else "[FAKE]"
        print(f"{system}: {status}")

    if all(results.values()):
        print("\n[VERIFIED] 모든 시스템이 진짜 AI로 작동합니다!")
        print("- 실제 향수 화학 지식 기반")
        print("- 의미있는 최적화 수행")
        print("- 실제 학습과 개선 발생")
    else:
        print("\n[WARNING] 일부 시스템이 아직 불완전합니다.")


if __name__ == "__main__":
    main()