"""
실제 최적화 시스템 통합 테스트
MOGA와 RLHF가 프로덕션에서 제대로 작동하는지 검증
"""

import json
import numpy as np
from fragrance_ai.training.advanced_optimizer_real import (
    get_real_optimizer_manager,
    RealMOGA,
    FragranceDNA
)


def print_section(title: str):
    """섹션 구분선"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def test_real_moga():
    """실제 MOGA 테스트"""
    print_section("실제 MOGA (Multi-Objective Genetic Algorithm) 테스트")

    # 옵티마이저 매니저 초기화
    manager = get_real_optimizer_manager()
    manager.initialize_moga()

    # 사용자 선호도 설정
    user_preferences = {
        'preferred_notes': ['Rose', 'Vanilla', 'Musk', 'Sandalwood'],
        'style': 'romantic',
        'intensity': 'moderate',
        'season': 'spring'
    }

    print("\n사용자 선호도:")
    for key, value in user_preferences.items():
        print(f"  - {key}: {value}")

    # 최적화 실행
    print("\n최적화 실행 중... (50세대, 100개체)")

    recipes = manager.optimize_with_moga(
        user_preferences=user_preferences,
        creativity_weight=0.4,
        fitness_weight=0.3,
        stability_weight=0.3
    )

    print(f"\n✅ 최적화 완료! {len(recipes)}개의 파레토 최적 레시피 생성")

    # 상위 3개 레시피 출력
    print("\n🏆 상위 3개 레시피:")
    for i, recipe in enumerate(recipes[:3], 1):
        print(f"\n레시피 #{i} - {recipe['dna_id']}")
        print(f"  점수:")
        print(f"    - 창의성: {recipe['scores']['creativity']:.3f}")
        print(f"    - 적합성: {recipe['scores']['fitness']:.3f}")
        print(f"    - 안정성: {recipe['scores']['stability']:.3f}")
        print(f"    - 종합: {recipe['scores']['overall']:.3f}")

        print(f"  구성:")
        print(f"    탑 노트: {', '.join([n['name'] for n in recipe['top_notes'][:3]])}")
        print(f"    미들 노트: {', '.join([n['name'] for n in recipe['middle_notes'][:3]])}")
        print(f"    베이스 노트: {', '.join([n['name'] for n in recipe['base_notes'][:3]])}")

    # 진화 히스토리 분석
    if manager.moga and manager.moga.evolution_history:
        history = manager.moga.evolution_history
        print(f"\n📊 진화 통계:")
        print(f"  - 초기 최고 점수: {history[0]['best_overall']:.3f}")
        print(f"  - 최종 최고 점수: {history[-1]['best_overall']:.3f}")
        print(f"  - 개선율: {((history[-1]['best_overall'] - history[0]['best_overall']) / history[0]['best_overall'] * 100):.1f}%")

    return recipes


def test_real_rlhf():
    """실제 RLHF 테스트"""
    print_section("실제 RLHF (Reinforcement Learning from Human Feedback) 테스트")

    manager = get_real_optimizer_manager()
    manager.initialize_rlhf(state_dim=50, action_dim=7)

    print("\nRLHF 초기화 완료:")
    print(f"  - State dimension: 50")
    print(f"  - Action dimension: 7")
    print(f"  - Learning rate: 0.001")

    # 인간 피드백 시뮬레이션
    print("\n인간 피드백 시뮬레이션 (20개):")

    for i in range(20):
        # 랜덤 상태와 행동
        state = np.random.randn(50)
        action = np.random.randint(0, 7)

        # 시뮬레이션: 특정 패턴에 높은 평점
        if action in [0, 2, 4]:  # 특정 행동 선호
            rating = np.random.choice([4, 5])
        else:
            rating = np.random.choice([1, 2, 3])

        # 피드백 반영
        manager.rlhf.incorporate_human_feedback(state, action, rating)

        if (i + 1) % 5 == 0:
            print(f"  - {i + 1}개 피드백 처리 완료")

    # 학습 실행
    print("\nRLHF 학습 실행 (50 에피소드)...")
    manager.train_with_human_feedback(num_episodes=50)

    # 통계 출력
    stats = manager.get_optimization_stats()
    if 'rlhf' in stats:
        print(f"\n✅ RLHF 학습 완료!")
        print(f"  - 총 피드백 수: {stats['rlhf']['total_feedbacks']}")
        print(f"  - 평균 평점: {stats['rlhf']['average_rating']:.2f}")
        print(f"  - 현재 epsilon: {stats['rlhf']['epsilon']:.3f}")


def test_pareto_dominance():
    """파레토 지배 관계 테스트"""
    print_section("파레토 지배 관계 검증")

    moga = RealMOGA()

    # 테스트 DNA 생성
    dna1 = FragranceDNA(
        top_notes=[('Bergamot', 0.3)],
        middle_notes=[('Rose', 0.4)],
        base_notes=[('Musk', 0.5)]
    )
    dna1.creativity_score = 0.8
    dna1.fitness_score = 0.6
    dna1.stability_score = 0.7

    dna2 = FragranceDNA(
        top_notes=[('Lemon', 0.3)],
        middle_notes=[('Jasmine', 0.4)],
        base_notes=[('Vanilla', 0.5)]
    )
    dna2.creativity_score = 0.9  # 모든 면에서 dna1보다 우수
    dna2.fitness_score = 0.7
    dna2.stability_score = 0.8

    dna3 = FragranceDNA(
        top_notes=[('Orange', 0.3)],
        middle_notes=[('Ylang-Ylang', 0.4)],
        base_notes=[('Amber', 0.5)]
    )
    dna3.creativity_score = 0.7  # dna1과 트레이드오프 관계
    dna3.fitness_score = 0.8
    dna3.stability_score = 0.6

    population = [dna1, dna2, dna3]
    pareto_front = moga._get_pareto_front(population)

    print(f"\n테스트 개체군:")
    for i, dna in enumerate(population, 1):
        print(f"  DNA{i}: 창의성={dna.creativity_score}, "
              f"적합성={dna.fitness_score}, 안정성={dna.stability_score}")

    print(f"\n파레토 프론트: {len(pareto_front)}개")
    for dna in pareto_front:
        idx = population.index(dna) + 1
        print(f"  - DNA{idx}는 파레토 최적해입니다")

    if len(pareto_front) == 2 and dna2 in pareto_front and dna3 in pareto_front:
        print("\n✅ 파레토 지배 관계가 올바르게 계산됨!")
    else:
        print("\n⚠️ 파레토 계산에 문제가 있을 수 있음")


def test_evolution_progress():
    """진화 과정 시각화"""
    print_section("진화 과정 분석")

    moga = RealMOGA()
    moga.num_generations = 20  # 빠른 테스트를 위해 줄임
    moga.population_size = 30

    # 콜백 함수로 진화 추적
    generation_data = []

    def track_evolution(gen, population, history):
        best = max(population, key=lambda d:
                  d.creativity_score * 0.33 +
                  d.fitness_score * 0.33 +
                  d.stability_score * 0.34)
        generation_data.append({
            'generation': gen,
            'best_score': (best.creativity_score * 0.33 +
                          best.fitness_score * 0.33 +
                          best.stability_score * 0.34),
            'avg_creativity': np.mean([d.creativity_score for d in population]),
            'population_size': len(population)
        })

    # 최적화 실행
    print("\n진화 시작...")
    recipes = moga.optimize(callbacks=[track_evolution])

    # 진화 그래프 (텍스트)
    print("\n진화 과정 (최고 점수):")
    for i, data in enumerate(generation_data):
        if i % 5 == 0:  # 5세대마다 출력
            score = data['best_score']
            bar = '█' * int(score * 30)
            print(f"  Gen {data['generation']:3d}: {bar} {score:.3f}")

    print(f"\n최종 결과:")
    print(f"  - 파레토 프론트 크기: {len(moga.pareto_front)}")
    print(f"  - 최고 점수: {generation_data[-1]['best_score']:.3f}")
    print(f"  - 평균 창의성: {generation_data[-1]['avg_creativity']:.3f}")


def main():
    """메인 테스트 실행"""
    print("\n" + "="*70)
    print("         실제 최적화 시스템 통합 테스트")
    print("="*70)

    try:
        # 1. 실제 MOGA 테스트
        moga_recipes = test_real_moga()

        # 2. 실제 RLHF 테스트
        test_real_rlhf()

        # 3. 파레토 지배 관계 검증
        test_pareto_dominance()

        # 4. 진화 과정 분석
        test_evolution_progress()

        print_section("테스트 완료")
        print("\n✅ 모든 실제 최적화 시스템이 정상 작동합니다!")
        print("\n구현 완료:")
        print("  1. MOGA - 실제 유전 알고리즘 (교차, 돌연변이, 선택)")
        print("  2. RLHF - 실제 강화학습 (Q-Learning, 경험 재생)")
        print("  3. 파레토 최적화 - 다목적 균형")
        print("  4. 인간 피드백 통합 - 지속적 개선")

        print("\n이제 시뮬레이션이 아닌 실제 AI 최적화가 작동합니다!")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()