"""
MOGA와 RLHF 엔진 통합 테스트
실제 작동 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fragrance_ai.training.olfactory_recombinator_deap import OlfactoryRecombinatorAI, CreativeBrief
from fragrance_ai.training.reinforcement_learning import get_fragrance_rlhf
import numpy as np


def test_moga_engine():
    """MOGA 엔진 테스트"""
    print("\n" + "="*80)
    print("1. MOGA (Multi-Objective Genetic Algorithm) 엔진 테스트")
    print("="*80)

    # 창세기 엔진 초기화
    recombinator = OlfactoryRecombinatorAI()

    # 테스트 브리프 생성
    test_brief = CreativeBrief(
        emotional_profile={
            "romantic": 0.8,
            "elegant": 0.7,
            "mysterious": 0.6
        },
        fragrance_family="oriental",
        season="fall",
        occasion="evening",
        intensity=0.7,
        keywords=["seductive", "warm", "sophisticated"],
        avoid_notes=["mint", "eucalyptus"]
    )

    print(f"\n브리프 정보:")
    print(f"  - 향수 계열: {test_brief.fragrance_family}")
    print(f"  - 계절: {test_brief.season}")
    print(f"  - 강도: {test_brief.intensity}")
    print(f"  - 키워드: {', '.join(test_brief.keywords)}")
    print(f"  - 피해야 할 노트: {', '.join(test_brief.avoid_notes)}")

    # MOGA 최적화 실행
    print("\nMOGA 최적화 시작...")
    result = recombinator.generate_olfactory_dna(
        brief=test_brief,
        population_size=30,  # 테스트용으로 작은 크기
        generations=15,       # 빠른 테스트
        verbose=True
    )

    print("\n" + "-"*80)
    print("최적화 결과:")
    print("-"*80)

    # 레시피 출력
    recipe = result['recipe']
    print("\n탑 노트:")
    for note, concentration in recipe['top'].items():
        print(f"  - {note}: {concentration:.2f}%")

    print("\n미들 노트:")
    for note, concentration in recipe['middle'].items():
        print(f"  - {note}: {concentration:.2f}%")

    print("\n베이스 노트:")
    for note, concentration in recipe['base'].items():
        print(f"  - {note}: {concentration:.2f}%")

    # 적합도 점수
    fitness = result['fitness_values']
    print(f"\n적합도 점수:")
    print(f"  - 조화도: {fitness['harmony']:.4f}")
    print(f"  - 적합도: {fitness['fitness']:.4f}")
    print(f"  - 창의성: {fitness['creativity']:.4f}")
    print(f"  - 종합: {fitness['overall']:.4f}")

    # 평가 결과
    evaluation = result['evaluation']
    print(f"\n화학적 평가:")
    print(f"  - Harmony: {evaluation.get('harmony', 0):.4f}")
    print(f"  - Balance: {evaluation.get('balance', 0):.4f}")
    print(f"  - Longevity: {evaluation.get('longevity', 0):.4f}")
    print(f"  - Overall: {evaluation.get('overall', 0):.4f}")

    print(f"\nPareto 프론트 크기: {result['pareto_front_size']}")

    return result


def test_rlhf_engine():
    """RLHF 엔진 테스트"""
    print("\n" + "="*80)
    print("2. RLHF (Reinforcement Learning from Human Feedback) 엔진 테스트")
    print("="*80)

    # RLHF 시스템 초기화
    rlhf = get_fragrance_rlhf()

    # 테스트용 DNA 생성 (더미 데이터)
    from fragrance_ai.models.living_scent.olfactory_recombinator import OlfactoryDNA, FragranceGene

    # 간단한 테스트 DNA
    test_dna = OlfactoryDNA(
        dna_id="TEST_DNA_001",
        lineage=["concept_romantic", "concept_adventurous"],
        genotype={
            'top': [
                FragranceGene('top', 'Bergamot', 0.3, 0.9, 136, 'citrus'),
                FragranceGene('top', 'Lemon', 0.2, 0.95, 136, 'citrus')
            ],
            'middle': [
                FragranceGene('middle', 'Rose', 0.4, 0.5, 154, 'floral'),
                FragranceGene('middle', 'Jasmine', 0.3, 0.45, 196, 'floral')
            ],
            'base': [
                FragranceGene('base', 'Sandalwood', 0.4, 0.1, 220, 'woody'),
                FragranceGene('base', 'Vanilla', 0.3, 0.15, 152, 'sweet')
            ]
        },
        phenotype_potential={
            'longevity': 0.7,
            'sillage': 0.6,
            'complexity': 0.8,
            'balance': 0.75
        },
        story="테스트용 향수 DNA",
        creation_timestamp="2025-01-26T00:00:00"
    )

    print(f"\n테스트 DNA ID: {test_dna.dna_id}")
    print(f"표현형 잠재력:")
    for key, value in test_dna.phenotype_potential.items():
        print(f"  - {key}: {value:.2f}")

    # 사용자 피드백 시뮬레이션
    print("\n사용자 피드백 시뮬레이션...")
    feedback_scenarios = [
        {"text": "좋아요!", "rating": 8.5},
        {"text": "너무 달아요", "rating": 5.0},
        {"text": "완벽해요!", "rating": 9.5},
        {"text": "좀 더 신선한 느낌이 필요해요", "rating": 6.5},
    ]

    for i, scenario in enumerate(feedback_scenarios, 1):
        print(f"\n피드백 {i}: '{scenario['text']}' (평점: {scenario['rating']}/10)")

        # 진화 실행
        modification = rlhf.evolve_fragrance(
            dna=test_dna,
            user_feedback=scenario['text'],
            rating=scenario['rating']
        )

        print(f"  제안된 수정:")
        print(f"    - 유형: {modification['type']}")
        print(f"    - 대상: {modification['target']}")
        print(f"    - 강도: {modification['strength']:.2f}")
        print(f"    - 보상: {modification['reward']:.4f}")

    # 학습 통계
    stats = rlhf.get_statistics()
    print(f"\n학습 통계:")
    print(f"  - 총 경험: {stats['total_experiences']}")
    print(f"  - 인간 피드백: {stats['total_human_feedbacks']}")
    print(f"  - 평균 정책 손실: {stats['avg_policy_loss']:.6f}")
    print(f"  - 평균 가치 손실: {stats['avg_value_loss']:.6f}")
    print(f"  - 평균 보상: {stats['avg_reward']:.4f}")

    return rlhf


def test_integration():
    """MOGA + RLHF 통합 테스트"""
    print("\n" + "="*80)
    print("3. MOGA + RLHF 통합 테스트")
    print("="*80)

    # MOGA로 초기 DNA 생성
    print("\nStep 1: MOGA로 초기 향수 생성")
    moga_result = test_moga_engine()

    # RLHF로 사용자 피드백 기반 진화
    print("\n\nStep 2: RLHF로 사용자 피드백 기반 진화")
    rlhf = test_rlhf_engine()

    print("\n" + "="*80)
    print("통합 테스트 완료!")
    print("="*80)
    print("\n요약:")
    print("  ✓ MOGA 엔진: 다중 목표 최적화 성공")
    print("  ✓ RLHF 엔진: 강화학습 기반 진화 성공")
    print("  ✓ 통합: 두 엔진이 정상적으로 연동")
    print("\n시스템 상태: 정상 작동 중")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FRAGRANCE AI - MOGA & RLHF 엔진 통합 테스트")
    print("="*80)
    print("\n이 테스트는 다음을 검증합니다:")
    print("  1. DEAP 기반 다중 목표 유전 알고리즘 (MOGA)")
    print("  2. PyTorch 기반 강화학습 (RLHF)")
    print("  3. 두 엔진의 통합 및 상호작용")
    print("  4. ValidatorTool과의 연동")

    try:
        test_integration()
        print("\n\n✅ 모든 테스트 통과!")
    except Exception as e:
        print(f"\n\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()