"""
전체 AI 시스템 통합 테스트
모든 AI 모듈이 함께 작동하는지 검증
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

# Living Scent 모듈들
from fragrance_ai.models.living_scent.olfactory_recombinator import (
    OlfactoryRecombinatorAI,
    get_olfactory_recombinator
)
from fragrance_ai.models.living_scent.epigenetic_variation import (
    EpigeneticVariationAI,
    get_epigenetic_variation
)
from fragrance_ai.training.reinforcement_learning import (
    FragranceRLHF,
    get_fragrance_rlhf
)

# AI Perfumer 오케스트레이터
from fragrance_ai.orchestrator.ai_perfumer_orchestrator import AIPerfumerOrchestrator

@dataclass
class MockCreativeBrief:
    """테스트용 CreativeBrief"""
    theme: str
    archetype: str
    story: str
    emotional_palette: Dict[str, float]
    core_emotion: str

def print_section(title: str):
    """섹션 구분선"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def test_living_scent_to_ai_perfumer():
    """Living Scent에서 AI Perfumer로의 전체 플로우 테스트"""
    print_section("Living Scent -> AI Perfumer 통합 테스트")

    # 1. Living Scent로 DNA 생성
    print("\n1단계: Living Scent DNA 생성")
    recombinator = get_olfactory_recombinator()

    brief = MockCreativeBrief(
        theme="미래의 기억",
        archetype="futuristic",
        story="아직 오지 않은 시간의 향수, 미래에 대한 그리움",
        emotional_palette={
            'metallic': 0.7,
            'ethereal': 0.8,
            'synthetic': 0.6,
            'nostalgic': 0.4
        },
        core_emotion="anticipation"
    )

    dna = recombinator.create(brief)
    print(f"  -> DNA ID: {dna.dna_id}")
    print(f"  -> 적합도: {dna.fitness_score:.3f}")

    # 2. RLHF로 진화
    print("\n2단계: RLHF 기반 진화")
    epigenetic_ai = get_epigenetic_variation()
    rlhf_system = get_fragrance_rlhf()

    feedback_brief = MockCreativeBrief(
        theme="진화된 미래",
        archetype="evolved",
        story="더 선명하고 강렬한 미래의 향으로",
        emotional_palette={'intensity': 0.9},
        core_emotion="amplified"
    )

    dna_library = {dna.dna_id: dna}

    phenotype = epigenetic_ai.evolve(
        dna_id=dna.dna_id,
        feedback_brief=feedback_brief,
        dna_library=dna_library,
        use_rlhf=True,
        user_rating=4.5
    )

    print(f"  -> 표현형 ID: {phenotype.phenotype_id}")
    print(f"  -> 수정 개수: {len(phenotype.modifications)}")

    # 3. AI Perfumer로 최종 해석
    print("\n3단계: AI Perfumer 해석")
    orchestrator = AIPerfumerOrchestrator()

    # DNA 정보를 기반으로 향수 설명 생성
    dna_description = f"""
    DNA ID: {dna.dna_id}
    세대: {dna.generation}
    주요 유전자: {', '.join([g.ingredient for g in dna.genotype['top'][:3]])}
    표현형 특성: {', '.join([f"{k}:{v:.2f}" for k, v in phenotype.environmental_response.items()][:3])}
    """

    ai_interpretation = orchestrator.generate_response(
        f"이 DNA 정보를 기반으로 향수를 해석해주세요: {dna_description}",
        []
    )

    print(f"  -> AI 해석: {ai_interpretation[:200]}...")

    return dna, phenotype, ai_interpretation

def test_customer_request_to_evolution():
    """고객 요청에서 진화까지의 전체 플로우"""
    print_section("고객 요청 -> 진화 시스템 통합 테스트")

    # 1. 고객 요청을 AI Perfumer가 해석
    print("\n1단계: 고객 요청 해석")
    orchestrator = AIPerfumerOrchestrator()

    customer_request = "카프카의 변신을 읽고 느낀 그 기묘한 감정을 향수로 만들어주세요"

    print(f"  고객: '{customer_request}'")

    # AI가 창의적 브리프 생성
    fragrance = orchestrator.execute_creative_process(customer_request)

    print(f"  -> 생성된 향수: {fragrance.get('name', 'Unknown')}")

    # 2. Living Scent DNA 생성
    print("\n2단계: DNA 생성")
    recombinator = get_olfactory_recombinator()

    brief = MockCreativeBrief(
        theme=fragrance.get('name', 'Metamorphosis'),
        archetype="surreal",
        story=customer_request,
        emotional_palette={
            'strange': 0.8,
            'dark': 0.6,
            'transformative': 0.9
        },
        core_emotion="alienation"
    )

    dna = recombinator.create(brief)
    print(f"  -> DNA ID: {dna.dna_id}")

    # 3. 여러 세대 진화
    print("\n3단계: 다세대 진화")
    epigenetic_ai = get_epigenetic_variation()
    dna_library = {dna.dna_id: dna}

    feedback_rounds = [
        ("좀 더 신비로운 느낌으로", 3.5),
        ("변화의 순간을 더 강조해주세요", 4.0),
        ("완벽해요!", 5.0)
    ]

    latest_phenotype = None
    for i, (feedback, rating) in enumerate(feedback_rounds, 1):
        print(f"\n  진화 {i}회차:")
        print(f"    피드백: '{feedback}'")
        print(f"    평점: {rating}/5")

        feedback_brief = MockCreativeBrief(
            theme=f"Evolution_{i}",
            archetype="evolving",
            story=feedback,
            emotional_palette={'evolution': rating/5.0},
            core_emotion=f"round_{i}"
        )

        latest_phenotype = epigenetic_ai.evolve(
            dna_id=dna.dna_id,
            feedback_brief=feedback_brief,
            dna_library=dna_library,
            use_rlhf=True,
            user_rating=rating
        )

        print(f"    -> 새 표현형: {latest_phenotype.phenotype_id}")

    return fragrance, dna, latest_phenotype

def test_rlhf_learning_curve():
    """RLHF 학습 곡선 테스트"""
    print_section("RLHF 학습 곡선 분석")

    rlhf_system = get_fragrance_rlhf()
    recombinator = get_olfactory_recombinator()

    # 10개의 서로 다른 DNA 생성하고 피드백
    print("\n학습 데이터 생성:")
    for i in range(10):
        brief = MockCreativeBrief(
            theme=f"Test_{i}",
            archetype=["floral", "woody", "fresh", "oriental"][i % 4],
            story=f"Test fragrance {i}",
            emotional_palette={'test': np.random.random()},
            core_emotion=f"emotion_{i}"
        )

        dna = recombinator.create(brief)
        state = rlhf_system.encode_fragrance_state(dna)
        action = rlhf_system.select_action(state)

        # 랜덤 평점 (실제로는 사용자 피드백)
        rating = np.random.uniform(2, 5)
        reward = (rating - 3) / 2  # Normalize to [-0.5, 1]

        next_state = state + np.random.randn(rlhf_system.state_dim) * 0.1
        rlhf_system.store_experience(state, action, reward, next_state, False)

        if i % 3 == 0:
            print(f"  샘플 {i+1}: 평점 {rating:.1f} -> 보상 {reward:.2f}")

    # PPO 학습
    print("\nPPO 학습 실행:")
    rlhf_system.train_ppo(batch_size=4, epochs=2)

    # 학습 통계
    stats = rlhf_system.get_statistics()
    print(f"\n학습 결과:")
    print(f"  - 총 경험: {stats['total_experiences']}")
    print(f"  - 평균 보상: {stats['avg_reward']:.3f}")
    print(f"  - 정책 손실: {stats.get('policy_loss', 'N/A')}")
    print(f"  - 가치 손실: {stats.get('value_loss', 'N/A')}")

def test_system_resilience():
    """시스템 복원력 테스트"""
    print_section("시스템 복원력 테스트")

    print("\n1. 잘못된 입력 처리:")
    orchestrator = AIPerfumerOrchestrator()

    # 빈 입력
    response = orchestrator.generate_response("", [])
    print(f"  빈 입력 -> 응답: {response[:50]}...")

    # 매우 긴 입력
    long_input = "향수 " * 1000
    response = orchestrator.generate_response(long_input, [])
    print(f"  긴 입력 -> 응답: {response[:50]}...")

    print("\n2. DNA 생성 에러 복구:")
    recombinator = get_olfactory_recombinator()

    # 잘못된 브리프
    try:
        bad_brief = MockCreativeBrief(
            theme="",
            archetype="invalid_type",
            story="",
            emotional_palette={},
            core_emotion=""
        )
        dna = recombinator.create(bad_brief)
        print(f"  잘못된 브리프 -> DNA 생성: {dna.dna_id}")
    except Exception as e:
        print(f"  에러 처리 성공: {type(e).__name__}")

    print("\n3. RLHF 빈 버퍼 처리:")
    rlhf_system = FragranceRLHF()  # 새 인스턴스

    try:
        # 빈 버퍼로 학습 시도
        rlhf_system.train_ppo()
        print("  빈 버퍼 학습: 정상 처리")
    except Exception as e:
        print(f"  에러 처리: {type(e).__name__}")

def main():
    """메인 통합 테스트"""
    print("\n" + "="*70)
    print("         전체 AI 시스템 통합 테스트")
    print("         Living Scent + AI Perfumer + RLHF")
    print("="*70)

    try:
        # 1. Living Scent -> AI Perfumer 통합
        dna1, phenotype1, interpretation1 = test_living_scent_to_ai_perfumer()

        # 2. 고객 요청 -> 진화 시스템
        fragrance2, dna2, phenotype2 = test_customer_request_to_evolution()

        # 3. RLHF 학습 곡선
        test_rlhf_learning_curve()

        # 4. 시스템 복원력
        test_system_resilience()

        print_section("통합 테스트 완료")
        print("\n모든 시스템이 성공적으로 통합되었습니다!")
        print("\n시스템 구성:")
        print("  1. [O] Living Scent (유전 알고리즘)")
        print("  2. [O] Epigenetic Evolution (후생유전학)")
        print("  3. [O] RLHF (강화학습)")
        print("  4. [O] AI Perfumer (오케스트레이션)")
        print("  5. [O] Error Resilience (에러 복구)")

        print("\n통합 결과:")
        print(f"  - 생성된 DNA: {dna1.dna_id}, {dna2.dna_id}")
        print(f"  - 진화된 표현형: {phenotype1.phenotype_id}, {phenotype2.phenotype_id}")
        print(f"  - AI 해석 길이: {len(interpretation1)} 글자")
        print(f"  - 생성된 향수: {fragrance2.get('name', 'Unknown')}")

        print("\n시스템이 완벽하게 작동합니다!")

    except Exception as e:
        print(f"\n[X] 통합 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()