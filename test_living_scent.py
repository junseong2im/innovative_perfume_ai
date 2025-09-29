"""
Living Scent System Test
살아있는 향수 시스템 테스트 스크립트

이 스크립트는 Living Scent 시스템의 핵심 기능을 테스트합니다:
1. 텍스트 입력 처리 (LinguisticReceptor)
2. 감정 해석 (CognitiveCore)
3. DNA 생성 (OlfactoryRecombinator)
4. DNA 진화 (EpigeneticVariation)
"""

import json
from fragrance_ai.orchestrator.living_scent_orchestrator import get_living_scent_orchestrator


def print_section(title: str):
    """섹션 구분선 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_living_scent_creation():
    """Living Scent 생성 테스트"""
    print_section("Living Scent 생성 테스트")

    # 오케스트레이터 초기화 (DB 없이 메모리만 사용)
    orchestrator = get_living_scent_orchestrator(db_session=None)

    # 테스트 케이스 1: 노스탤지어 향수
    test_input_1 = "옛날 할머니 댁 다락방에서 나던 낡고 포근한 느낌의 냄새를 만들어줘"
    print(f"\n테스트 입력 1: {test_input_1}")

    result_1 = orchestrator.process_user_input(
        user_input=test_input_1,
        user_id="test_user_1"
    )

    if result_1['success']:
        print(f"✓ 성공적으로 처리됨")
        print(f"  - 의도: {result_1['intent']}")
        print(f"  - DNA ID: {result_1['result']['dna_id']}")
        print(f"  - 테마: {result_1['metadata']['theme']}")
        print(f"  - 핵심 감정: {result_1['metadata']['core_emotion']}")
        print(f"  - 스토리: {result_1['metadata']['story'][:100]}...")

        # 레시피 출력
        print(f"\n  생성된 레시피:")
        recipe = result_1['result']['recipe']
        for note_type, ingredients in recipe.items():
            if ingredients:
                print(f"    {note_type.upper()}: {', '.join(ingredients[:3])}")

        # DNA ID 저장 (진화 테스트용)
        created_dna_id = result_1['result']['dna_id']
    else:
        print(f"✗ 처리 실패: {result_1.get('message', 'Unknown error')}")
        return

    # 테스트 케이스 2: 로맨틱 향수
    test_input_2 = "첫사랑과 함께 걸었던 봄날의 벚꽃길 향기를 담아줘"
    print(f"\n테스트 입력 2: {test_input_2}")

    result_2 = orchestrator.process_user_input(
        user_input=test_input_2,
        user_id="test_user_2"
    )

    if result_2['success']:
        print(f"✓ 성공적으로 처리됨")
        print(f"  - DNA ID: {result_2['result']['dna_id']}")
        print(f"  - 테마: {result_2['metadata']['theme']}")
        print(f"  - 표현형 잠재력:")
        for key, value in result_2['result']['phenotype_potential'].items():
            print(f"    - {key}: {value:.2f}")

    # 테스트 케이스 3: DNA 진화
    print_section("DNA 진화 테스트")

    evolution_input = "이 향수를 더 스모키하고 강렬하게 만들어줘"
    print(f"\n진화 요청: {evolution_input}")
    print(f"대상 DNA: {created_dna_id}")

    result_3 = orchestrator.process_user_input(
        user_input=evolution_input,
        user_id="test_user_1",
        existing_dna_id=created_dna_id
    )

    if result_3['success']:
        print(f"✓ 성공적으로 진화됨")
        print(f"  - 의도: {result_3['intent']}")
        print(f"  - 표현형 ID: {result_3['result']['phenotype_id']}")
        print(f"  - 기반 DNA: {result_3['result']['based_on_dna']}")
        print(f"  - 설명: {result_3['result']['description'][:150]}...")

        print(f"\n  적용된 후생유전학적 수정:")
        for mod in result_3['result']['modifications']:
            print(f"    - {mod['type']}: {mod['target']} (강도: {mod['factor']})")

        print(f"\n  진화된 레시피:")
        evolved_recipe = result_3['result']['recipe']
        for note_type, ingredients in evolved_recipe.items():
            if ingredients:
                print(f"    {note_type.upper()}: {', '.join(ingredients[:3])}")

    # 테스트 케이스 4: 다중 진화
    print_section("다중 진화 테스트")

    evolution_input_2 = "좀 더 가볍고 상큼하게 바꿔줘"
    print(f"\n추가 진화 요청: {evolution_input_2}")

    result_4 = orchestrator.process_user_input(
        user_input=evolution_input_2,
        user_id="test_user_1",
        existing_dna_id=created_dna_id
    )

    if result_4['success']:
        print(f"✓ 성공적으로 재진화됨")
        print(f"  - 새 표현형 ID: {result_4['result']['phenotype_id']}")
        print(f"  - 환경 반응성:")
        for key, value in result_4['result']['environmental_response'].items():
            print(f"    - {key}: {value:.2f}")

    # 진화 트리 조회
    print_section("진화 트리")

    evolution_tree = orchestrator.get_evolution_tree(created_dna_id)
    print(f"\nDNA {created_dna_id}의 진화 트리:")
    print(f"  - 총 표현형: {len(evolution_tree['nodes'])}개")
    print(f"  - 최대 세대: {evolution_tree['total_generations']}")

    # 통계 출력
    print_section("테스트 요약")

    print(f"\n생성된 DNA 총 개수: {len(orchestrator.memory_dna_library)}")
    print(f"생성된 표현형 총 개수: {len(orchestrator.memory_phenotype_library)}")

    print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")


def test_edge_cases():
    """엣지 케이스 테스트"""
    print_section("엣지 케이스 테스트")

    orchestrator = get_living_scent_orchestrator(db_session=None)

    # 테스트 1: 매우 짧은 입력
    test_input = "향수"
    print(f"\n짧은 입력 테스트: '{test_input}'")
    result = orchestrator.process_user_input(test_input)
    print(f"  결과: {result['success']}")

    # 테스트 2: 영어 입력
    test_input = "Create a fresh morning scent"
    print(f"\n영어 입력 테스트: '{test_input}'")
    result = orchestrator.process_user_input(test_input)
    print(f"  결과: {result['success']}")

    # 테스트 3: 혼합 언어
    test_input = "vintage한 느낌의 classic 향수를 만들어줘"
    print(f"\n혼합 언어 테스트: '{test_input}'")
    result = orchestrator.process_user_input(test_input)
    print(f"  결과: {result['success']}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("         Living Scent System Test Suite")
    print("         살아있는 향수 시스템 테스트")
    print("="*60)

    try:
        # 메인 테스트 실행
        test_living_scent_creation()

        # 엣지 케이스 테스트
        test_edge_cases()

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("         테스트 완료")
    print("="*60)