"""
AI Perfumer 직접 테스트 (서버 없이)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fragrance_ai.orchestrator.ai_perfumer_orchestrator import AIPerfumerOrchestrator

def print_section(title: str):
    """섹션 구분선"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def test_basic_conversation():
    """기본 대화 테스트"""
    print_section("AI Perfumer 기본 대화 테스트")

    orchestrator = AIPerfumerOrchestrator()

    # 테스트 메시지
    messages = [
        "나는 누구인지 모르겠어요. 정체성에 대한 향을 만들어주세요.",
        "시간이 멈춘 것 같은 고독한 순간을 표현하고 싶어요.",
        "카프카의 변신처럼, 나도 완전히 다른 존재가 되고 싶어요."
    ]

    context = []
    for message in messages:
        print(f"\n사용자: {message}")
        response = orchestrator.generate_response(message, context)
        print(f"AI: {response}")
        context.append(message)

def test_fragrance_creation():
    """향수 생성 테스트"""
    print_section("향수 생성 테스트")

    orchestrator = AIPerfumerOrchestrator()

    # 컨텍스트 준비
    full_text = "정체성과 시간, 그리고 고독을 담은 향수를 만들어주세요"

    print(f"입력: {full_text}")

    # 향수 생성
    fragrance = orchestrator.execute_creative_process(full_text)

    print(f"\n생성된 향수:")
    print(f"  이름: {fragrance.get('name', 'Unknown')}")
    print(f"  한국어 이름: {fragrance.get('korean_name', '이름 없음')}")

    if fragrance.get('story'):
        print(f"\n  스토리:")
        for line in fragrance['story'].split('\n'):
            if line.strip():
                print(f"    {line.strip()}")

    if fragrance.get('composition'):
        print(f"\n  구성:")
        comp = fragrance['composition']

        if comp.get('top_notes'):
            print("    탑 노트:")
            for note in comp['top_notes']:
                print(f"      - {note.get('name', 'Unknown')}: {note.get('description', '')}")

        if comp.get('heart_notes'):
            print("    하트 노트:")
            for note in comp['heart_notes']:
                print(f"      - {note.get('name', 'Unknown')}: {note.get('description', '')}")

        if comp.get('base_notes'):
            print("    베이스 노트:")
            for note in comp['base_notes']:
                print(f"      - {note.get('name', 'Unknown')}: {note.get('description', '')}")

    if fragrance.get('characteristics'):
        print(f"\n  특성:")
        chars = fragrance['characteristics']
        print(f"    - 강도: {chars.get('intensity', 'Unknown')}")
        print(f"    - 지속시간: {chars.get('longevity', 'Unknown')}")
        print(f"    - 확산력: {chars.get('sillage', 'Unknown')}")
        print(f"    - 계절: {chars.get('season', 'Unknown')}")
        print(f"    - 성별: {chars.get('gender', 'Unknown')}")

def test_synesthetic_translation():
    """공감각적 번역 테스트"""
    print_section("공감각적 번역 테스트")

    orchestrator = AIPerfumerOrchestrator()

    # 다양한 개념 테스트
    concepts = [
        "빨간색 화요일의 소리",
        "차가운 음악의 온도",
        "달콤한 침묵의 질감",
        "시간의 색깔과 공간의 향기"
    ]

    for concept in concepts:
        print(f"\n개념: '{concept}'")
        response = orchestrator.generate_response(concept, [])
        print(f"AI 해석: {response[:200]}...")

def test_emotional_journey():
    """감정 여정 테스트"""
    print_section("감정 여정 향수 테스트")

    orchestrator = AIPerfumerOrchestrator()

    # 감정의 여정
    journey = "설렘에서 시작해 불안을 거쳐 고독으로 빠졌다가 마침내 평화를 찾는 여정"

    print(f"감정 여정: {journey}")

    # 향수 생성
    fragrance = orchestrator.execute_creative_process(journey)

    print(f"\n생성된 향수: {fragrance.get('name', 'Unknown')}")
    print(f"주요 테마: {fragrance.get('creation_context', {}).get('dominant_theme', 'Unknown')}")

def main():
    """메인 테스트 실행"""
    print("\n" + "="*70)
    print("         AI Perfumer 직접 테스트")
    print("         서버 없이 모듈 직접 호출")
    print("="*70)

    try:
        # 1. 기본 대화 테스트
        test_basic_conversation()

        # 2. 향수 생성 테스트
        test_fragrance_creation()

        # 3. 공감각적 번역 테스트
        test_synesthetic_translation()

        # 4. 감정 여정 테스트
        test_emotional_journey()

        print_section("테스트 완료")
        print("\nAI Perfumer 시스템이 정상적으로 작동합니다!")
        print("세상의 모든 개념을 향으로 번역하는 예술가가 준비되었습니다.")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()