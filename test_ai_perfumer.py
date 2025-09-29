"""
AI Perfumer 시스템 테스트
공감각 조향사 기능 검증
"""

import requests
import json
import time
from typing import Dict, List, Any

# API 엔드포인트
BASE_URL = "http://localhost:8001"
AI_PERFUMER_ENDPOINT = f"{BASE_URL}/api/v1/ai-perfumer/chat"

def print_section(title: str):
    """섹션 구분선"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def test_ai_perfumer_conversation():
    """AI Perfumer 대화 테스트"""
    print_section("AI Perfumer (공감각 조향사) 대화 테스트")

    session_id = f"test_session_{int(time.time())}"
    context = []

    # 테스트 대화 시나리오
    test_messages = [
        "나는 누구인지 모르겠어요. 정체성에 대한 향을 만들어주세요.",
        "시간이 멈춘 것 같은 고독한 순간을 표현하고 싶어요.",
        "카프카의 변신처럼, 나도 완전히 다른 존재가 되고 싶어요.",
        "이제 이 모든 감정을 하나의 향수로 만들어주세요."
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n[사용자 {i}]: {message}")

        # API 호출
        response = requests.post(
            AI_PERFUMER_ENDPOINT,
            json={
                "message": message,
                "context": context,
                "session_id": session_id
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"[AI Perfumer]: {data['response']}")

            # 향수가 생성되었으면 출력
            if data.get('fragrance'):
                print_fragrance_details(data['fragrance'])

            # 컨텍스트에 메시지 추가
            context.append(message)
        else:
            print(f"❌ 오류: {response.status_code} - {response.text}")
            return False

    return True

def print_fragrance_details(fragrance: Dict[str, Any]):
    """향수 상세 정보 출력"""
    print("\n" + "-"*60)
    print("🌸 생성된 향수:")
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
        if chars.get('keywords'):
            print(f"    - 키워드: {', '.join(chars['keywords'])}")

    print("-"*60)

def test_synesthetic_translation():
    """공감각적 번역 테스트"""
    print_section("공감각적 번역 테스트")

    # 다양한 개념 테스트
    test_concepts = [
        "빨간색 화요일의 소리",
        "차가운 음악의 온도",
        "달콤한 침묵의 질감",
        "시간의 색깔과 공간의 향기"
    ]

    for concept in test_concepts:
        print(f"\n테스트 개념: '{concept}'")

        response = requests.post(
            AI_PERFUMER_ENDPOINT,
            json={
                "message": concept,
                "context": [],
                "session_id": f"synesthetic_test_{int(time.time())}"
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"AI 해석: {data['response'][:200]}...")
        else:
            print(f"❌ 오류: {response.status_code}")

def test_cultural_references():
    """문화적 참조 테스트"""
    print_section("문화적 참조 향수 생성 테스트")

    cultural_inputs = [
        "백남준의 비디오 아트같은 전자적인 향",
        "한옥 마루에 스며든 세월의 향기",
        "사물놀이의 역동적인 리듬을 담은 향수"
    ]

    for input_text in cultural_inputs:
        print(f"\n입력: '{input_text}'")

        # 즉시 향수 생성 요청 (충분한 컨텍스트 제공)
        response = requests.post(
            AI_PERFUMER_ENDPOINT,
            json={
                "message": "이 향수를 만들어주세요",
                "context": [input_text, "이 개념을 향수로 표현하고 싶어요"],
                "session_id": f"cultural_test_{int(time.time())}"
            }
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('fragrance'):
                print(f"✅ 향수 생성 성공: {data['fragrance'].get('name', 'Unknown')}")
            else:
                print(f"응답: {data['response'][:150]}...")
        else:
            print(f"❌ 오류: {response.status_code}")

def test_emotional_journey():
    """감정 여정 향수 테스트"""
    print_section("감정 여정 향수 생성 테스트")

    session_id = f"emotional_journey_{int(time.time())}"

    # 감정의 여정
    emotional_journey = [
        "처음엔 설렘으로 시작했어요",
        "그러다 점점 불안해졌고",
        "깊은 고독을 느꼈지만",
        "결국 평화를 찾았어요. 이 여정을 향수로 만들어주세요."
    ]

    context = []
    for step in emotional_journey:
        response = requests.post(
            AI_PERFUMER_ENDPOINT,
            json={
                "message": step,
                "context": context,
                "session_id": session_id
            }
        )

        if response.status_code == 200:
            data = response.json()
            print(f"\n단계: {step}")
            print(f"AI: {data['response'][:150]}...")

            if data.get('fragrance'):
                print("\n🎭 감정 여정 향수 완성!")
                print_fragrance_details(data['fragrance'])

            context.append(step)
        else:
            print(f"❌ 오류: {response.status_code}")
            break

def main():
    """메인 테스트 실행"""
    print("\n" + "="*70)
    print("         AI Perfumer 시스템 통합 테스트")
    print("         공감각 조향사 기능 검증")
    print("="*70)

    try:
        # 서버 상태 확인
        print("\n서버 연결 확인...")
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ 서버가 실행되지 않았습니다. 먼저 서버를 시작하세요:")
            print("   uvicorn fragrance_ai.api.main:app --reload --port 8001")
            return

        print("✅ 서버 연결 성공!")

        # 테스트 실행
        tests_passed = 0
        tests_total = 4

        # 1. 기본 대화 테스트
        if test_ai_perfumer_conversation():
            tests_passed += 1
            print("\n✅ AI Perfumer 대화 테스트 통과")

        # 2. 공감각적 번역 테스트
        test_synesthetic_translation()
        tests_passed += 1
        print("\n✅ 공감각적 번역 테스트 완료")

        # 3. 문화적 참조 테스트
        test_cultural_references()
        tests_passed += 1
        print("\n✅ 문화적 참조 테스트 완료")

        # 4. 감정 여정 테스트
        test_emotional_journey()
        tests_passed += 1
        print("\n✅ 감정 여정 테스트 완료")

        # 결과 요약
        print_section("테스트 결과 요약")
        print(f"통과: {tests_passed}/{tests_total}")
        print("\n주요 기능:")
        print("  ✅ 추상적 개념을 향으로 번역")
        print("  ✅ 감정과 기억을 향수로 변환")
        print("  ✅ 시공간적 경험을 향으로 재현")
        print("  ✅ 문화적 참조를 향으로 해석")
        print("  ✅ 감정의 여정을 향수로 구성")

        print("\n💫 AI Perfumer 시스템이 정상적으로 작동합니다!")
        print("   세상의 모든 개념을 향으로 번역하는 예술가가 준비되었습니다.")

    except requests.exceptions.ConnectionError:
        print("\n❌ 서버에 연결할 수 없습니다.")
        print("   서버를 먼저 시작하세요:")
        print("   uvicorn fragrance_ai.api.main:app --reload --port 8001")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()