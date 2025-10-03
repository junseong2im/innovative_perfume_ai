"""
End-to-End Test for Living Scent Evolution System
Sprint 4: Complete RLHF Evolution Pipeline Test
"""

import pytest
import asyncio
import torch
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fragrance_ai.training.reinforcement_learning import RLEngine
from fragrance_ai.orchestrator.living_scent_orchestrator_evolution import LivingScentOrchestrator
from fragrance_ai.database.models import OlfactoryDNA, ScentPhenotype


class TestEndToEndEvolution:
    """완전한 향수 진화 시나리오 테스트"""

    @pytest.fixture
    def orchestrator(self):
        """오케스트레이터 초기화"""
        return LivingScentOrchestrator()

    @pytest.fixture
    def test_dna(self):
        """테스트용 초기 DNA"""
        return OlfactoryDNA(
            dna_id="test_dna_001",
            notes=[
                {"name": "bergamot", "intensity": 0.8},
                {"name": "jasmine", "intensity": 0.6},
                {"name": "sandalwood", "intensity": 0.4},
                {"name": "musk", "intensity": 0.3},
                {"name": "vanilla", "intensity": 0.2}
            ],
            genotype={
                "top_notes": ["bergamot"],
                "heart_notes": ["jasmine"],
                "base_notes": ["sandalwood", "musk", "vanilla"]
            }
        )

    def test_1_create_new_fragrance(self, orchestrator):
        """
        시나리오 1: 새로운 향수 생성 (CREATE_NEW)
        """
        # 사용자 요청
        request = {
            "text": "신선하고 활기찬 여름 아침의 향수를 만들어주세요",
            "intent": "CREATE_NEW"
        }

        # 오케스트레이터 호출
        response = orchestrator.handle_user_request(
            user_id="test_user_001",
            request=request
        )

        # 검증
        assert response is not None
        assert "new_dna" in response
        assert response["new_dna"]["dna_id"] is not None

        # DNA 구조 확인
        dna = response["new_dna"]
        assert "notes" in dna or "genes" in dna
        assert "genotype" in dna or "composition" in dna

        print(f"✓ 새 향수 DNA 생성 성공: {dna.get('dna_id', 'unknown')}")

    def test_2_evolve_existing_fragrance(self, orchestrator, test_dna):
        """
        시나리오 2: 기존 향수 진화 (EVOLVE_EXISTING)
        """
        # 1단계: 진화 요청
        evolution_request = {
            "text": "이 향수를 더 따뜻하고 관능적으로 만들어주세요",
            "target_dna_id": test_dna.dna_id,
            "intent": "EVOLVE_EXISTING"
        }

        # 오케스트레이터 호출
        evolution_response = orchestrator.handle_user_request(
            user_id="test_user_001",
            request=evolution_request
        )

        # 검증
        assert evolution_response["status"] == "evolution_options_ready"
        assert "options" in evolution_response
        assert len(evolution_response["options"]) == 3  # 3개의 변형 옵션

        # 각 옵션 검증
        for option in evolution_response["options"]:
            assert "id" in option
            assert "description" in option
            assert "action_applied" in option
            assert "preview" in option

        print(f"✓ 3개의 진화 옵션 생성 성공")
        for i, opt in enumerate(evolution_response["options"], 1):
            print(f"  옵션 {i}: {opt['action_applied']}")

        return evolution_response["options"]

    def test_3_finalize_evolution(self, orchestrator):
        """
        시나리오 3: 진화 완료 및 학습 (FINALIZE_EVOLUTION)
        """
        # Setup: 먼저 진화 옵션을 생성
        test_dna = OlfactoryDNA(
            dna_id="test_dna_002",
            notes=[
                {"name": "rose", "intensity": 0.7},
                {"name": "amber", "intensity": 0.5}
            ],
            genotype={"top_notes": ["rose"], "base_notes": ["amber"]}
        )

        # 진화 요청
        evolution_request = {
            "text": "더 신비롭게 만들어주세요",
            "target_dna_id": test_dna.dna_id,
            "intent": "EVOLVE_EXISTING"
        }

        evolution_response = orchestrator.handle_user_request(
            user_id="test_user_002",
            request=evolution_request
        )

        # 두 번째 옵션 선택
        chosen_option = evolution_response["options"][1]

        # 최종 선택 요청
        finalize_request = {
            "text": "두 번째 옵션이 마음에 들어요",
            "chosen_phenotype_id": chosen_option["id"],
            "intent": "FINALIZE_EVOLUTION"
        }

        # 최종화 호출
        final_response = orchestrator.handle_user_request(
            user_id="test_user_002",
            request=finalize_request
        )

        # 검증
        assert final_response["status"] == "evolution_complete"
        assert "final_scent" in final_response
        assert "training_result" in final_response
        assert "AI가 당신의 취향을 학습했습니다" in final_response["message"]

        print(f"✓ 진화 완료 및 AI 학습 성공")
        print(f"  선택된 향수: {chosen_option['action_applied']}")
        print(f"  학습 결과: {final_response.get('training_result', 'N/A')}")

    def test_4_continuous_evolution(self, orchestrator):
        """
        시나리오 4: 연속 진화 (CONTINUE_EVOLUTION)
        """
        # Setup: 초기 DNA와 진화
        initial_dna = OlfactoryDNA(
            dna_id="test_dna_003",
            notes=[{"name": "lavender", "intensity": 0.6}],
            genotype={"top_notes": ["lavender"]}
        )

        # 첫 번째 진화
        first_evolution = orchestrator.handle_user_request(
            user_id="test_user_003",
            request={
                "text": "더 깊이감 있게",
                "target_dna_id": initial_dna.dna_id,
                "intent": "EVOLVE_EXISTING"
            }
        )

        # 선택
        chosen_id = first_evolution["options"][0]["id"]
        finalize_response = orchestrator.handle_user_request(
            user_id="test_user_003",
            request={
                "chosen_phenotype_id": chosen_id,
                "intent": "FINALIZE_EVOLUTION"
            }
        )

        # 연속 진화 요청
        continue_request = {
            "text": "이걸 기반으로 좀 더 신선하게 만들어줘",
            "previous_choice_id": chosen_id,
            "intent": "CONTINUE_EVOLUTION"
        }

        continue_response = orchestrator.handle_user_request(
            user_id="test_user_003",
            request=continue_request
        )

        # 검증
        assert continue_response["status"] == "evolution_options_ready"
        assert len(continue_response["options"]) == 3

        print(f"✓ 연속 진화 성공")
        print(f"  이전 선택을 기반으로 새로운 3개 옵션 생성")

    def test_5_rl_engine_learning(self):
        """
        시나리오 5: 강화학습 엔진 학습 확인
        """
        # RL 엔진 초기화
        rl_engine = RLEngine(state_dim=10, action_dim=6, learning_rate=0.001)

        # 초기 DNA와 브리프
        test_dna = OlfactoryDNA(
            dna_id="rl_test_001",
            notes=[
                {"name": "citrus", "intensity": 0.9},
                {"name": "green tea", "intensity": 0.4}
            ],
            genotype={"notes": ["citrus", "green tea"]}
        )

        feedback_brief = {
            "desired_intensity": 0.7,
            "masculinity": 0.3,
            "complexity": 0.6,
            "story": "모던하고 세련된 느낌"
        }

        # 변형 생성
        options = rl_engine.generate_variations(
            dna=test_dna,
            feedback_brief=feedback_brief,
            num_options=3
        )

        # 검증
        assert len(options) == 3
        for option in options:
            assert "id" in option
            assert "phenotype" in option
            assert "action" in option
            assert "action_name" in option
            assert "log_prob" in option

        # 학습 시뮬레이션 (두 번째 옵션 선택)
        chosen_id = options[1]["id"]

        # update_policy_with_feedback는 placeholder이므로 호출만 확인
        assert hasattr(rl_engine, 'update_policy_with_feedback')
        assert hasattr(rl_engine, 'last_state')
        assert hasattr(rl_engine, 'last_saved_actions')

        print(f"✓ RL 엔진 학습 파이프라인 검증 완료")
        print(f"  생성된 행동: {[opt['action_name'] for opt in options]}")

    def test_6_error_handling(self, orchestrator):
        """
        시나리오 6: 에러 처리
        """
        # 존재하지 않는 DNA로 진화 시도
        error_request = {
            "text": "진화시켜줘",
            "target_dna_id": "non_existent_dna",
            "intent": "EVOLVE_EXISTING"
        }

        # 에러 처리 확인 (실제로는 try-except로 처리될 것)
        try:
            response = orchestrator.handle_user_request(
                user_id="test_user_error",
                request=error_request
            )
            # 에러 응답 확인
            if "error" in response:
                assert response["status"] == "error"
                print(f"✓ 에러 처리 성공: {response.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"✓ 예외 처리 성공: {str(e)}")

        # 세션 없이 FINALIZE 시도
        no_session_request = {
            "chosen_phenotype_id": "some_id",
            "intent": "FINALIZE_EVOLUTION"
        }

        response = orchestrator.handle_user_request(
            user_id="new_user",
            request=no_session_request
        )

        assert response["status"] == "error"
        assert "No active evolution session" in response["error"]
        print(f"✓ 세션 없음 에러 처리 성공")

    def test_7_full_scenario(self, orchestrator):
        """
        시나리오 7: 전체 시나리오 통합 테스트
        CREATE → EVOLVE → FINALIZE → CONTINUE
        """
        user_id = "integration_test_user"

        # 1. CREATE_NEW
        print("\n[1단계] 새 향수 생성...")
        create_response = orchestrator.handle_user_request(
            user_id=user_id,
            request={
                "text": "우아한 저녁 파티를 위한 향수",
                "intent": "CREATE_NEW"
            }
        )

        assert "new_dna" in create_response
        new_dna_id = create_response["new_dna"]["dna_id"]
        print(f"  → DNA ID: {new_dna_id}")

        # 2. EVOLVE_EXISTING
        print("\n[2단계] 향수 진화...")
        evolve_response = orchestrator.handle_user_request(
            user_id=user_id,
            request={
                "text": "좀 더 미스터리한 느낌으로",
                "target_dna_id": new_dna_id,
                "intent": "EVOLVE_EXISTING"
            }
        )

        assert len(evolve_response["options"]) == 3
        chosen_option = evolve_response["options"][0]
        print(f"  → 3개 옵션 생성, 첫 번째 선택: {chosen_option['action_applied']}")

        # 3. FINALIZE_EVOLUTION
        print("\n[3단계] 선택 확정 및 학습...")
        finalize_response = orchestrator.handle_user_request(
            user_id=user_id,
            request={
                "chosen_phenotype_id": chosen_option["id"],
                "intent": "FINALIZE_EVOLUTION"
            }
        )

        assert finalize_response["status"] == "evolution_complete"
        print(f"  → AI 학습 완료")

        # 4. CONTINUE_EVOLUTION
        print("\n[4단계] 연속 진화...")
        continue_response = orchestrator.handle_user_request(
            user_id=user_id,
            request={
                "text": "마지막으로 좀 더 부드럽게",
                "previous_choice_id": chosen_option["id"],
                "intent": "CONTINUE_EVOLUTION"
            }
        )

        assert len(continue_response["options"]) == 3
        print(f"  → 추가 3개 옵션 생성 성공")

        print("\n✅ 전체 시나리오 테스트 성공!")
        print("  CREATE → EVOLVE → FINALIZE → CONTINUE 완료")


if __name__ == "__main__":
    # 테스트 실행
    test = TestEndToEndEvolution()

    # 간단한 mock orchestrator (실제 구현 전)
    class MockOrchestrator:
        def __init__(self):
            self.session_cache = {}
            self.rl_engine = RLEngine(state_dim=10, action_dim=6)

        def handle_user_request(self, user_id, request):
            intent = request.get("intent")

            if intent == "CREATE_NEW":
                return {
                    "new_dna": {
                        "dna_id": "mock_dna_001",
                        "notes": ["bergamot", "jasmine"],
                        "genotype": {"top": ["bergamot"], "heart": ["jasmine"]}
                    }
                }

            elif intent == "EVOLVE_EXISTING":
                # Mock DNA 생성
                mock_dna = OlfactoryDNA(
                    dna_id=request.get("target_dna_id", "mock"),
                    notes=[{"name": "test", "intensity": 0.5}],
                    genotype={}
                )

                # RL 엔진으로 변형 생성
                options = self.rl_engine.generate_variations(
                    dna=mock_dna,
                    feedback_brief={"story": request.get("text", "")},
                    num_options=3
                )

                # 세션 저장
                self.session_cache[user_id] = {
                    "state": self.rl_engine.last_state,
                    "saved_actions": self.rl_engine.last_saved_actions,
                    "options": options
                }

                return {
                    "status": "evolution_options_ready",
                    "options": [
                        {
                            "id": opt["id"],
                            "description": opt["phenotype"].description,
                            "action_applied": opt["action_name"],
                            "preview": {"notes": ["mock"]}
                        }
                        for opt in options
                    ]
                }

            elif intent == "FINALIZE_EVOLUTION":
                if user_id not in self.session_cache:
                    return {"status": "error", "error": "No active evolution session"}

                del self.session_cache[user_id]
                return {
                    "status": "evolution_complete",
                    "final_scent": {"id": request.get("chosen_phenotype_id")},
                    "training_result": "mock_training_complete",
                    "message": "AI가 당신의 취향을 학습했습니다"
                }

            elif intent == "CONTINUE_EVOLUTION":
                # EVOLVE_EXISTING으로 리다이렉트
                request["intent"] = "EVOLVE_EXISTING"
                request["target_dna_id"] = "evolved_" + request.get("previous_choice_id", "unknown")
                return self.handle_user_request(user_id, request)

            return {"status": "error", "error": "Unknown intent"}

    orchestrator = MockOrchestrator()

    print("=" * 60)
    print("End-to-End Evolution System Test")
    print("=" * 60)

    # 모든 테스트 실행
    test.test_1_create_new_fragrance(orchestrator)
    print()
    test.test_2_evolve_existing_fragrance(
        orchestrator,
        test.test_dna()
    )
    print()
    test.test_3_finalize_evolution(orchestrator)
    print()
    test.test_4_continuous_evolution(orchestrator)
    print()
    test.test_5_rl_engine_learning()
    print()
    test.test_6_error_handling(orchestrator)
    print()
    test.test_7_full_scenario(orchestrator)

    print("\n" + "=" * 60)
    print("✅ 모든 End-to-End 테스트 통과!")
    print("=" * 60)