# fragrance_ai/orchestrator/living_scent_orchestrator.py (수도코드 예시)

from fragrance_ai.training.reinforcement_learning import RLEngine
# ... other imports

class LivingScentOrchestrator:
    def __init__(self):
        # ... 기존 초기화 코드

        # RL 엔진 초기화 (state_dim, action_dim은 설정 파일에서 가져와야 함)
        self.rl_engine = RLEngine(state_dim=10, action_dim=6)
        self.rl_engine.load_model()  # 저장된 모델 로드
        self.session_cache = {}  # 대화 세션별 학습 데이터 임시 저장

    def handle_user_request(self, user_id, request):
        """
        사용자 요청을 처리하는 메인 메서드
        """
        # 자연어 처리를 통해 사용자 의도 파악
        structured_input = self.linguistic_receptor.process(request.text)

        if structured_input.intent == 'CREATE_NEW':
            # 새로운 향수 DNA 생성 (기존 MOGA 로직)
            dna = self.moga_optimizer.create_new_fragrance(
                brief=structured_input.creative_brief
            )
            return {"new_dna": dna.to_dict()}

        elif structured_input.intent == 'EVOLVE_EXISTING':
            # === 강화학습 기반 진화 프로세스 ===

            # 1. 사용자가 변형을 원하는 DNA ID를 가져온다
            dna_id = request.target_dna_id
            dna = self.db.get_dna_by_id(dna_id)

            # 2. 피드백을 CreativeBrief로 변환한다
            feedback_brief = self.cognitive_core.process(structured_input)

            # 3. RL 엔진을 사용해 변형 후보들을 생성한다
            options = self.rl_engine.generate_variations(
                dna=dna,
                feedback_brief=feedback_brief,
                num_options=3  # 3개의 변형 후보 생성
            )

            # 4. 다음 피드백을 위해 학습에 필요한 데이터를 세션에 저장한다
            self.session_cache[user_id] = {
                "state": self.rl_engine.last_state,
                "saved_actions": self.rl_engine.last_saved_actions,
                "options": options,
                "original_dna": dna,
                "feedback_brief": feedback_brief
            }

            # 5. 사용자에게 후보들을 반환한다
            response_options = []
            for opt in options:
                response_options.append({
                    "id": opt["id"],
                    "description": opt["phenotype"].description,
                    "action_applied": opt["action_name"],
                    "preview": self.generate_scent_preview(opt["phenotype"])
                })

            return {
                "status": "evolution_options_ready",
                "options": response_options,
                "message": "3가지 진화 방향을 제안드립니다. 가장 마음에 드는 것을 선택해주세요."
            }

        elif structured_input.intent == 'FINALIZE_EVOLUTION':
            # === 사용자 선택 기반 학습 프로세스 ===

            # 1. 사용자가 최종 선택한 Phenotype ID를 가져온다
            chosen_id = request.chosen_phenotype_id

            # 2. 세션에서 이전 학습 데이터를 가져온다
            cached_data = self.session_cache.get(user_id)
            if not cached_data:
                return {
                    "status": "error",
                    "error": "No active evolution session.",
                    "message": "진화 세션이 없습니다. 먼저 EVOLVE_EXISTING으로 시작해주세요."
                }

            # 3. RL 엔진의 정책을 업데이트한다 (REINFORCE/PPO 알고리즘)
            # 사용자가 평점을 제공한 경우 활용
            user_rating = request.get('rating', None)  # 1-5 스케일 (선택사항)

            training_result = self.rl_engine.update_policy_with_feedback(
                chosen_phenotype_id=chosen_id,
                options=cached_data['options'],
                state=cached_data['state'],
                saved_actions=cached_data['saved_actions'],
                rating=user_rating
            )

            # 학습 결과 로깅
            print(f"[RL Training] User: {user_id}, Loss: {training_result.get('loss', 'N/A'):.4f}, "
                  f"Reward: {training_result.get('reward', 'N/A'):.2f}, "
                  f"Algorithm: {training_result.get('algorithm', 'REINFORCE')}")

            # 4. 선택된 Phenotype를 데이터베이스에 저장
            chosen_phenotype = next(
                (opt["phenotype"] for opt in cached_data['options']
                 if opt["id"] == chosen_id),
                None
            )

            if chosen_phenotype:
                self.db.save_phenotype(chosen_phenotype)
                self.db.log_user_preference(
                    user_id=user_id,
                    phenotype_id=chosen_id,
                    preference_score=1.0  # 선택됨 = 최고 점수
                )

            # 5. 학습 모델을 저장
            self.rl_engine.save_model(f"models/rl_model_{user_id}.pth")

            # 6. 세션 데이터를 삭제한다
            del self.session_cache[user_id]

            # 7. 최종 선택된 향수 정보를 반환한다
            return {
                "status": "evolution_complete",
                "final_scent": chosen_phenotype.to_dict() if chosen_phenotype else None,
                "training_result": training_result,
                "message": f"진화가 완료되었습니다. AI가 당신의 취향을 학습했습니다.",
                "next_steps": "이제 더 정교한 추천이 가능합니다."
            }

        elif structured_input.intent == 'CONTINUE_EVOLUTION':
            # === 연속 진화: 선택한 것을 바탕으로 추가 진화 ===

            # 이전에 선택한 Phenotype를 새로운 DNA로 변환
            previous_phenotype_id = request.previous_choice_id
            previous_phenotype = self.db.get_phenotype_by_id(previous_phenotype_id)

            # Phenotype를 DNA로 변환 (에피제네틱 변형을 유전자로 고정)
            evolved_dna = self.phenotype_to_dna(previous_phenotype)

            # 새로운 피드백을 받아 다시 진화 시작
            new_feedback_brief = self.cognitive_core.process(structured_input)

            # 재귀적으로 EVOLVE_EXISTING 프로세스 실행
            request.target_dna_id = evolved_dna.dna_id
            structured_input.intent = 'EVOLVE_EXISTING'

            return self.handle_user_request(user_id, request)

    def generate_scent_preview(self, phenotype):
        """
        Phenotype의 향수 미리보기 생성 (시각화나 설명)
        """
        # 향수의 노트 구성, 강도, 지속성 등을 설명
        preview = {
            "top_notes": phenotype.get_top_notes(),
            "heart_notes": phenotype.get_heart_notes(),
            "base_notes": phenotype.get_base_notes(),
            "intensity": phenotype.calculate_intensity(),
            "longevity": phenotype.estimate_longevity(),
            "character": phenotype.describe_character()
        }
        return preview

    def phenotype_to_dna(self, phenotype):
        """
        Phenotype의 변형을 영구적인 DNA로 변환
        """
        # 에피제네틱 변형을 유전자 수준으로 고정
        new_dna = OlfactoryDNA(
            dna_id=f"evolved_{phenotype.phenotype_id}",
            genotype=phenotype.recipe_adjusted,
            lineage=phenotype.based_on_dna,
            generation=phenotype.generation + 1
        )
        self.db.save_dna(new_dna)
        return new_dna


# === 사용 예시 (수도코드) ===

# 사용자 1: "이 향수를 더 따뜻하고 관능적으로 만들어줘"
orchestrator = LivingScentOrchestrator()
response1 = orchestrator.handle_user_request(
    user_id="user_123",
    request={
        "text": "이 향수를 더 따뜻하고 관능적으로 만들어줘",
        "target_dna_id": "dna_456",
        "intent": "EVOLVE_EXISTING"
    }
)
# response1: 3개의 변형 옵션 반환

# 사용자 2: "두 번째 옵션이 마음에 들어"
response2 = orchestrator.handle_user_request(
    user_id="user_123",
    request={
        "text": "두 번째 옵션이 마음에 들어",
        "chosen_phenotype_id": response1["options"][1]["id"],
        "intent": "FINALIZE_EVOLUTION"
    }
)
# response2: 최종 선택 완료 및 AI 학습 완료

# 사용자 3: "이걸 기반으로 좀 더 신선하게 만들어줘"
response3 = orchestrator.handle_user_request(
    user_id="user_123",
    request={
        "text": "이걸 기반으로 좀 더 신선하게 만들어줘",
        "previous_choice_id": response2["final_scent"]["phenotype_id"],
        "intent": "CONTINUE_EVOLUTION"
    }
)
# response3: 새로운 3개의 진화 옵션 (이전 선택을 기반으로)