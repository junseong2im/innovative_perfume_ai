"""
Living Scent Optimizer Test Suite
각 옵티마이저의 성능을 테스트하고 검증

1. AdamW - 신경망 수렴 테스트
2. NSGA-III - 파레토 최적해 찾기 테스트
3. PPO-RLHF - 강화학습 보상 개선 테스트
"""

import torch
import torch.nn as nn
import numpy as np
from fragrance_ai.training.living_scent_optimizers import (
    NeuralNetworkOptimizer,
    AdamWConfig,
    NSGAIII,
    PPO_RLHF,
    Fragrance,
    Experience,
    get_optimizer_manager
)


def print_section(title: str):
    """섹션 구분선 출력"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_adamw_optimizer():
    """AdamW 옵티마이저 테스트"""
    print_section("AdamW Optimizer Test")

    # 간단한 신경망 모델
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 50)
            self.fc3 = nn.Linear(50, 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    # 모델과 옵티마이저 초기화
    model = SimpleNet()
    config = AdamWConfig(
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=100
    )
    optimizer = NeuralNetworkOptimizer(model, config)

    print("✓ AdamW 옵티마이저 초기화 완료")
    print(f"  - Learning Rate: {config.learning_rate}")
    print(f"  - Weight Decay: {config.weight_decay}")

    # 더미 데이터로 훈련 테스트
    losses = []
    for step in range(100):
        # 더미 데이터
        inputs = torch.randn(32, 10)
        labels = torch.randint(0, 2, (32,))

        # 손실 함수
        loss_fn = nn.CrossEntropyLoss()

        # 훈련 스텝
        loss = optimizer.train_step(inputs, labels,
                                   lambda outputs, labels: loss_fn(model(inputs), labels))
        losses.append(loss)

        if step % 20 == 0:
            current_lr = optimizer.get_current_lr()
            print(f"  Step {step}: Loss = {loss:.4f}, LR = {current_lr:.2e}")

    # 손실 감소 확인
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    improvement = (initial_loss - final_loss) / initial_loss * 100

    print(f"\n✓ 훈련 완료")
    print(f"  - 초기 손실: {initial_loss:.4f}")
    print(f"  - 최종 손실: {final_loss:.4f}")
    print(f"  - 개선율: {improvement:.1f}%")

    if improvement > 10:
        print("  ✅ AdamW 테스트 성공!")
    else:
        print("  ⚠️ 개선율이 낮습니다")

    return losses


def test_nsga3_optimizer():
    """NSGA-III 다목적 최적화 테스트"""
    print_section("NSGA-III Multi-Objective Optimizer Test")

    # NSGA-III 초기화
    optimizer = NSGAIII(
        population_size=50,
        num_generations=20,
        crossover_prob=0.9,
        mutation_prob=0.1
    )

    print("✓ NSGA-III 옵티마이저 초기화 완료")
    print(f"  - Population Size: {optimizer.population_size}")
    print(f"  - Generations: {optimizer.num_generations}")

    # 사용자 요구사항
    user_requirements = {
        'fresh': 0.8,
        'woody': 0.6,
        'lasting': 0.7
    }

    print(f"\n사용자 요구사항: {user_requirements}")

    # 최적화 실행
    print("\n최적화 실행 중...")
    pareto_front = optimizer.optimize(user_requirements)

    print(f"\n✓ 최적화 완료")
    print(f"  - 파레토 최적해 개수: {len(pareto_front)}")

    # 상위 3개 해 출력
    if pareto_front:
        print("\n상위 3개 파레토 최적 향수:")
        for i, fragrance in enumerate(pareto_front[:3]):
            print(f"\n  향수 #{i+1}:")
            print(f"    - 조화성: {fragrance.objectives['harmony']:.3f}")
            print(f"    - 독창성: {fragrance.objectives['uniqueness']:.3f}")
            print(f"    - 사용자 적합성: {fragrance.objectives['user_fitness']:.3f}")

    # 진화 히스토리 분석
    if optimizer.evolution_history:
        first_gen = optimizer.evolution_history[0]
        last_gen = optimizer.evolution_history[-1]

        print("\n진화 통계:")
        print(f"  첫 세대 최고 조화성: {first_gen['best_harmony']:.3f}")
        print(f"  마지막 세대 최고 조화성: {last_gen['best_harmony']:.3f}")

        harmony_improvement = (last_gen['best_harmony'] - first_gen['best_harmony']) / first_gen['best_harmony'] * 100
        print(f"  조화성 개선율: {harmony_improvement:.1f}%")

        if len(pareto_front) > 0:
            print("  ✅ NSGA-III 테스트 성공!")

    return pareto_front


def test_ppo_rlhf_optimizer():
    """PPO-RLHF 강화학습 테스트"""
    print_section("PPO-RLHF Reinforcement Learning Test")

    # PPO-RLHF 초기화
    state_dim = 20
    action_dim = 5
    optimizer = PPO_RLHF(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        gamma=0.99
    )

    print("✓ PPO-RLHF 옵티마이저 초기화 완료")
    print(f"  - State Dimension: {state_dim}")
    print(f"  - Action Dimension: {action_dim}")
    print(f"  - Learning Rate: 3e-4")

    # 에피소드 시뮬레이션
    episode_rewards = []
    print("\n강화학습 훈련 시작...")

    for episode in range(50):
        state = np.random.randn(state_dim)
        episode_reward = 0

        for step in range(20):
            # 행동 선택
            action, log_prob, value = optimizer.select_action(state)

            # 환경 시뮬레이션
            next_state = state + np.random.randn(state_dim) * 0.1

            # 보상 계산 (간단한 시뮬레이션)
            if action == 0 and state[0] > 0:
                reward = 1.0
            elif action == 1 and state[0] < 0:
                reward = 0.5
            else:
                reward = -0.1

            # 경험 저장
            exp = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(step == 19),
                log_prob=log_prob,
                value=value
            )
            optimizer.store_experience(exp)

            # 인간 피드백 시뮬레이션
            optimizer.collect_human_feedback(state, action, reward)

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # 10 에피소드마다 훈련
        if episode % 10 == 0:
            optimizer.train(batch_size=32, epochs=5)
            print(f"  Episode {episode}: Avg Reward = {np.mean(episode_rewards[-10:]):.2f}")

    # 학습 개선 확인
    early_rewards = np.mean(episode_rewards[:10])
    late_rewards = np.mean(episode_rewards[-10:])
    improvement = late_rewards - early_rewards

    print(f"\n✓ 훈련 완료")
    print(f"  - 초기 평균 보상: {early_rewards:.2f}")
    print(f"  - 최종 평균 보상: {late_rewards:.2f}")
    print(f"  - 개선: {improvement:.2f}")

    if improvement > 0:
        print("  ✅ PPO-RLHF 테스트 성공!")
    else:
        print("  ⚠️ 보상이 개선되지 않았습니다")

    # 보상 모델 정확도
    if optimizer.training_stats['human_feedback_accuracy']:
        final_accuracy = np.mean(optimizer.training_stats['human_feedback_accuracy'][-10:])
        print(f"  - 인간 피드백 모델 손실: {final_accuracy:.4f}")

    return episode_rewards


def test_optimizer_manager():
    """옵티마이저 매니저 테스트"""
    print_section("Optimizer Manager Test")

    manager = get_optimizer_manager()

    # 각 옵티마이저 등록
    simple_model = nn.Linear(10, 2)
    manager.register_neural_optimizer("test_neural", simple_model)
    manager.register_genetic_optimizer("test_genetic", population_size=20)
    manager.register_rl_optimizer("test_rl", state_dim=10, action_dim=5)

    print("✓ 옵티마이저 매니저 테스트")
    print(f"  - 등록된 옵티마이저 수: {len(manager.optimizers)}")

    for name in manager.optimizers:
        print(f"  - {name}: ✓")

    # 옵티마이저 가져오기 테스트
    neural_opt = manager.get_optimizer("test_neural")
    if neural_opt:
        print("  ✅ 옵티마이저 매니저 테스트 성공!")


def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("       Living Scent Optimizer Test Suite")
    print("="*60)

    try:
        # 1. AdamW 테스트
        adamw_losses = test_adamw_optimizer()

        # 2. NSGA-III 테스트
        pareto_front = test_nsga3_optimizer()

        # 3. PPO-RLHF 테스트
        episode_rewards = test_ppo_rlhf_optimizer()

        # 4. 옵티마이저 매니저 테스트
        test_optimizer_manager()

        print_section("테스트 요약")
        print("\n✅ 모든 옵티마이저 테스트 완료!")
        print("\n구현된 옵티마이저:")
        print("  1. AdamW - 뇌 신경망 훈련용 ✓")
        print("  2. NSGA-III - 다목적 최적화 (DNA 창조) ✓")
        print("  3. PPO-RLHF - 인간 피드백 강화학습 (진화) ✓")

        print("\n각 옵티마이저 특징:")
        print("  • AdamW: 가중치 감쇠로 과적합 방지, 동적 학습률")
        print("  • NSGA-III: 조화성, 독창성, 사용자 적합성 동시 최적화")
        print("  • PPO-RLHF: 사용자 피드백으로 지속적 개선")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("         테스트 종료")
    print("="*60)


if __name__ == "__main__":
    main()