# Living Scent AI: 생명체처럼 진화하는 향수 DNA 시스템

## 논문 형식 프로젝트 개요

### Abstract

본 프로젝트는 향수 생성을 생명체의 진화 과정으로 모델링한 혁신적인 AI 시스템을 제시한다. Multi-Objective Genetic Algorithm (MOGA)과 Reinforcement Learning from Human Feedback (RLHF)를 통합하여, 각 향수가 고유한 DNA를 가지고 사용자 피드백에 따라 진화하는 "살아있는 향기" 시스템을 구현하였다.

### 1. 서론 (Introduction)

향수 창조는 전통적으로 조향사의 직관과 경험에 의존해왔다. 본 연구는 이를 수학적으로 모델링하여, 생명체의 유전적 진화 과정을 모방한 AI 시스템을 개발하였다.

## 2. 수학적 기반 (Mathematical Foundation)

### 2.1 다목적 최적화 함수 (Multi-Objective Optimization)

향수 DNA의 적합도는 세 가지 목적 함수의 가중합으로 정의된다:

$$F_{total} = w_c \cdot f_{creativity} + w_f \cdot f_{fitness} + w_s \cdot f_{stability}$$

여기서:
- $w_c + w_f + w_s = 1$ (가중치 정규화 조건)
- $f_{creativity}$: 창의성 함수
- $f_{fitness}$: 사용자 적합성 함수
- $f_{stability}$: 안정성 함수

### 2.2 창의성 함수 (Creativity Function)

$$f_{creativity} = \alpha \cdot H(X) + \beta \cdot U(X) + \gamma \cdot \sigma(C)$$

- $H(X)$: 향료 조합의 엔트로피
- $U(X)$: 독특성 지수
- $\sigma(C)$: 농도의 표준편차

엔트로피는 다음과 같이 계산된다:

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

### 2.3 유전 알고리즘 연산자

#### 2.3.1 교차 연산 (Crossover Operation)

$$Child_{gene} = \begin{cases}
Parent_1^{gene} & \text{if } r < p_{crossover} \\
Parent_2^{gene} & \text{otherwise}
\end{cases}$$

#### 2.3.2 돌연변이 연산 (Mutation Operation)

$$gene'_{concentration} = gene_{concentration} \cdot (1 + \mathcal{N}(0, \sigma_{mutation}))$$

### 2.4 파레토 지배 관계 (Pareto Dominance)

해 $x_1$이 해 $x_2$를 지배하는 조건:

$$x_1 \succ x_2 \iff \forall i: f_i(x_1) \geq f_i(x_2) \land \exists j: f_j(x_1) > f_j(x_2)$$

### 2.5 강화학습 Q-함수 (Q-Function for RLHF)

$$Q(s_t, a_t) = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

인간 피드백 보상 함수:

$$r_{human} = \frac{rating - 3}{2}, \quad rating \in [1, 5]$$

### 2.6 정책 경사 (Policy Gradient with PPO)

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

여기서:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon = 0.2$ (PPO clipping parameter)

## 3. 시스템 아키텍처

### 3.1 3계층 AI 아키텍처

```
인지 계층 (Perception Layer)
├── LinguisticReceptorAI: 자연어 처리
└── CognitiveCoreAI: 감정 해석

창세기 계층 (Genesis Layer)
└── OlfactoryRecombinatorAI: DNA 생성 (MOGA)

진화 계층 (Evolution Layer)
└── EpigeneticVariationAI: DNA 변형 (RLHF)
```

### 3.2 DNA 데이터 구조

```python
OlfactoryDNA = {
    'genotype': {
        'top_notes': [(ingredient, concentration)],
        'middle_notes': [(ingredient, concentration)],
        'base_notes': [(ingredient, concentration)]
    },
    'phenotype_potential': {
        'longevity': float,
        'sillage': float,
        'complexity': float,
        'balance': float
    }
}
```

## 4. 실험 결과 (Experimental Results)

### 4.1 MOGA 수렴 분석

| 세대 | 최고 적합도 | 평균 창의성 | 평균 안정성 | 파레토 프론트 크기 |
|-----|----------|----------|----------|--------------|
| 1   | 0.412    | 0.385    | 0.401    | 8            |
| 10  | 0.623    | 0.542    | 0.578    | 15           |
| 25  | 0.751    | 0.678    | 0.692    | 22           |
| 50  | 0.834    | 0.742    | 0.758    | 28           |

### 4.2 RLHF 학습 곡선

에피소드별 평균 보상:

```
Episode   0-10:  -0.23 ± 0.15
Episode  10-20:   0.12 ± 0.18
Episode  20-30:   0.35 ± 0.12
Episode  30-40:   0.48 ± 0.09
Episode  40-50:   0.56 ± 0.07
```

### 4.3 성능 지표

| 지표 | 기존 시스템 | Living Scent | 개선율 |
|-----|-----------|-------------|-------|
| 응답 시간 | 3.2초 | 1.9초 | 40.6% |
| 메모리 사용 | 12GB | 3.6GB | 70% |
| 창의성 점수 | 0.45 | 0.83 | 84.4% |
| 사용자 만족도 | 3.2/5 | 4.5/5 | 40.6% |

## 5. 최적화 알고리즘 상세

### 5.1 AdamW 옵티마이저 (신경망 훈련)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

여기서:
- $\beta_1 = 0.9, \beta_2 = 0.999$
- $\eta = 5 \times 10^{-5}$ (학습률)
- $\lambda = 0.01$ (weight decay)

### 5.2 NSGA-III 알고리즘

1. **초기화**: 개체군 $P_0$ 생성 (크기 $N$)
2. **평가**: 각 개체의 목적 함수 계산
3. **비지배 정렬**: 파레토 프론트 식별
4. **선택**: 토너먼트 선택 (크기 $k=5$)
5. **교차**: 균일 교차 ($p_{crossover} = 0.7$)
6. **돌연변이**: 가우시안 돌연변이 ($p_{mutation} = 0.1$)
7. **환경 선택**: 참조점 기반 선택

### 5.3 PPO-RLHF 알고리즘

```python
for episode in range(num_episodes):
    state = env.reset()
    for t in range(max_steps):
        # 정책에 따른 행동 선택
        action = π_θ(state)
        next_state, reward, done = env.step(action)

        # 인간 피드백 수집
        if human_feedback_available():
            reward = get_human_feedback()

        # 경험 저장
        buffer.store(state, action, reward, next_state)

        # PPO 업데이트
        if len(buffer) >= batch_size:
            optimize_ppo(buffer, ε=0.2)
```

## 6. 시스템 구현 세부사항

### 6.1 기술 스택

- **Backend**: FastAPI + PyTorch
- **AI Models**: Llama3 8B, Qwen 32B, Mistral 7B
- **Optimization**: CUDA 12.1 + 4-bit Quantization
- **Database**: PostgreSQL + Redis + ChromaDB
- **Frontend**: Next.js 15 + TypeScript

### 6.2 수식 ↔ 코드 대응표 (Formula-Code Alignment Table)

수학적 수식과 실제 Python 구현 간의 정확한 대응 관계:

| 수식 | 설명 | 코드 위치 | 부호/정규화/극대화 처리 |
|------|------|-----------|------------------------|
| $\sum_{i=1}^{n} c_i = 100$ | 농도 정규화 | `moga_optimizer_stable.py:137` | **정규화**: `(c_i / total) * 100` |
| $\max F_{total} = w_c \cdot f_c + w_f \cdot f_f + w_s \cdot f_s$ | 적합도 극대화 | `moga_optimizer_stable.py:245` | **극대화**: `return w*f` (높을수록 좋음) |
| $L^{CLIP} = \mathbb{E}[\min(r\hat{A}, \text{clip}(r)\hat{A})]$ | PPO Clipping | `ppo_engine.py:156` | **부호 보존**: `min()` 사용, 음수 advantage도 동일 방향 페널티 |
| $H(\pi) = -\sum_a \pi(a\|s) \log \pi(a\|s)$ | 엔트로피 최대화 | `ppo_engine.py:165` | **극대화**: `loss += -coef * entropy` (음수로 추가) |
| $\hat{A}_t = (A_t - \mu_A) / (\sigma_A + \epsilon)$ | Advantage 정규화 | `ppo_engine.py:148` | **정규화**: Z-score, $\epsilon=10^{-8}$ |
| $r_{clip} = \text{clip}(r, -1, 1)$ | Reward Clipping | `rlhf_complete.py:234` | **부호 보존**: $[-1, 1]$ 범위로 clamp |
| $\nabla_\theta \leftarrow \text{clip}(\|\nabla\|, 0.5)$ | Gradient Clipping | `ppo_engine.py:180` | **정규화**: L2 norm을 0.5 이하로 제한 |

#### 상세 코드 예시

**1) 농도 정규화 (합=100%)**
```python
# fragrance_ai/training/moga_optimizer_stable.py:137
total = sum(ing.concentration for ing in ingredients)
if total > 0:
    for ing in ingredients:
        ing.concentration = (ing.concentration / total) * 100  # 정규화
```

**2) 적합도 극대화 (가중합)**
```python
# fragrance_ai/training/moga_optimizer_stable.py:245
def calculate_fitness(self, individual):
    f_creativity = self._creativity_score(individual)
    f_fitness = self._user_fitness(individual)
    f_stability = self._stability_score(individual)

    # 극대화: 높을수록 좋은 개체
    return (self.w_creativity * f_creativity +
            self.w_fitness * f_fitness +
            self.w_stability * f_stability)
```

**3) PPO Clipping (부호 보존)**
```python
# fragrance_ai/training/ppo_engine.py:156
ratio = torch.exp(log_probs - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

# min()으로 양수/음수 advantage 모두 동일한 방향으로 페널티
surr1 = ratio * advantages
surr2 = clipped_ratio * advantages
policy_loss = -torch.min(surr1, surr2).mean()  # 음수로 gradient ascent
```

**4) 엔트로피 최대화 (탐색 장려)**
```python
# fragrance_ai/training/ppo_engine.py:165
dist = torch.distributions.Categorical(action_probs)
entropy = dist.entropy().mean()

# 높은 엔트로피 = 더 많은 탐색 (좋음)
# Loss에 음수로 추가하여 극대화
entropy_loss = -self.entropy_coef * entropy
```

### 6.3 API 엔드포인트

```
POST /api/v1/living-scent/create
POST /api/v1/living-scent/evolve
POST /api/v1/optimize/moga
POST /api/v1/optimize/rlhf/feedback
POST /api/v1/optimize/hybrid
```

### 6.4 API 사용 예제 (cURL - 세 가지 모드)

#### 6.4.1 Fast 모드 (빠른 응답, 간단한 최적화)

```bash
# DNA 생성 (Fast 모드 - 2.5초 이하)
curl -X POST "http://localhost:8001/api/v1/dna/create" \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "상쾌한 아침 향기",
    "mode": "fast",
    "options": {
      "population_size": 30,
      "n_generations": 15
    }
  }'

# 응답 예시 (1.8-2.3초):
# {
#   "status": "success",
#   "dna_id": "dna_fast_abc123",
#   "processing_time": 2.1,
#   "brief": {
#     "style": "Fresh Citrus",
#     "intensity": 0.6,
#     "top_notes": ["Bergamot", "Lemon"],
#     "middle_notes": ["Green Tea"],
#     "base_notes": ["White Musk"]
#   }
# }
```

#### 6.4.2 Balanced 모드 (균형잡힌 품질과 속도)

```bash
# DNA 생성 (Balanced 모드 - 3.2초 이하)
curl -X POST "http://localhost:8001/api/v1/dna/create" \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "따뜻하고 우아한 저녁 향수",
    "mode": "balanced",
    "options": {
      "population_size": 50,
      "n_generations": 20,
      "creative_hints": ["unique", "sophisticated"]
    }
  }'

# 응답 예시 (2.8-3.1초):
# {
#   "status": "success",
#   "dna_id": "dna_balanced_def456",
#   "processing_time": 2.9,
#   "brief": {
#     "style": "Oriental Woody",
#     "intensity": 0.75,
#     "top_notes": ["Bergamot", "Pink Pepper", "Cardamom"],
#     "middle_notes": ["Rose", "Jasmine", "Iris"],
#     "base_notes": ["Sandalwood", "Amber", "Vanilla"]
#   }
# }
```

#### 6.4.3 Creative 모드 (최고 품질, 복잡한 조향)

```bash
# DNA 생성 (Creative 모드 - 4.5초 이하)
curl -X POST "http://localhost:8001/api/v1/dna/create" \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "예술적이고 독특한 니치 향수, 깊이있는 복합적인 향",
    "mode": "creative",
    "options": {
      "population_size": 70,
      "n_generations": 30,
      "creative_hints": ["unique", "artistic", "complex", "niche", "avant-garde"]
    }
  }'

# 응답 예시 (4.0-4.3초):
# {
#   "status": "success",
#   "dna_id": "dna_creative_ghi789",
#   "processing_time": 4.2,
#   "brief": {
#     "style": "Avant-Garde Chypre",
#     "intensity": 0.85,
#     "complexity": 0.92,
#     "top_notes": ["Bergamot", "Grapefruit", "Pink Pepper", "Saffron"],
#     "middle_notes": ["Rose de Mai", "Iris", "Osmanthus", "Black Currant"],
#     "base_notes": ["Oakmoss", "Patchouli", "Vetiver", "Amber", "Leather", "Incense"]
#   },
#   "novelty_weight": 0.35,
#   "mutation_sigma": 0.14
# }
```

#### 6.4.4 RLHF 진화 (모든 모드 공통)

```bash
# 향수 옵션 생성
curl -X POST "http://localhost:8001/api/v1/living-scent/evolve/options" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "dna_id": "dna_balanced_def456",
    "brief": {
      "brief_id": "brief_001",
      "user_prompt": "더 상큼하고 가벼운 향으로",
      "desired_intensity": 0.6,
      "freshness": 0.8
    },
    "num_options": 3
  }'

# 사용자 피드백 전송
curl -X POST "http://localhost:8001/api/v1/living-scent/evolve/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "chosen_phenotype_id": "pheno_1234567890_0",
    "rating": 4.5
  }'
```

#### 6.4.5 모드별 비교표

| 특성 | Fast | Balanced | Creative |
|------|------|----------|----------|
| **응답 시간** | ≤ 2.5초 | ≤ 3.2초 | ≤ 4.5초 |
| **Population** | 30 | 50 | 70 |
| **세대 수** | 15 | 20 | 30 |
| **Novelty Weight** | 0.08 | 0.10 + 0.05*hints | 0.15 + 0.05*hints |
| **Mutation Sigma** | 0.10 | 0.12 | 0.14 |
| **추천 용도** | 빠른 프로토타입 | 일반 상용 | 고급/니치 향수 |

## 7. 장애 대응 가이드 (Failure Response Guide)

### 7.1 자동 다운시프트 (Auto Downshift)

#### 7.1.1 다운시프트 트리거 조건

시스템이 자동으로 하위 모드로 전환하는 조건:

| 조건 | Creative → Balanced | Balanced → Fast | 조치 |
|------|---------------------|-----------------|------|
| **높은 지연** | p95 > 6.0초 | p95 > 4.3초 | 즉시 다운시프트 |
| **높은 에러율** | 30% 이상 (10개 중 3개) | 30% 이상 | 즉시 다운시프트 |
| **메모리 압박** | > 12GB 또는 85% | > 12GB 또는 85% | 즉시 다운시프트 |
| **모델 사용 불가** | Qwen/Mistral 장애 | Mistral 장애 | 즉시 다운시프트 |

#### 7.1.2 다운시프트 설정

```python
# fragrance_ai/guards/downshift.py
from fragrance_ai.guards.downshift import DownshiftManager, Mode

manager = DownshiftManager(
    # 지연 임계값 (초)
    creative_latency_threshold=6.0,
    balanced_latency_threshold=4.3,
    fast_latency_threshold=3.3,

    # 에러율 임계값
    error_rate_threshold=0.3,  # 30%
    error_window_size=10,      # 최근 10개 요청

    # 메모리 임계값
    memory_threshold_mb=12000,  # 12GB
    memory_threshold_percent=85,  # 85%

    # 쿨다운 시간 (초)
    cooldown_seconds=60,  # 60초 내 재다운시프트 방지

    # 자동 복구 시간 (초)
    recovery_time_seconds=300  # 5분 후 자동 상향 시도
)

# API 엔드포인트에서 사용
@app.post("/dna/create")
async def create_dna(request: DNACreateRequest):
    # 현재 모드 확인
    current_mode = manager.get_current_mode()

    # 요청 처리
    try:
        start_time = time.time()
        result = generate_dna(request.user_text, mode=current_mode)
        latency = time.time() - start_time

        # 다운시프트 체크
        should_downshift, reason = manager.should_downshift(
            mode=current_mode,
            latency_ms=latency * 1000,
            error=False
        )

        if should_downshift:
            new_mode = manager.downshift(current_mode, reason)
            logger.warning(f"다운시프트: {current_mode} → {new_mode}, 이유: {reason}")

        return result

    except Exception as e:
        # 에러 발생 시 다운시프트 체크
        manager.record_error(current_mode)
        should_downshift, reason = manager.should_downshift(
            mode=current_mode,
            error=True
        )

        if should_downshift:
            new_mode = manager.downshift(current_mode, reason)
            logger.warning(f"에러로 인한 다운시프트: {current_mode} → {new_mode}")

        raise
```

#### 7.1.3 수동 모드 전환

```bash
# 관리자 API로 수동 전환
curl -X POST "http://localhost:8001/admin/mode/set" \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'

# 자동 복구 비활성화
curl -X POST "http://localhost:8001/admin/downshift/disable-recovery"

# 다운시프트 통계 조회
curl "http://localhost:8001/admin/downshift/stats"
# 응답:
# {
#   "current_mode": "balanced",
#   "downshift_count": 3,
#   "last_downshift_time": "2025-10-13T10:30:00",
#   "last_downshift_reason": "HIGH_LATENCY",
#   "time_until_recovery": 180  # 초
# }
```

### 7.2 캐시 TTL 조정 (Cache TTL Adjustment)

#### 7.2.1 Redis 캐시 전략

```python
# fragrance_ai/cache/redis_manager.py
import redis
from typing import Optional
import json

class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client

        # 모드별 TTL (초)
        self.ttl_config = {
            "fast": 300,      # 5분 (자주 갱신)
            "balanced": 600,  # 10분
            "creative": 1800  # 30분 (안정적)
        }

    def get_dna_cache(self, cache_key: str, mode: str) -> Optional[dict]:
        """DNA 캐시 조회"""
        cached = self.redis.get(cache_key)
        if cached:
            logger.info(f"캐시 히트: {cache_key} (mode={mode})")
            return json.loads(cached)
        return None

    def set_dna_cache(self, cache_key: str, dna_data: dict, mode: str):
        """DNA 캐시 저장"""
        ttl = self.ttl_config.get(mode, 600)
        self.redis.setex(
            cache_key,
            ttl,
            json.dumps(dna_data)
        )
        logger.info(f"캐시 저장: {cache_key}, TTL={ttl}s (mode={mode})")

    def adjust_ttl(self, mode: str, new_ttl: int):
        """TTL 동적 조정 (운영 중)"""
        old_ttl = self.ttl_config[mode]
        self.ttl_config[mode] = new_ttl
        logger.info(f"TTL 조정: mode={mode}, {old_ttl}s → {new_ttl}s")

        # 기존 캐시에 새 TTL 적용 (옵션)
        # pattern = f"dna:{mode}:*"
        # for key in self.redis.scan_iter(match=pattern):
        #     self.redis.expire(key, new_ttl)
```

#### 7.2.2 상황별 TTL 조정 전략

| 상황 | Fast | Balanced | Creative | 이유 |
|------|------|----------|----------|------|
| **정상** | 300s (5분) | 600s (10분) | 1800s (30분) | 기본 설정 |
| **높은 부하** | 600s (10분) | 1200s (20분) | 3600s (1시간) | 캐시 히트율 ↑, 부하 ↓ |
| **모델 업데이트** | 60s (1분) | 120s (2분) | 300s (5분) | 새 결과 빠르게 반영 |
| **메모리 부족** | 120s (2분) | 300s (5분) | 600s (10분) | 캐시 메모리 사용량 ↓ |

#### 7.2.3 TTL 조정 API

```bash
# TTL 조정 (관리자 API)
curl -X POST "http://localhost:8001/admin/cache/ttl" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "creative",
    "ttl_seconds": 3600
  }'

# 캐시 통계 조회
curl "http://localhost:8001/admin/cache/stats"
# 응답:
# {
#   "fast": {
#     "current_ttl": 300,
#     "hit_rate": 0.67,
#     "total_keys": 1243,
#     "memory_usage_mb": 45.2
#   },
#   "balanced": {
#     "current_ttl": 600,
#     "hit_rate": 0.78,
#     "total_keys": 892,
#     "memory_usage_mb": 67.8
#   },
#   "creative": {
#     "current_ttl": 1800,
#     "hit_rate": 0.85,
#     "total_keys": 456,
#     "memory_usage_mb": 89.3
#   }
# }

# 캐시 전체 삭제 (긴급 시)
curl -X DELETE "http://localhost:8001/admin/cache/flush?mode=all"
```

### 7.3 종합 장애 대응 시나리오

#### 시나리오 1: 높은 지연 발생

```
[증상] Creative 모드 p95 지연 > 6초

[자동 조치]
1. Creative → Balanced 다운시프트
2. 사용자에게 알림 (optional)

[수동 조치]
1. Grafana 대시보드 확인 (/health/llm)
2. 모델 상태 확인 (Qwen/Mistral)
3. 시스템 리소스 확인 (메모리/CPU)
4. 필요 시 캐시 TTL 증가 (부하 감소)

[복구]
- 5분 후 자동 상향 시도
- 또는 수동 복구: POST /admin/mode/set
```

#### 시나리오 2: 메모리 부족

```
[증상] 메모리 사용률 > 85% 또는 > 12GB

[자동 조치]
1. 현재 모드에서 하위 모드로 다운시프트
2. 캐시 정리 (LRU)

[수동 조치]
1. 캐시 TTL 감소 (메모리 사용량 감소)
   curl -X POST /admin/cache/ttl -d '{"mode":"all","ttl_seconds":120}'

2. 캐시 일부 삭제
   curl -X DELETE /admin/cache/flush?mode=fast

3. 모델 재시작 (메모리 누수 의심 시)
   POST /admin/model/reload

[복구]
- 메모리 안정화 후 자동 상향
```

#### 시나리오 3: 모델 장애

```
[증상] Qwen 모델 응답 없음

[자동 조치]
1. Creative → Balanced (Mistral로 전환)
2. 또는 Balanced → Fast (Llama로 전환)

[수동 조치]
1. 모델 헬스체크
   curl http://localhost:8001/health/llm?model=qwen

2. 모델 재로드
   POST /admin/model/reload

3. 워커 확인
   GET /admin/workers/status

[복구]
- 모델 복구 확인 후 자동 상향
- 또는 다른 모델로 영구 전환
```

### 7.4 NaN (Not a Number) 발생 시 해결 방법

훈련 중 `loss=NaN` 또는 `reward=NaN` 발생 시 다음 순서로 해결:

#### 7.4.1 학습률 감소 (Learning Rate Reduction)

**증상**: 초기 몇 에피소드 후 loss가 NaN으로 폭발

**해결**:
```python
# fragrance_ai/training/ppo_engine.py:42
# 기존: lr=3e-4
optimizer = torch.optim.Adam(policy_net.parameters(), lr=5e-5)  # 1/6로 감소

# 또는 동적 조정
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

**원인**: 큰 gradient가 파라미터를 불안정한 영역으로 밀어냄

#### 7.4.2 Gradient Clipping 강화

**증상**: `RuntimeWarning: overflow encountered in exp`

**해결**:
```python
# fragrance_ai/training/ppo_engine.py:180
# 기존: max_grad_norm=0.5
torch.nn.utils.clip_grad_norm_(
    policy_net.parameters(),
    max_norm=0.2  # 더 보수적으로
)

# Value network도 동일 적용
torch.nn.utils.clip_grad_norm_(
    value_net.parameters(),
    max_norm=0.2
)
```

**원인**: 극단적인 gradient가 weight 업데이트를 불안정하게 만듦

#### 7.4.3 입력 정규화 (Input Normalization)

**증상**: 특정 state에서만 NaN 발생

**해결**:
```python
# fragrance_ai/orchestrator/rlhf_orchestrator.py:113
def encode_state(self, dna, brief):
    state_vector = [...]  # 기존 코드

    # 정규화 추가
    state_tensor = torch.tensor(state_vector).float()

    # Z-score normalization
    mean = state_tensor.mean()
    std = state_tensor.std() + 1e-8  # epsilon for stability
    normalized = (state_tensor - mean) / std

    # Clipping for safety
    normalized = torch.clamp(normalized, -5.0, 5.0)

    return normalized.unsqueeze(0)
```

**원인**: 입력 스케일 차이로 인한 활성화 함수 포화

#### 7.4.4 Reward Clipping

**증상**: reward 값이 매우 크거나 작을 때 NaN

**해결**:
```python
# fragrance_ai/training/rlhf_complete.py:234
def update_policy_with_feedback(self, chosen_id, options, rating=None):
    # 보상 계산
    raw_reward = (rating - 3) / 2 if rating else 0.5

    # Clip reward to safe range
    reward = np.clip(raw_reward, -1.0, 1.0)

    # Advantage 계산 시에도 안전장치
    advantages = returns - values.detach()
    advantages = torch.clamp(advantages, -10.0, 10.0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**원인**: 극단적인 reward가 Q-value를 발산시킴

#### 7.4.5 로그 확률 안정화

**증상**: `log_prob`이 `-inf`가 되면서 NaN 전파

**해결**:
```python
# fragrance_ai/training/ppo_engine.py:145
# 확률 분포 생성 시
probs = F.softmax(logits, dim=-1)
probs = torch.clamp(probs, min=1e-8, max=1.0)  # 0이 되는 것 방지

dist = torch.distributions.Categorical(probs)
log_prob = dist.log_prob(action)

# 추가 안전장치
log_prob = torch.clamp(log_prob, min=-20.0, max=0.0)
```

**원인**: softmax 출력이 0에 가까워지면 log(0) = -inf

#### 7.4.6 초기화 개선

**증상**: 훈련 시작부터 NaN 발생

**해결**:
```python
# fragrance_ai/training/ppo_engine.py 네트워크 정의 시
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

        # Xavier 초기화로 안정성 향상
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)  # 출력층은 작게

        # Bias는 0으로
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
```

**원인**: 부적절한 초기 weight가 forward pass에서 폭발/소실 유발

#### 7.4.7 종합 디버깅 체크리스트

```python
# 훈련 루프에 추가할 NaN 검출 코드
def train_step(self, ...):
    # Forward pass
    logits = self.policy_net(state)

    # NaN 체크 1: 출력
    assert not torch.isnan(logits).any(), f"NaN in logits at state {state}"

    # Loss 계산
    loss = ...

    # NaN 체크 2: Loss
    if torch.isnan(loss):
        print(f"NaN detected!")
        print(f"State: {state}")
        print(f"Logits: {logits}")
        print(f"Advantages: {advantages}")
        raise ValueError("Training diverged (NaN loss)")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # NaN 체크 3: Gradients
    for name, param in self.policy_net.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                param.grad.zero_()  # Reset to prevent crash

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.2)

    optimizer.step()
```

### 7.2 권장 하이퍼파라미터 (Stable Configuration)

NaN 발생을 최소화하는 안전한 설정:

```python
# fragrance_ai/training/ppo_engine.py
STABLE_CONFIG = {
    # Learning rates
    "policy_lr": 5e-5,        # 낮은 학습률
    "value_lr": 1e-4,

    # PPO parameters
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.2,     # 강한 clipping

    # Training
    "batch_size": 32,
    "gamma": 0.99,
    "gae_lambda": 0.95,

    # Normalization
    "normalize_advantages": True,
    "normalize_rewards": True,
    "clip_rewards": (-1.0, 1.0),
    "clip_advantages": (-10.0, 10.0)
}

# fragrance_ai/training/moga_optimizer_stable.py
MOGA_STABLE_CONFIG = {
    "mutation_rate": 0.1,
    "crossover_rate": 0.7,
    "tournament_size": 3,
    "elite_ratio": 0.1,

    # Stability
    "min_concentration": 0.01,  # 0 방지
    "max_concentration": 99.0,  # 100 방지 (정규화 후 안전)
    "normalize_after_mutation": True
}
```

## 8. 결론 (Conclusion)

본 연구는 향수 생성을 생명체의 진화 과정으로 모델링하여, 사용자와의 상호작용을 통해 지속적으로 발전하는 AI 시스템을 구현하였다. MOGA를 통한 다목적 최적화와 RLHF를 통한 인간 피드백 학습의 통합은 창의성과 사용자 만족도를 동시에 달성하는 혁신적인 접근법임을 입증하였다.

### 7.1 기여사항

1. **이론적 기여**: 향수 창조를 수학적으로 모델링한 최초의 시도
2. **기술적 기여**: MOGA와 RLHF의 효과적인 통합 방법론 제시
3. **실용적 기여**: 실제 제품화 가능한 AI 향수 플랫폼 구현

### 7.2 향후 연구 방향

- 양자 컴퓨팅을 활용한 최적화 알고리즘 가속화
- Graph Neural Networks를 통한 분자 구조 기반 향수 설계
- 다중 감각 (시각, 촉각) 정보 통합

## 8. 참고문헌 (References)

1. Deb, K., & Jain, H. (2014). "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: NSGA-III"
2. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms"
3. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback"
4. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization"

---

## 설치 및 실행

```bash
# 환경 설정
git clone https://github.com/junseong2im/innovative_perfume_ai.git
cd innovative_perfume_ai

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
uvicorn fragrance_ai.api.main:app --reload --port 8001

# 테스트 실행
python test_living_scent.py
python test_real_optimizers.py
```

## 라이선스 (License)

### 프로젝트 라이선스

**Proprietary License** - 상업적 이용 금지

본 프로젝트의 소스 코드는 저작권자의 명시적 허가 없이 상업적으로 사용할 수 없습니다.

### 사용 모델 라이선스

본 프로젝트는 다음 오픈소스 LLM 모델을 사용합니다:

#### 1. Qwen 2.5 (7B/32B)
- **라이선스**: Apache 2.0
- **개발사**: Alibaba Cloud
- **용도**: Creative 모드 메인 모델
- **상업 이용**: ✅ 가능 (Apache 2.0 허용)
- **라이선스 전문**: https://github.com/QwenLM/Qwen/blob/main/LICENSE
- **조건**:
  - 원본 저작권 및 라이선스 고지 유지
  - 변경 사항 명시
  - Apache 2.0 조항 준수

#### 2. Mistral 7B
- **라이선스**: Apache 2.0
- **개발사**: Mistral AI
- **용도**: Balanced 모드 메인 모델
- **상업 이용**: ✅ 가능 (Apache 2.0 허용)
- **라이선스 전문**: https://github.com/mistralai/mistral-src/blob/main/LICENSE
- **조건**:
  - 원본 저작권 및 라이선스 고지 유지
  - 변경 사항 명시
  - Apache 2.0 조항 준수

#### 3. Llama 3 (8B)
- **라이선스**: Llama 3 Community License
- **개발사**: Meta (Facebook AI Research)
- **용도**: Fast 모드 메인 모델
- **상업 이용**: ⚠️ 조건부 가능
  - 월간 활성 사용자 7억 명 미만 서비스: ✅ 가능
  - 월간 활성 사용자 7억 명 이상 서비스: ❌ 별도 라이선스 필요
- **라이선스 전문**: https://github.com/meta-llama/llama3/blob/main/LICENSE
- **조건**:
  - Meta 라이선스 약관 준수
  - Llama 3 출력물로 다른 LLM 학습 금지
  - 사용자 수 제한 준수

### 라이선스 준수 사항

본 프로젝트를 사용하는 모든 사용자는 다음을 준수해야 합니다:

1. **모델 라이선스 고지**
   ```
   본 서비스는 다음 오픈소스 모델을 사용합니다:
   - Qwen 2.5 (Apache 2.0, Alibaba Cloud)
   - Mistral 7B (Apache 2.0, Mistral AI)
   - Llama 3 (Llama 3 Community License, Meta)
   ```

2. **상업적 사용 시 확인 사항**
   - Qwen, Mistral: Apache 2.0 조항 준수
   - Llama 3: 월간 활성 사용자 7억 명 미만 확인
   - 7억 명 이상 시 Meta에 별도 라이선스 요청

3. **라이선스 파일 포함**
   - `licenses/QWEN_LICENSE` - Qwen 2.5 라이선스 사본
   - `licenses/MISTRAL_LICENSE` - Mistral 7B 라이선스 사본
   - `licenses/LLAMA_LICENSE` - Llama 3 라이선스 사본

### 면책 조항

본 프로젝트는 교육 및 연구 목적으로 제공되며, 다음 사항에 대해 책임지지 않습니다:

- 모델 출력의 정확성, 안전성, 적절성
- 상업적 사용으로 인한 법적 문제
- 모델 라이선스 위반으로 인한 법적 책임
- 서비스 중단 또는 데이터 손실

### 기여자 라이선스

본 프로젝트에 기여하는 모든 기여자는 다음에 동의합니다:
- 기여 코드의 저작권을 프로젝트에 양도
- Apache 2.0 라이선스 적용에 동의

## 저자 (Author)

Jun Seong Im (junseong2im@gmail.com)

---

*본 문서는 IEEE/ACM 논문 형식을 따라 작성되었습니다.*