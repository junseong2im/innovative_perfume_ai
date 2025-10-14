# Advanced AI System Architecture
**Fragrance AI - 고급 기능 통합 설계**

생성일: 2025-10-14

---

## 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    Artisan Cloud Hub (IPFS + Redis)             │
│            학습 데이터·피드백 로그 분산 관리                        │
└──────────────────┬──────────────────────────────────────────────┘
                   │
    ┌──────────────┴──────────────┬──────────────────────────┐
    │                             │                          │
┌───▼────────────┐    ┌──────────▼──────────┐    ┌─────────▼────────┐
│  향수 조합 AI   │◄──►│   감정 인식 AI      │◄──►│  IFRA 규제 AI    │
│  (Qwen RLHF)   │    │  (Multi-Agent)      │    │  (Safety)        │
└───┬────────────┘    └──────────┬──────────┘    └─────────┬────────┘
    │                             │                          │
    │         Multi-Agent Evolution System                   │
    │                             │                          │
    └──────────────┬──────────────┴──────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  MOGA ↔ RL Hybrid   │
        │  Exploration/        │
        │  Exploitation Loop   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Policy Distillation  │
        │  PPO → Llama Teacher │
        └─────────────────────┘
```

---

## 1. Qwen RLHF 연결

### 목적
Qwen2.5 API를 내부 Fine-tuner와 연결하여 사용자 피드백을 RLHF(Reinforcement Learning from Human Feedback)로 직접 반영

### 아키텍처

```python
User Feedback → Redis Queue → RLHF Trainer → Qwen Fine-tuner → Updated Model
                      ↓
              Reward Modeling
                      ↓
              PPO Policy Update
```

### 주요 컴포넌트

#### 1.1 피드백 수집 시스템
- **Endpoint**: `/feedback/submit`
- **저장소**: Redis Stream
- **데이터 형식**:
  ```json
  {
    "recipe_id": "uuid",
    "user_rating": 4.5,
    "feedback_text": "향이 너무 강함",
    "preferred_notes": ["citrus", "woody"],
    "timestamp": "2025-10-14T23:30:00Z"
  }
  ```

#### 1.2 Reward Model
- **입력**: Recipe + User Feedback
- **출력**: Reward Score (-1.0 ~ 1.0)
- **학습 방식**: Binary Classification (좋음/나쁨) → Reward

#### 1.3 Fine-tuner Integration
- **모델**: Qwen2.5-32B-Instruct
- **방법**: LoRA (Low-Rank Adaptation)
- **업데이트 주기**: 1000 피드백 누적 시
- **GPU 메모리**: ~18GB VRAM 필요

### 구현 단계
1. ✅ 피드백 수집 API 엔드포인트
2. ⏳ Reward Model 학습 파이프라인
3. ⏳ Qwen LoRA Fine-tuning 스크립트
4. ⏳ 모델 Hot-reload 메커니즘

---

## 2. MOGA ↔ RL Hybrid Loop

### 목적
Exploration(LLM)과 Exploitation(RL) 루프를 자동으로 전환하여 최적의 향수 레시피 생성

### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                 Hybrid Controller                    │
│  - Exploration Budget Tracker                        │
│  - Performance Metrics Monitor                       │
│  - Mode Switching Logic                              │
└──────────────┬────────────────────────────────┬─────┘
               │                                │
      ┌────────▼──────────┐          ┌────────▼──────────┐
      │  MOGA (Exploration)│          │  RL (Exploitation) │
      │  - LLM Generation  │          │  - PPO Policy      │
      │  - Genetic Ops     │◄────────►│  - Value Network   │
      │  - Diversity Max   │          │  - Reward Max      │
      └───────────────────┘          └───────────────────┘
```

### 전환 로직

#### 2.1 Exploration Mode (MOGA)
**조건**:
- 초기 학습 단계 (< 1000 episodes)
- 최근 100 에피소드의 reward 표준편차 < 0.5
- Diversity metric < 0.6

**동작**:
- LLM으로 새로운 레시피 생성
- Crossover/Mutation으로 다양성 증가
- 목표: 탐색 공간 확장

#### 2.2 Exploitation Mode (RL)
**조건**:
- 충분한 탐색 완료 (> 1000 episodes)
- 최근 100 에피소드의 reward 표준편차 > 0.5
- Diversity metric > 0.6

**동작**:
- PPO 정책으로 최적 레시피 선택
- Value Network로 가치 평가
- 목표: 보상 최대화

#### 2.3 자동 전환 메트릭
```python
exploration_budget = 0.3  # 30% exploration
epsilon = max(0.1, 1.0 - episode / max_episodes)

if performance_plateau() or diversity_low():
    mode = "exploration"
elif exploration_budget_used > 0.3:
    mode = "exploitation"
```

### 구현 단계
1. ⏳ Hybrid Controller 구현
2. ⏳ Mode Switching Logic
3. ⏳ Performance Monitoring Dashboard
4. ⏳ A/B Test Framework

---

## 3. Policy Distillation

### 목적
학습된 PPO 정책을 Llama 기반 Teacher 모델로 distill하여 더 작고 빠른 모델 생성

### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│         Teacher Model (PPO Policy)                   │
│  - Actor Network (32B params)                        │
│  - Critic Network (Value estimation)                 │
│  - Trained on 100K+ episodes                         │
└──────────────┬──────────────────────────────────────┘
               │ Knowledge Transfer
               │
┌──────────────▼──────────────────────────────────────┐
│      Student Model (Llama-3.1-8B-Instruct)          │
│  - Smaller architecture (8B params)                  │
│  - Mimics teacher's action distribution             │
│  - 4x faster inference                               │
└─────────────────────────────────────────────────────┘
```

### Distillation 방법

#### 3.1 Behavioral Cloning
- **데이터**: Teacher의 (state, action) 페어
- **손실 함수**: Cross-Entropy Loss
- **장점**: 간단, 빠른 학습

#### 3.2 KL Divergence Minimization
- **손실 함수**:
  ```python
  L = KL(Teacher_Policy || Student_Policy) + λ * Task_Loss
  ```
- **장점**: 확률 분포 전체를 학습

#### 3.3 DAgger (Dataset Aggregation)
- **방법**: Student 실행 → Teacher 라벨링 → 재학습
- **장점**: Distribution shift 방지

### 성능 목표
| 메트릭 | Teacher (PPO) | Student (Llama) | 개선 |
|--------|---------------|-----------------|------|
| Inference Time | 2.5s | 0.6s | **4.2x** |
| VRAM Usage | 16GB | 5GB | **3.2x** |
| Reward | 22.5 | 20.1 | -10.7% |

### 구현 단계
1. ⏳ Teacher Policy Data Collection (100K samples)
2. ⏳ Llama Fine-tuning Pipeline
3. ⏳ KL Divergence Loss Implementation
4. ⏳ Performance Benchmarking

---

## 4. Multi-Agent Evolution

### 목적
향수 조합 AI, 감정 인식 AI, IFRA 규제 AI 3개가 협업하여 최적의 향수 레시피 생성

### 아키텍처

```
        ┌────────────────────────────────┐
        │   Orchestrator (Meta-Agent)     │
        │  - Task Allocation               │
        │  - Conflict Resolution           │
        │  - Final Decision Making         │
        └─────────┬──────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
┌─────▼─────┐ ┌──▼──────┐ ┌─▼─────────┐
│ Fragrance │ │ Emotion │ │   IFRA    │
│ Composer  │ │ Analyzer│ │ Validator │
│  (Qwen)   │ │(Mistral)│ │  (Llama)  │
└───────────┘ └─────────┘ └───────────┘
```

### Agent 역할

#### 4.1 Fragrance Composer (Qwen2.5-32B)
**역할**: 창의적인 향수 레시피 생성
**입력**: 사용자 프롬프트, 스타일, 계절
**출력**: 원료 조합 (Top/Middle/Base notes)
**목표**: 예술성, 독창성

#### 4.2 Emotion Analyzer (Mistral-7B)
**역할**: 향수와 감정 매칭
**입력**: 레시피, 목표 감정
**출력**: 감정 스코어 (Joy, Calm, Energy, Romance)
**목표**: 감정 표현 정확도

#### 4.3 IFRA Validator (Llama-3.1-8B)
**역할**: 안전 규제 검증
**입력**: 레시피, 농도
**출력**: 규제 준수 여부, 권장 수정사항
**목표**: 100% 안전성 준수

### 협업 프로세스

```python
# 1단계: 병렬 생성
fragrance = composer.generate(prompt)
emotion = analyzer.predict(fragrance, target_emotion)
safety = validator.check(fragrance)

# 2단계: 협상 (Negotiation)
while not consensus():
    if safety.violations:
        fragrance = composer.adjust(safety.suggestions)
    if emotion.score < threshold:
        fragrance = composer.enhance_emotion(emotion.target)

# 3단계: 최종 결정
final_recipe = orchestrator.merge([
    fragrance,
    emotion.recommendations,
    safety.approved_ingredients
])
```

### 구현 단계
1. ⏳ 3개 Agent API 통합
2. ⏳ Orchestrator Logic
3. ⏳ Negotiation Protocol
4. ⏳ Conflict Resolution Algorithm

---

## 5. Artisan Cloud Hub

### 목적
학습 데이터와 피드백 로그를 IPFS + Redis로 분산 관리하여 확장성과 데이터 무결성 보장

### 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              Application Layer                       │
│  - FastAPI Endpoints                                 │
│  - Data Upload/Download                              │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│           Data Management Layer                      │
│  - Deduplication (Hash-based)                        │
│  - Compression (zstd)                                │
│  - Encryption (AES-256)                              │
└──────────────┬──────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
┌─────▼─────┐    ┌─────▼─────┐
│   Redis    │    │   IPFS    │
│  (Metadata)│    │  (Content)│
│  - Fast    │    │ - Immut.  │
│  - TTL     │    │ - Distrib.│
└───────────┘    └───────────┘
```

### 데이터 흐름

#### 5.1 업로드 프로세스
```python
# 1. 데이터 전처리
data_hash = sha256(data)
compressed = zstd.compress(data)
encrypted = aes256.encrypt(compressed, key)

# 2. IPFS 저장
cid = ipfs.add(encrypted)

# 3. Redis 메타데이터 저장
redis.hset(f"data:{data_hash}", {
    "cid": cid,
    "size": len(data),
    "type": "feedback",
    "timestamp": now(),
    "tags": ["rlhf", "qwen"]
})
redis.expire(f"data:{data_hash}", 86400 * 30)  # 30일 TTL
```

#### 5.2 다운로드 프로세스
```python
# 1. Redis에서 메타데이터 조회
metadata = redis.hgetall(f"data:{data_hash}")

# 2. IPFS에서 콘텐츠 가져오기
encrypted = ipfs.cat(metadata["cid"])

# 3. 복호화 및 압축 해제
compressed = aes256.decrypt(encrypted, key)
data = zstd.decompress(compressed)
```

### 스토리지 최적화

| 데이터 타입 | 저장 위치 | 압축률 | TTL |
|------------|----------|--------|-----|
| 피드백 로그 | IPFS | 80% | 30일 |
| 학습 체크포인트 | IPFS | - | 90일 |
| 메트릭 (실시간) | Redis | - | 1시간 |
| 세션 데이터 | Redis | - | 24시간 |

### 구현 단계
1. ⏳ IPFS Node 설치 및 설정
2. ⏳ Redis Cluster 구성
3. ⏳ Upload/Download API 구현
4. ⏳ 암호화/압축 파이프라인
5. ⏳ 자동 백업 스크립트

---

## 통합 로드맵

### Phase 1: 기반 구축 (1-2주)
- [ ] Redis Queue 설정
- [ ] IPFS 노드 설치
- [ ] 기본 API 엔드포인트
- [ ] 메트릭 수집 파이프라인

### Phase 2: 단일 기능 구현 (2-3주)
- [ ] Qwen RLHF 피드백 수집
- [ ] Hybrid Loop Controller
- [ ] Multi-Agent 기본 통합

### Phase 3: 고급 기능 (3-4주)
- [ ] Policy Distillation 파이프라인
- [ ] Artisan Cloud Hub 완성
- [ ] 성능 최적화

### Phase 4: 운영 및 모니터링 (계속)
- [ ] Grafana 대시보드 확장
- [ ] A/B 테스트 프레임워크
- [ ] 자동 스케일링

---

## 참고 문서

- `RLHF_ADVANCED.md` - RLHF 구현 상세
- `DEPLOYMENT_GUIDE.md` - 배포 가이드
- `OPERATIONS_GUIDE.md` - 운영 가이드
- `RUNBOOK.md` - 장애 대응

---

*문서 버전: 1.0*
*최종 업데이트: 2025-10-14*
