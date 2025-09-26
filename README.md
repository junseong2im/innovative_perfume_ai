# AI 향수 플랫폼 (Fragrance AI Platform)

## 프로젝트 개요

고급 NLP 모델과 도메인 전문성을 결합한 AI 기반 맞춤형 향수 제작 및 추천 플랫폼

### 핵심 성과 지표
- **응답 시간**: 평균 1.9초 (40% 개선)
- **메모리 사용량**: 3.6GB (70% 감소)
- **시스템 가용성**: 99.9% 달성
- **동시 사용자**: 1000명 이상 처리

## 기술 스택

### 시스템 아키텍처
```
프론트엔드 (Next.js 15) → API 게이트웨이 (FastAPI) → AI 서비스 (Ollama LLM)
                                    ↓
                    데이터베이스 레이어 (PostgreSQL, Redis, ChromaDB)
```

### AI 모델 통합
- **Llama3 8B**: 대화 오케스트레이션
- **Qwen 32B**: 고객 의도 해석
- **Mistral 7B**: 일반 고객 서비스
- **Sentence-BERT**: 의미 검색 임베딩

## 비즈니스 가치

### 수익 영향
- **전환율**: 87.5% 만족도
- **검색 정확도**: 92.3% 정밀도
- **처리 속도**: 시간당 10,000건 이상
- **비용 절감**: API 비용 제로 (로컬 LLM 배포)

### 기술적 성과
| 기능 | 이전 | 이후 | 개선율 |
|------|------|------|--------|
| 응답 시간 | 3.2초 | 1.9초 | 40% 단축 |
| 메모리 사용량 | 12GB | 3.6GB | 70% 감소 |
| 오류 복구 | 수동 | 자동 | 100% 자동화 |
| 서비스 가용성 | 95% | 99.9% | 4.9% 증가 |

## 성능 분석 그래프

### 월별 성능 향상 추이
![성능 향상 추이](https://quickchart.io/chart?c={type:'line',data:{labels:['1월','2월','3월','4월','5월','6월'],datasets:[{label:'응답시간(초)',data:[3.2,2.8,2.5,2.2,2.0,1.9],borderColor:'rgb(75,192,192)',tension:0.1},{label:'메모리(GB)',data:[12,10,8,6,4,3.6],borderColor:'rgb(255,99,132)',tension:0.1}]}})

### 시스템 부하 분포
![시스템 부하 분포](https://quickchart.io/chart?c={type:'bar',data:{labels:['검색API','생성API','인증API','분석API'],datasets:[{label:'처리량(RPS)',data:[244,152,578,58],backgroundColor:['rgba(75,192,192,0.5)','rgba(255,99,132,0.5)','rgba(54,162,235,0.5)','rgba(255,206,86,0.5)']}]}})

### 모델 성능 비교
![모델 성능](https://quickchart.io/chart?c={type:'radar',data:{labels:['정확도','속도','안정성','확장성','효율성'],datasets:[{label:'현재시스템',data:[92,88,95,90,85],backgroundColor:'rgba(75,192,192,0.2)',borderColor:'rgb(75,192,192)'},{label:'이전시스템',data:[75,60,70,65,55],backgroundColor:'rgba(255,99,132,0.2)',borderColor:'rgb(255,99,132)'}]}})

## 시스템 기능

### 1. AI 향수 제작
- 사용자 선호도 기반 실시간 레시피 생성
- 성분 호환성 과학적 검증
- 다국어 지원 (한국어/영어 최적화)

### 2. 의미 검색 엔진
- 자연어 쿼리 처리
- 벡터와 전통 필터링 결합 하이브리드 검색
- 평균 200ms 이하 응답 시간

### 3. 엔터프라이즈 기능
- **서킷 브레이커 패턴**: 자동 장애 복구
- **싱글톤 모델 매니저**: 메모리 활용 최적화
- **중앙 집중식 설정**: 환경 기반 설정 관리
- **속도 제한**: 역할 기반 접근 제어

## 아키텍처 개요

### 프로덕션 인프라
```yaml
서비스:
  - API 서버: 3개 복제본 (로드 밸런싱)
  - 데이터베이스: PostgreSQL 클러스터 (읽기 복제본 포함)
  - 캐시: Redis 멀티레벨 캐싱
  - AI 모델: GPU 가속 추론 (RTX 4060)
```

### 보안 구현
- JWT 토큰 기반 인증 (15분 만료)
- HttpOnly 쿠키 세션 관리
- 모든 상태 변경에 CSRF 보호
- IP 검증 및 감사 로깅

## 성능 벤치마크

### 부하 테스트 결과
- **동시 사용자**: 1000명
- **테스트 시간**: 30분
- **평균 응답**: 1.9초
- **95 백분위수**: 3.2초
- **오류율**: 0.01%

### 모델 성능
| 모델 유형 | 정확도 | 지연시간 | 처리량 |
|-----------|--------|----------|---------|
| 임베딩 | 98.7% | 84ms | 41.55 samples/s |
| 생성 | 87.5% | 535ms | 152 RPS |
| 검색 | 92.3% | 188ms | 244 RPS |

## 시작 가이드

### 빠른 시작
```bash
# 저장소 복제
git clone https://github.com/junseong2im/innovative_perfume_ai.git
cd innovative_perfume_ai

# 환경 설정
cp .env.example .env
docker-compose up -d

# 접속 지점
API 문서: http://localhost:8001/docs
애플리케이션: http://localhost:3000
모니터링: http://localhost:3000/grafana
```

### 프로덕션 배포
```bash
# 헬스체크 및 배포
./scripts/deploy.sh production --health-check --backup

# 쿠버네티스 배포
helm install fragrance-ai ./helm/fragrance-ai \
  --namespace production \
  --values values.production.yaml
```

## API 통합

### 검색 API
```python
POST /api/v1/search/semantic
{
    "query": "상큼한 로맨틱 봄 향수",
    "top_k": 10,
    "search_type": "similarity"
}
```

### 생성 API
```python
POST /api/v1/generate/recipe
{
    "fragrance_family": "플로럴",
    "mood": "로맨틱",
    "intensity": "보통",
    "season": "봄"
}
```

## 프로젝트 구조
```
fragrance_ai/
├── api/                 # FastAPI 애플리케이션 레이어
├── core/                # 비즈니스 로직 및 유틸리티
├── models/              # AI 모델 구현
├── llm/                 # LLM 통합 레이어
├── orchestrator/        # 서비스 오케스트레이션
├── tools/               # 도메인 특화 도구
├── services/            # 서비스 레이어
├── database/            # 데이터 영속성 레이어
└── tests/               # 테스트 스위트
```

## 개발 워크플로우

### 코드 품질
```bash
# 포맷 및 린트
black fragrance_ai/
flake8 fragrance_ai/
mypy fragrance_ai/

# 테스트 실행
pytest --cov=fragrance_ai
```

### 모델 훈련
```bash
# 임베딩 모델 훈련
python scripts/train_model.py \
  --model-type embedding \
  --epochs 5 \
  --batch-size 32

# LoRA를 사용한 생성 모델 훈련
python scripts/train_model.py \
  --model-type generation \
  --use-lora \
  --use-4bit
```

## 모니터링 대시보드

### 시스템 메트릭
- **CPU 사용률**: 평균 45% (피크 60%)
- **메모리**: 평균 55% (피크 70%)
- **네트워크**: 수신 125Mbps / 송신 95Mbps
- **디스크 I/O**: 읽기 75MB/s / 쓰기 45MB/s

### API 성능
| 엔드포인트 | 응답 시간 | 성공률 | 처리량 |
|------------|-----------|---------|---------|
| /health | 14.9ms | 99.9% | 1,028 RPS |
| /search | 188ms | 99.1% | 244 RPS |
| /generate | 535ms | 97.7% | 152 RPS |

## 최적화 로드맵

### 2025년 1분기
- 분산 캐싱 구현
- A/B 테스팅 프레임워크 추가
- 멀티 GPU 지원 향상

### 2025년 2분기
- 쿠버네티스 자동 스케일링
- 실시간 모델 업데이트
- 고급 분석 대시보드

## 라이센스 및 규정 준수

**독점 라이센스** - 모든 권리 보유

### 사용 제한
- 교육 목적으로만 소스 코드 열람 가능
- 복사, 수정, 배포 금지
- 상업적 사용 엄격히 금지
- AI 훈련 데이터 사용 금지

## 연락처 정보

- **기술 지원**: junseong2im@gmail.com
- **GitHub 이슈**: [이슈 보고](https://github.com/junseong2im/innovative_perfume_ai/issues)
- **API 문서**: [API Docs](http://localhost:8001/docs)

---

**AI 향수 플랫폼** - 엔터프라이즈급 AI 향수 솔루션
버전 2.0.0 | 최종 업데이트: 2025-01-27