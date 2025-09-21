# Fragrance AI - Claude 개발 가이드

## 프로젝트 개요

**Fragrance AI**는 최신 AI 기술을 활용한 향수 레시피 생성 및 검색 플랫폼입니다.

### 핵심 기능
- AI 기반 향수 레시피 자동 생성
- 의미 기반 향수 검색 시스템
- 실시간 성능 모니터링
- 하이브리드 검색 (벡터 + 전통적 필터링)
- RESTful API 제공

## 프로젝트 구조

```
fragrance_ai/
├── fragrance_ai/                    # 메인 애플리케이션 패키지
│   ├── api/                         # FastAPI 웹 애플리케이션 레이어
│   │   ├── main.py                  # 메인 FastAPI 애플리케이션
│   │   ├── routes/                  # API 엔드포인트 라우터
│   │   │   ├── admin.py             # 관리자 API
│   │   │   ├── generation.py        # 향수 레시피 생성 API
│   │   │   ├── monitoring.py        # 모니터링 API
│   │   │   ├── search.py            # 검색 API
│   │   │   └── training.py          # 모델 훈련 API
│   │   ├── schemas.py               # Pydantic 데이터 스키마
│   │   ├── middleware.py            # HTTP 미들웨어
│   │   └── auth.py                  # 인증/인가 시스템
│   ├── core/                        # 핵심 비즈니스 로직
│   │   ├── config.py                # 애플리케이션 설정 관리
│   │   ├── vector_store.py          # 벡터 데이터베이스 관리
│   │   ├── auth.py                  # 핵심 인증 시스템
│   │   ├── advanced_logging.py      # 고급 로깅 시스템
│   │   ├── comprehensive_monitoring.py # 종합 모니터링
│   │   └── intelligent_cache.py     # 지능형 캐싱 시스템
│   ├── models/                      # AI 모델 레이어
│   │   ├── embedding.py             # 임베딩 모델 (Sentence-BERT)
│   │   ├── generator.py             # 향수 레시피 생성 모델
│   │   └── base.py                  # 모델 베이스 클래스
│   ├── services/                    # 서비스 레이어
│   │   ├── search_service.py        # 검색 비즈니스 로직
│   │   ├── generation_service.py    # 생성 비즈니스 로직
│   │   ├── monitoring_service.py    # 모니터링 서비스
│   │   └── cache_service.py         # 캐싱 서비스
│   ├── training/                    # 모델 훈련 시스템
│   │   ├── peft_trainer.py          # PEFT (LoRA) 훈련
│   │   └── advanced_optimizer.py    # 고급 옵티마이저
│   ├── evaluation/                  # 모델 평가 시스템
│   │   └── metrics.py               # 평가 메트릭
│   ├── database/                    # 데이터베이스 레이어
│   │   ├── models.py                # SQLAlchemy ORM 모델
│   │   └── base.py                  # 데이터베이스 연결 관리
│   └── admin/                       # 관리자 인터페이스
│       ├── auth.py                  # 관리자 인증
│       └── dashboard.py             # 관리자 대시보드
├── scripts/                         # 유틸리티 스크립트
│   ├── train_model.py               # 모델 훈련 스크립트
│   ├── evaluate_model.py            # 모델 평가 스크립트
│   ├── deploy.sh                    # 배포 스크립트
│   └── deploy_advanced.py           # 고급 배포 스크립트
├── configs/                         # 설정 파일
│   └── optimizer_examples.json      # 옵티마이저 설정 예제
├── tests/                           # 테스트 코드
│   ├── performance/                 # 성능 테스트
│   └── test_comprehensive_auth.py   # 종합 인증 테스트
├── docker/                          # Docker 설정
├── data/                            # 데이터 디렉토리
├── models/                          # 훈련된 모델 저장소
├── logs/                            # 로그 파일
└── performance_graphs/              # 성능 그래프
```

## 개발 환경 설정

### 필수 요구사항
- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU (권장, AI 모델 가속화용)

### 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# 주요 환경 변수
DATABASE_URL=postgresql://user:password@localhost/fragrance_ai
REDIS_URL=redis://localhost:6379
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

### 의존성 설치
```bash
# 개발 환경
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 프로덕션 환경
pip install -r requirements-prod.txt
```

### 데이터베이스 마이그레이션
```bash
# 마이그레이션 실행
alembic upgrade head

# 새 마이그레이션 생성
alembic revision --autogenerate -m "description"
```

## 개발 워크플로우

### 1. 개발 서버 실행
```bash
# FastAPI 개발 서버
uvicorn fragrance_ai.api.main:app --reload --host 0.0.0.0 --port 8000

# 또는 Docker Compose 사용
docker-compose up -d
```

### 2. 코드 품질 검사
```bash
# 린팅 및 포맷팅
black fragrance_ai/
isort fragrance_ai/
flake8 fragrance_ai/
mypy fragrance_ai/
```

### 3. 테스트 실행
```bash
# 전체 테스트
pytest

# 커버리지 포함
pytest --cov=fragrance_ai --cov-report=html

# 성능 테스트
python run_performance_tests.py
```

## API 사용법

### 주요 엔드포인트

#### 1. 향수 검색
```bash
POST /api/v1/search/semantic
{
    "query": "상큼하고 로맨틱한 봄 향수",
    "top_k": 10,
    "search_type": "similarity"
}
```

#### 2. 레시피 생성
```bash
POST /api/v1/generate/recipe
{
    "fragrance_family": "floral",
    "mood": "romantic",
    "intensity": "moderate",
    "gender": "feminine",
    "season": "spring"
}
```

#### 3. 배치 생성
```bash
POST /api/v1/generate/batch
{
    "requests": [
        {
            "fragrance_family": "citrus",
            "mood": "fresh",
            "intensity": "light"
        }
    ]
}
```

### 모니터링 엔드포인트
- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics`
- **API Docs**: `GET /docs`

## 모델 훈련

### 임베딩 모델 훈련
```bash
python scripts/train_model.py \
    --model-type embedding \
    --data-path ./data/training/embedding_data.json \
    --output-dir ./checkpoints/embedding \
    --epochs 5 \
    --batch-size 32 \
    --wandb-project fragrance-ai
```

### 생성 모델 훈련 (LoRA/PEFT)
```bash
python scripts/train_model.py \
    --model-type generation \
    --data-path ./data/training/generation_data.json \
    --output-dir ./checkpoints/generation \
    --use-lora \
    --use-4bit \
    --epochs 3 \
    --batch-size 4
```

## 배포

### 개발 환경
```bash
./scripts/deploy.sh development --health-check
```

### 프로덕션 환경
```bash
./scripts/deploy.sh production --backup --health-check --cleanup
```

### Docker 배포
```bash
# 전체 스택 배포
docker-compose -f docker-compose.production.yml up -d

# 스케일링
docker-compose -f docker-compose.scale.yml up -d
```

## 성능 최적화

### 현재 성능 지표
- **검색 응답시간**: < 200ms (평균)
- **레시피 생성시간**: < 3초 (기본)
- **동시 사용자**: 1000+ concurrent users
- **처리량**: 10,000+ requests/hour

### 최적화 권장사항
1. **임베딩 최적화**: 배치 크기 32 사용
2. **캐싱 전략**: Redis 멀티레벨 캐싱
3. **모델 경량화**: 4bit 양자화 적용
4. **비동기 처리**: FastAPI async 최대 활용

## 보안 설정

### 인증/인가
- JWT 토큰 기반 인증
- Role-based Access Control (RBAC)
- API Rate Limiting
- CORS 설정

### 보안 모범 사례
- 환경 변수로 시크릿 관리
- HTTPS 강제 (프로덕션)
- 입력 검증 및 새니타이제이션
- 로그 민감정보 마스킹

## 모니터링 및 로깅

### 접속 정보
- **API 문서**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Flower (Celery)**: http://localhost:5555

### 주요 메트릭
- API 응답 시간 및 처리량
- 모델 추론 성능
- 데이터베이스 성능
- 캐시 히트율
- 시스템 리소스 사용률

## 트러블슈팅

### 자주 발생하는 문제

#### 1. 모델 로딩 실패
```bash
# GPU 메모리 부족
export CUDA_VISIBLE_DEVICES=0
# 또는 4bit 양자화 활성화
```

#### 2. 데이터베이스 연결 오류
```bash
# 연결 확인
psql -h localhost -U user -d fragrance_ai

# 마이그레이션 상태 확인
alembic current
```

#### 3. Redis 연결 문제
```bash
# Redis 서비스 확인
redis-cli ping

# 캐시 클리어
redis-cli flushall
```

## 기여 가이드라인

### 코딩 컨벤션
- **Python**: PEP 8 준수
- **타입 힌트**: 모든 함수에 타입 힌트 필수
- **문서화**: Docstring 필수 (Google 스타일)
- **테스트**: 새 기능에 대한 테스트 코드 필수

### 커밋 메시지 형식
```
type(scope): description

feat(api): add fragrance recommendation endpoint
fix(db): resolve connection pool issue
docs(readme): update installation instructions
test(search): add unit tests for semantic search
```

### Pull Request 체크리스트
- [ ] 코드 품질 검사 통과 (black, flake8, mypy)
- [ ] 테스트 작성 및 통과
- [ ] 문서 업데이트
- [ ] 성능 영향 검토
- [ ] 보안 검토

## 라이센스 및 제한사항

**독점 라이센스(Proprietary License)** - 자세한 내용은 [LICENSE](LICENSE) 참조

### 중요 제한사항
- 개인 학습 목적으로만 열람 가능
- 복사, 수정, 배포 금지
- 상업적 이용 금지
- AI 학습 데이터 사용 금지

## 지원 및 문의

- **이슈 리포팅**: GitHub Issues
- **기능 요청**: GitHub Discussions
- **이메일**: junseong2im@gmail.com

---

**개발 시 참고사항**
- 이 문서는 프로젝트 진행에 따라 지속적으로 업데이트됩니다
- 새로운 기능 추가 시 반드시 문서를 함께 업데이트해야 합니다
- 성능 테스트 결과는 자동으로 README.md에 반영됩니다