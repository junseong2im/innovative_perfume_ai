# Fragrance AI - 프로젝트 구조 가이드

## 개요

이 문서는 Fragrance AI 프로젝트의 전체 구조와 각 디렉토리/파일의 역할을 설명합니다.

## 전체 디렉토리 구조

```
fragrance_ai/
├── .benchmarks/              # 성능 벤치마크 결과
├── .claude/                  # Claude Code 설정
│   └── settings.local.json   # 로컬 Claude 설정
├── .git/                     # Git 버전 관리
├── .github/                  # GitHub 워크플로우 및 템플릿
├── .pytest_cache/            # pytest 캐시
├── airflow/                  # Apache Airflow DAGs
│   └── dags/
│       └── fragrance_data_pipeline.py
├── alembic/                  # 데이터베이스 마이그레이션
│   └── versions/
├── checkpoints/              # 훈련된 모델 체크포인트
├── configs/                  # 설정 파일들
│   └── optimizer_examples.json
├── data/                     # 데이터 저장소
├── docker/                   # Docker 관련 파일
├── fragrance_ai/             # 메인 애플리케이션 패키지
├── frontend/                 # 프론트엔드 (React)
├── haproxy/                  # HAProxy 로드 밸런서 설정
├── logs/                     # 로그 파일
├── migrations/               # 추가 마이그레이션
├── models/                   # 저장된 AI 모델
├── monitoring/               # 모니터링 설정 (Prometheus, Grafana)
├── nginx/                    # Nginx 설정
├── performance_graphs/       # 성능 테스트 결과 그래프
├── scripts/                  # 유틸리티 스크립트
└── tests/                    # 테스트 코드
```

## 메인 애플리케이션 구조 (`fragrance_ai/`)

### API 레이어 (`fragrance_ai/api/`)
웹 API 및 HTTP 인터페이스를 담당하는 레이어

```
api/
├── main.py                   # FastAPI 메인 애플리케이션
├── main_legacy.py            # 레거시 API (호환성 유지)
├── auth.py                   # API 인증 시스템
├── user_auth.py              # 사용자 인증 라우터
├── dependencies.py           # API 의존성 주입
├── error_handlers.py         # 글로벌 에러 핸들링
├── external_services.py      # 외부 서비스 통합
├── middleware.py             # HTTP 미들웨어
├── schemas.py                # Pydantic 데이터 스키마
├── versioning.py             # API 버전 관리
└── routes/                   # API 엔드포인트 라우터
    ├── __init__.py
    ├── admin.py              # 관리자 API
    ├── generation.py         # 향수 레시피 생성 API
    ├── monitoring.py         # 모니터링 API
    ├── search.py             # 검색 API
    └── training.py           # 모델 훈련 API
```

### 핵심 비즈니스 로직 (`fragrance_ai/core/`)
애플리케이션의 핵심 기능을 담당하는 레이어

```
core/
├── config.py                 # 애플리케이션 설정 관리
├── vector_store.py           # 벡터 데이터베이스 관리
├── auth.py                   # 핵심 인증 시스템
├── advanced_logging.py       # 고급 로깅 시스템
├── comprehensive_monitoring.py # 종합 모니터링
└── intelligent_cache.py      # 지능형 캐싱 시스템
```

### AI 모델 레이어 (`fragrance_ai/models/`)
AI 모델 정의 및 관리

```
models/
├── base.py                   # 모델 베이스 클래스
├── embedding.py              # 임베딩 모델 (Sentence-BERT)
└── generator.py              # 향수 레시피 생성 모델
```

### 서비스 레이어 (`fragrance_ai/services/`)
비즈니스 로직 구현

```
services/
├── search_service.py         # 검색 비즈니스 로직
├── generation_service.py     # 생성 비즈니스 로직
├── monitoring_service.py     # 모니터링 서비스
└── cache_service.py          # 캐싱 서비스
```

### 모델 훈련 시스템 (`fragrance_ai/training/`)
AI 모델 훈련 관련 코드

```
training/
├── peft_trainer.py           # PEFT (LoRA) 훈련
└── advanced_optimizer.py     # 고급 옵티마이저
```

### 평가 시스템 (`fragrance_ai/evaluation/`)
모델 성능 평가

```
evaluation/
└── metrics.py                # 평가 메트릭
```

### 데이터베이스 레이어 (`fragrance_ai/database/`)
데이터 영속성 관리

```
database/
├── models.py                 # SQLAlchemy ORM 모델
└── base.py                   # 데이터베이스 연결 관리
```

### 관리자 인터페이스 (`fragrance_ai/admin/`)
시스템 관리 기능

```
admin/
├── auth.py                   # 관리자 인증
└── dashboard.py              # 관리자 대시보드
```

## 스크립트 및 유틸리티 (`scripts/`)

```
scripts/
├── train_model.py            # 모델 훈련 스크립트
├── evaluate_model.py         # 모델 평가 스크립트
├── deploy.sh                 # 배포 스크립트
└── deploy_advanced.py        # 고급 배포 스크립트
```

## 테스트 코드 (`tests/`)

```
tests/
├── performance/              # 성능 테스트
└── test_comprehensive_auth.py # 종합 인증 테스트
```

## 인프라 및 배포

### Docker 설정 (`docker/`)
컨테이너화 관련 파일들

### 모니터링 (`monitoring/`)
Prometheus, Grafana 설정 파일들

### 로드 밸런싱 (`haproxy/`, `nginx/`)
트래픽 분산 및 리버스 프록시 설정

## 설정 파일들

### 환경 설정
- `.env` - 개발 환경 변수
- `.env.example` - 환경 변수 예제
- `.env.production` - 프로덕션 환경 변수
- `.env.production.template` - 프로덕션 환경 변수 템플릿

### Docker 설정
- `Dockerfile` - 메인 Docker 이미지
- `docker-compose.yml` - 개발용 Docker Compose
- `docker-compose.production.yml` - 프로덕션용 Docker Compose
- `docker-compose.scale.yml` - 스케일링용 Docker Compose

### Python 설정
- `requirements.txt` - 기본 Python 의존성
- `requirements-dev.txt` - 개발용 의존성
- `requirements-prod.txt` - 프로덕션용 의존성
- `requirements-minimal.txt` - 최소 의존성
- `pyproject.toml` - Python 프로젝트 설정
- `setup.py` - 패키지 설정

### 기타 설정
- `pytest.ini` - pytest 설정
- `alembic.ini` - 데이터베이스 마이그레이션 설정
- `.gitignore` - Git 무시 파일 목록

## 아키텍처 패턴

### 레이어드 아키텍처
1. **API Layer** - HTTP 인터페이스 및 라우팅
2. **Service Layer** - 비즈니스 로직
3. **Model Layer** - AI 모델 및 데이터 모델
4. **Data Layer** - 데이터 영속성

### 의존성 방향
- 상위 레이어는 하위 레이어에만 의존
- 하위 레이어는 상위 레이어를 알지 못함
- 인터페이스를 통한 의존성 역전

### 모듈화
- 각 모듈은 단일 책임 원칙 준수
- 느슨한 결합, 높은 응집도
- 재사용 가능한 컴포넌트 설계

## 코딩 컨벤션

### 파일 명명 규칙
- Python 파일: `snake_case.py`
- 클래스명: `PascalCase`
- 함수/변수명: `snake_case`
- 상수명: `UPPER_SNAKE_CASE`

### 디렉토리 구조 원칙
- 기능별 그룹화
- 레이어별 분리
- 의존성 방향 고려
- 확장성 고려

### 문서화 표준
- 모든 모듈에 docstring 필수
- 복잡한 함수는 상세 설명 추가
- 타입 힌트 필수
- 주석은 한국어 또는 영어

## 개발 가이드라인

### 새 기능 추가 시
1. 적절한 레이어에 코드 배치
2. 테스트 코드 작성
3. 문서 업데이트
4. 설정 파일 확인

### 성능 고려사항
- 비동기 처리 활용
- 캐싱 전략 적용
- 데이터베이스 쿼리 최적화
- AI 모델 배치 처리

### 보안 고려사항
- 입력 검증 및 새니타이제이션
- 인증/인가 체크
- 민감 정보 암호화
- HTTPS 사용

## 배포 및 운영

### 환경별 설정
- Development: 로컬 개발 환경
- Staging: 테스트 환경
- Production: 실제 운영 환경

### 모니터링 포인트
- API 응답 시간
- 에러율
- 시스템 리소스 사용률
- AI 모델 성능 지표

### 백업 전략
- 데이터베이스 정기 백업
- 모델 체크포인트 보관
- 설정 파일 버전 관리
- 로그 파일 아카이빙

---

이 구조는 확장 가능하고 유지보수가 용이하도록 설계되었습니다. 새로운 기능 추가나 기존 기능 수정 시 이 가이드를 참조하여 일관성을 유지해주세요.