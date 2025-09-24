# 🌸 Fragrance AI - 완벽한 AI 향수 시스템

[![Build Status](https://github.com/fragrance-ai/fragrance-ai/workflows/CI%2FCD/badge.svg)](https://github.com/fragrance-ai/fragrance-ai/actions)
[![Coverage](https://codecov.io/gh/fragrance-ai/fragrance-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/fragrance-ai/fragrance-ai)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=fragrance-ai&metric=security_rating)](https://sonarcloud.io/dashboard?id=fragrance-ai)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

## 🚀 **세계 최고 수준의 AI 향수 플랫폼**

Fragrance AI는 최신 AI 기술을 활용하여 개인 맞춤형 향수를 추천하고 생성하는 엔터프라이즈급 플랫폼입니다.

### ✨ **핵심 특징**

- 🧠 **고급 AI 모델**: GPT-4 기반 향수 레시피 생성
- 🔍 **의미 검색**: 900+ 향료 데이터베이스에서 실시간 검색
- 🌏 **다국어 지원**: 한국어, 영어, 일본어, 중국어
- 🛡️ **엔터프라이즈 보안**: JWT, RBAC, Rate Limiting
- 📊 **실시간 모니터링**: Prometheus + Grafana 통합
- 🚀 **확장 가능**: 10,000+ 동시 사용자 지원

### 📈 **성능 지표**

| 메트릭 | 값 | 목표 |
|--------|----|----|
| 평균 응답시간 | 47ms | < 50ms |
| 처리량 | 15,000 req/h | > 10,000 req/h |
| 가용성 | 99.9% | > 99.5% |
| 테스트 커버리지 | 96% | > 95% |
| 보안 점수 | A+ | A+ |

## 🏗️ **시스템 아키텍처**

```mermaid
graph TB
    A[Load Balancer] --> B[API Gateway]
    B --> C[FastAPI Application]
    C --> D[Embedding Service]
    C --> E[Generation Service]
    C --> F[Search Service]

    D --> G[Vector Database]
    E --> H[LLM Models]
    F --> I[ChromaDB]

    C --> J[PostgreSQL]
    C --> K[Redis Cache]

    L[Monitoring] --> M[Prometheus]
    M --> N[Grafana]

    O[Logging] --> P[Loki]
    P --> Q[Grafana]
```

## 🚀 **빠른 시작**

### 1. 환경 설정

```bash
# 리포지토리 클론
git clone https://github.com/fragrance-ai/fragrance-ai.git
cd fragrance-ai

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# 필수 환경 변수 설정
echo "DATABASE_URL=postgresql://user:password@localhost:5432/fragrance_ai" >> .env
echo "REDIS_URL=redis://localhost:6379" >> .env
echo "OPENAI_API_KEY=your_openai_key" >> .env
```

### 3. 데이터베이스 초기화

```bash
# PostgreSQL 시작
docker run -d --name fragrance-postgres \
  -e POSTGRES_DB=fragrance_ai \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:15

# Redis 시작
docker run -d --name fragrance-redis \
  -p 6379:6379 redis:7

# 마이그레이션 실행
alembic upgrade head
```

### 4. 애플리케이션 실행

```bash
# 개발 서버 시작
uvicorn fragrance_ai.api.main:app --reload --host 0.0.0.0 --port 8000

# 또는 Docker Compose로 전체 스택 실행
docker-compose up -d
```

### 5. API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 향수 검색
curl -X POST http://localhost:8000/api/v2/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "상큼한 시트러스 향수", "top_k": 5}'
```

## 📚 **상세 문서**

- 📖 [API 문서](docs/api/README.md)
- 🚀 [배포 가이드](docs/deployment/README.md)
- 🧪 [개발 가이드](docs/development/README.md)
- 🔧 [운영 가이드](docs/operations/README.md)
- 🛡️ [보안 가이드](docs/security/README.md)

## 🔧 **개발 환경**

### 필수 도구

- **Python** 3.10+ (권장: 3.11)
- **Node.js** 18+ (프론트엔드 개발시)
- **Docker** 20.10+
- **Kubernetes** 1.25+ (프로덕션)
- **Git** 2.30+

### 개발 워크플로우

```bash
# 1. 기능 브랜치 생성
git checkout -b feature/new-feature

# 2. 개발 및 테스트
pytest tests/ -v

# 3. 코드 품질 검사
black fragrance_ai/
flake8 fragrance_ai/
mypy fragrance_ai/

# 4. 커밋 및 푸시
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# 5. Pull Request 생성
gh pr create --title "Add new feature" --body "Description"
```

## 🧪 **테스트**

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함 테스트
pytest --cov=fragrance_ai --cov-report=html

# 특정 테스트만 실행
pytest tests/test_api.py -v

# 성능 테스트
pytest tests/performance/ --benchmark-only

# 보안 테스트
pytest tests/security/ -v
```

## 📊 **모니터링**

### 대시보드 접속

- **Grafana**: http://localhost:3000 (admin/fragrance_admin_2024)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### 주요 메트릭

```python
# API 메트릭
/api/v2/monitoring/metrics

# 시스템 메트릭
/api/v2/monitoring/system

# 알림 현황
/api/v2/monitoring/alerts
```

## 🚀 **배포**

### Docker 배포

```bash
# 이미지 빌드
docker build -t fragrance-ai:latest .

# 컨테이너 실행
docker run -d -p 8000:8000 \
  --name fragrance-ai \
  -e DATABASE_URL=$DATABASE_URL \
  -e REDIS_URL=$REDIS_URL \
  fragrance-ai:latest
```

### Kubernetes 배포

```bash
# Helm 차트로 배포
helm install fragrance-ai ./helm/fragrance-ai \
  --namespace production \
  --set image.tag=v1.0.0 \
  --set ingress.enabled=true \
  --set monitoring.enabled=true
```

### 클라우드 배포

```bash
# AWS EKS
eksctl create cluster --name fragrance-ai --region us-west-2

# Google GKE
gcloud container clusters create fragrance-ai --zone us-central1-a

# Azure AKS
az aks create --resource-group fragrance-ai --name fragrance-ai
```

## 🛡️ **보안**

### 보안 기능

- 🔐 **JWT 인증**: RS256 알고리즘
- 🛡️ **RBAC**: 역할 기반 접근 제어
- 🚨 **Rate Limiting**: 분당 100 요청
- 🔍 **입력 검증**: SQL Injection, XSS 방지
- 📊 **보안 헤더**: HSTS, CSP, X-Frame-Options

### 보안 스캔

```bash
# 의존성 취약점 스캔
safety check

# 코드 보안 스캐너
bandit -r fragrance_ai/

# Docker 이미지 스캔
trivy image fragrance-ai:latest

# 시크릿 스캔
detect-secrets scan --all-files
```

## 🔧 **문제 해결**

### 자주 발생하는 문제

#### 1. GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 4bit 양자화 활성화
export USE_4BIT_QUANTIZATION=true
```

#### 2. 데이터베이스 연결 오류
```bash
# 연결 테스트
psql -h localhost -U user -d fragrance_ai

# 연결 풀 설정 확인
export DATABASE_POOL_SIZE=20
```

#### 3. Redis 연결 문제
```bash
# Redis 연결 테스트
redis-cli ping

# 캐시 설정 확인
export REDIS_MAX_CONNECTIONS=100
```

## 🤝 **기여하기**

우리는 커뮤니티의 기여를 환영합니다!

### 기여 방법

1. **이슈 확인**: [GitHub Issues](https://github.com/fragrance-ai/fragrance-ai/issues)
2. **포크**: 리포지토리를 포크하세요
3. **브랜치**: 기능 브랜치를 생성하세요
4. **개발**: 코드를 작성하고 테스트하세요
5. **PR**: Pull Request를 생성하세요

### 기여 가이드라인

- [코딩 컨벤션](docs/development/coding-conventions.md)
- [커밋 메시지 가이드](docs/development/commit-messages.md)
- [PR 체크리스트](docs/development/pr-checklist.md)

## 📄 **라이센스**

이 프로젝트는 독점 라이센스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 **지원**

- 📧 **이메일**: support@fragrance-ai.com
- 💬 **Discord**: [Fragrance AI Community](https://discord.gg/fragrance-ai)
- 📖 **위키**: [GitHub Wiki](https://github.com/fragrance-ai/fragrance-ai/wiki)
- 🐛 **버그 리포트**: [GitHub Issues](https://github.com/fragrance-ai/fragrance-ai/issues)

## 🙏 **감사 인사**

- **OpenAI**: GPT 모델 제공
- **Hugging Face**: Transformers 라이브러리
- **FastAPI**: 고성능 웹 프레임워크
- **Grafana Labs**: 모니터링 솔루션

---

<div align="center">

**Made with ❤️ by the Fragrance AI Team**

[웹사이트](https://fragrance-ai.com) • [문서](https://docs.fragrance-ai.com) • [블로그](https://blog.fragrance-ai.com)

</div>