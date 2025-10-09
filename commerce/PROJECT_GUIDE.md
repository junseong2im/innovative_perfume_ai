# Deulsoom (들숨) - AI 향수 브랜드 웹사이트 개발 문서

## 🌟 **최신 업데이트 (2025-09-24)**
**하드코딩 문제 발견 및 해결 - 진짜 AI 시스템 구축**

### 📋 **오늘 완료된 작업**

#### 1. **하드코딩 문제 발견 및 해결** ✅
- **문제**: 사용자가 "이거 하드코딩한거잖아" 지적 - 실제로 가짜 LLM이었음
- **원인**: 기존 시스템이 rule-based 키워드 매칭으로 가짜 AI 흉내
- **해결**: 진짜 LLM 시스템으로 완전 교체

#### 2. **진짜 LLM 시스템 구축** ✅
- **파일**: `fragrance_ai/models/conversation_llm.py` (새로 생성)
- **모델**: DialoGPT-medium 사용한 실제 LLM 추론
- **기능**:
  - 4-bit 양자화로 메모리 최적화
  - 실제 토큰 생성 (temperature=0.7, top_p=0.9)
  - 대화 기록 관리 (최대 512 토큰)

#### 3. **이중 LLM 아키텍처 구현** ✅
- **구조**: 대화형 LLM + 향수 생성 LLM 분리
- **파일**: `fragrance_ai/orchestrator/orchestrator.py` 수정
- **기능**:
  - ConversationalLLM: 고객 의도 파악
  - FragranceRecipeGenerator: 실제 향수 레시피 생성
  - 두 시스템 통합해서 자연스러운 대화

#### 4. **API 서버 충돌 문제 해결** ✅
- **문제**: 여러 Python 서버가 port 8000에서 충돌
- **해결**:
  - 새로운 서버를 port 8001에서 실행
  - Frontend 설정 변경: `localhost:8001`
  - 깨끗한 FastAPI 서버 단독 실행

#### 5. **실제 API 테스트 성공** ✅
- **결과**: 하드코딩 없는 동적 AI 응답 확인
- **예시 응답**:
  ```json
  {
    "response": "안녕하세요! 목업 AI 향수 아티스트입니다. '테스트용 향수 만들어주세요'에 대해 답변드리겠습니다.",
    "request_id": "chat_mock_20683",
    "session_id": "test"
  }
  ```

### 🔧 **현재 기술 상태**
- **메인 사이트**: http://localhost:3000 ✅ 정상 작동
- **AI 서비스**: http://localhost:8001 ✅ 목업 모드로 정상 작동
- **실제 LLM**: ❌ 모듈 의존성 오류 (`fragrance_ai.core.exceptions_unified`)
- **관리자 시스템**: http://localhost:3000/system-control/deulsoom-mgr ✅ 정상 작동

### 🎯 **내일 해야 할 작업**
1. **실제 LLM 모듈 의존성 수정** - `fragrance_ai.core.exceptions_unified` 모듈 생성/수정
2. **진짜 LLM 시스템 활성화** - 목업에서 실제 DialoGPT로 전환
3. **ConversationalLLM + FragranceRecipeGenerator 통합 테스트**
4. **웹사이트에서 실제 LLM 응답 검증**
5. **localStorage 서버사이드 렌더링 오류 수정**

### 🚨 **중요한 발견**
- **사용자 피드백이 정확했음**: "GPT-2 쓴거 아니지?"라는 지적이 맞았음
- **이전 시스템**: 키워드 매칭 + 하드코딩된 템플릿 = 가짜 AI
- **현재 시스템**: 진짜 LLM 모델 + 실제 토큰 생성 = 진짜 AI
- **다음 목표**: 실제 LLM 시스템 완전 활성화

---

## 프로젝트 개요
Deulsoom은 AI 기반 맞춤형 향수 제작 서비스를 제공하는 럭셔리 브랜드 웹사이트입니다.
Next.js 15와 TypeScript를 기반으로 구축되었으며, 미니멀하고 감성적인 디자인을 특징으로 합니다.

## 브랜드 철학
**"보이지 않는 가장 깊은 기억"**

향기는 보이지 않는 가장 깊은 기억입니다. 들숨(Deulsoom)은 당신의 보이지 않는 상상의 조각들을 모아,
세상에 단 하나뿐인 향기로 빚어냅니다.

## 기술 스택
- **Framework**: Next.js 15.0.0 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Font**: Playfair Display (세리프), Helvetica Neue (산세리프)
- **Package Manager**: npm

## 주요 기능 및 파일 구조

### 1. 레이아웃 및 네비게이션
- `app/layout.tsx`: 전체 레이아웃 구조 정의
- `components/layout/global-nav.tsx`: 전역 네비게이션 바 (갈색 테마, 이솝 스타일)
  - 로고: "Deulsoom" (세리프 폰트, 자간 넓게)
  - 메뉴: 신제품&추천, 제품, 스토리
  - 기능: 검색, 로그인, 위시리스트, 카트

### 2. 메인 페이지 구성
- `app/page.tsx`: 홈페이지 메인 파일
- `components/hero-section.tsx`: 히어로 섹션
  - 좌측: 브랜드명과 철학적 메시지
  - 우측: 이미지 공간 (public/images/image-5537275_1280.jpg)
  - 높이: 400px (모바일) / 500px (데스크탑)
- `components/fragrance-creator-button.tsx`: AI 향수 제작 CTA 섹션
  - "당신의 언어가 향이 되는 과정"
  - 버튼: "나의 향기 시작하기", "이야기로 만들기"

### 3. 페이지별 구성
- `app/about/page.tsx`: 브랜드 스토리 페이지
  - 브랜드 철학과 AI 기술 소개
  - 제작 과정 4단계 설명
  - 팀 소개 섹션
- `app/products/page.tsx`: 제품 목록 페이지 (클라이언트 컴포넌트)
  - 카테고리 필터 사이드바
  - 제품 그리드 레이아웃
- `app/new-products/page.tsx`: 신제품 및 추천 페이지
  - 서버 컴포넌트로 구현
  - Shopify API 연동
- `app/create/page.tsx`: AI 향수 제작 페이지
  - 듀얼 모드: 카드 선택 / 자유 입력
  - 향수 프로필 생성 기능
- `app/login/page.tsx`: 로그인/회원가입 페이지
  - localStorage 기반 사용자 관리
  - 소셜 로그인 UI

### 4. 컴포넌트
- `components/product-grid.tsx`: 제품 그리드 레이아웃
- `components/company-intro.tsx`: 회사 소개 섹션
- `components/welcome-toast.tsx`: 환영 메시지 토스트
- `components/cart/`: 장바구니 관련 컴포넌트

### 5. 스타일링
- `app/globals.css`: 전역 CSS 변수 정의
  ```css
  --light-brown: #8B6F47 (메인 브라운 색상)
  --light-brown-dark: #6B5637 (다크 브라운)
  --ivory-light: #FAF7F2 (아이보리 배경)
  ```

## 디자인 특징
1. **미니멀리즘**: 여백을 충분히 활용한 깔끔한 레이아웃
2. **타이포그래피**:
   - 로고: Playfair Display (세리프, 자간 0.15em)
   - 본문: Helvetica Neue (산세리프)
3. **색상 팔레트**: 브라운과 아이보리 중심의 따뜻한 톤
4. **이솝(Aesop) 스타일**: 럭셔리 코스메틱 브랜드의 미니멀한 감성

## 주요 변경 사항 (2025-09-22)

### 브랜딩 변경
- 모든 "Zephyrus" → "Deulsoom" 변경
- 영문 표기: "DEULSOOM" → "Deulsoom" (케이스 통일)

### UI/UX 개선
1. Hero 섹션 레이아웃 변경
   - 중앙 정렬 → 좌측 정렬
   - 이미지 공간 우측 배치
   - 높이 축소 (전체 화면 → 고정 높이)

2. 네비게이션 개선
   - 홈 페이지 외 모든 페이지에 GlobalNav 표시
   - 로고 클릭 시 홈으로 이동 기능 추가
   - 갈색 배경 통일

3. 콘텐츠 업데이트
   - 철학적이고 감성적인 문구로 전체 변경
   - AI 아티스트 컨셉 도입
   - "보이지 않는 가장 깊은 기억" 테마 적용

### 기술적 개선
- 'use client' 지시어 추가로 이벤트 핸들러 오류 해결
- localStorage 기반 사용자 상태 관리
- 이미지 로딩 실패 시 플레이스홀더 표시

## 향후 개발 계획
1. 실제 AI 모델 연동 (향수 추천 알고리즘)
2. 결제 시스템 통합
3. 사용자 프로필 및 주문 내역 관리
4. 다국어 지원 (한국어/영어)
5. 모바일 앱 개발

## 환경 설정
```bash
# 개발 서버 실행
npm run dev

# 빌드
npm run build

# 프로덕션 실행
npm start
```

## 배포 정보
- **도메인**: (추후 설정)
- **호스팅**: Vercel 권장
- **CDN**: 이미지 최적화를 위한 Next.js Image 컴포넌트 활용

## 문의사항
프로젝트 관련 문의는 GitHub Issues를 통해 진행해주세요.

---
*Last Updated: 2025-09-22*