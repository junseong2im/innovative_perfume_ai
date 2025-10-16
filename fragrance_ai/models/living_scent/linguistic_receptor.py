"""
Living Scent - Linguistic Receptor AI
시스템의 '감각기관' - 사용자의 텍스트를 이해하고 구조화된 입력으로 변환

이 AI는 사용자가 입력한 텍스트를 받아 전처리하고,
의도를 분류하며, 핵심 키워드를 추출하는 역할을 수행합니다.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline


class UserIntent(Enum):
    """사용자 의도 분류"""
    CREATE_NEW = "CREATE_NEW"  # 완전히 새로운 향수 생성
    EVOLVE_EXISTING = "EVOLVE_EXISTING"  # 기존 향수의 변형
    UNKNOWN = "UNKNOWN"  # 분류 불가


@dataclass
class StructuredInput:
    """LinguisticReceptor의 출력 데이터 구조"""
    intent: UserIntent
    keywords: List[str]
    cleaned_text: str
    emotional_keywords: List[str]
    contextual_keywords: List[str]
    temporal_keywords: List[str]
    confidence_score: float


class LinguisticReceptorAI:
    """
    사용자의 언어를 이해하고 구조화하는 AI
    생명체의 감각 수용체처럼 외부 자극(텍스트)을 내부 신호로 변환
    """

    def __init__(self):
        # 의미 이해를 위한 임베딩 모델
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 감정 분석 파이프라인
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

        # 키워드 패턴 정의
        self.emotion_patterns = {
            'nostalgic': ['옛날', '추억', '어린시절', '그리운', '할머니', '할아버지', '고향'],
            'romantic': ['사랑', '로맨틱', '연인', '데이트', '달콤한', '애인'],
            'fresh': ['상큼', '신선', '깨끗', '청량', '시원', '프레시'],
            'warm': ['따뜻', '포근', '온화', '부드러운', '안락'],
            'mysterious': ['신비', '미스터리', '오묘', '깊은', '알수없는'],
            'luxurious': ['고급', '럭셔리', '프리미엄', '우아', '품격'],
            'energetic': ['활발', '에너지', '활기', '생동감', '다이나믹']
        }

        self.temporal_patterns = {
            'spring': ['봄', '새싹', '벚꽃', '봄날'],
            'summer': ['여름', '바다', '해변', '태양'],
            'autumn': ['가을', '단풍', '낙엽', '수확'],
            'winter': ['겨울', '눈', '크리스마스', '연말']
        }

        self.creation_keywords = ['만들어', '생성', '창조', '새로운', '신규', '처음']
        self.evolution_keywords = ['변형', '수정', '조정', '변경', '바꿔', '더', '덜', '강하게', '약하게']

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 - 노이즈 제거 및 정규화"""
        # 소문자 변환
        text = text.lower()

        # 특수문자 제거 (한글, 영어, 숫자, 공백만 유지)
        text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)

        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def classify_intent(self, text: str) -> tuple[UserIntent, float]:
        """사용자 의도 분류"""
        cleaned = self.preprocess_text(text)

        # 키워드 기반 분류
        creation_score = sum(1 for kw in self.creation_keywords if kw in cleaned)
        evolution_score = sum(1 for kw in self.evolution_keywords if kw in cleaned)

        # 문맥 임베딩 기반 분류
        text_embedding = self.embedder.encode(cleaned)

        # 대표 문장들과의 유사도 계산
        creation_templates = [
            "새로운 향수를 만들어주세요",
            "향수를 생성해주세요",
            "나만의 향수를 창조해주세요"
        ]
        evolution_templates = [
            "이 향수를 더 강하게 만들어주세요",
            "조금 변형해주세요",
            "살짝 바꿔주세요"
        ]

        creation_embeddings = self.embedder.encode(creation_templates)
        evolution_embeddings = self.embedder.encode(evolution_templates)

        # 코사인 유사도 계산
        creation_similarity = np.mean([
            np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb))
            for emb in creation_embeddings
        ])
        evolution_similarity = np.mean([
            np.dot(text_embedding, emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(emb))
            for emb in evolution_embeddings
        ])

        # 종합 스코어
        total_creation = creation_score * 0.5 + creation_similarity * 10
        total_evolution = evolution_score * 0.5 + evolution_similarity * 10

        if total_creation > total_evolution and total_creation > 0.3:
            confidence = min(total_creation / (total_creation + total_evolution + 0.1), 0.99)
            return UserIntent.CREATE_NEW, confidence
        elif total_evolution > total_creation and total_evolution > 0.3:
            confidence = min(total_evolution / (total_creation + total_evolution + 0.1), 0.99)
            return UserIntent.EVOLVE_EXISTING, confidence
        else:
            # 명확하지 않은 경우 기본값은 CREATE_NEW
            return UserIntent.CREATE_NEW, 0.5

    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """다차원 키워드 추출"""
        cleaned = self.preprocess_text(text)
        words = cleaned.split()

        # 감정 키워드 추출
        emotional_keywords = []
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if pattern in cleaned:
                    emotional_keywords.append(emotion)
                    break

        # 시간/계절 키워드 추출
        temporal_keywords = []
        for season, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                if pattern in cleaned:
                    temporal_keywords.append(season)
                    break

        # 문맥 키워드 추출 (명사 위주)
        contextual_keywords = []
        # 간단한 명사 추출 (실제로는 형태소 분석기 사용 권장)
        noun_patterns = ['댁', '방', '나무', '향기', '느낌', '분위기', '기억', '순간']
        for word in words:
            for pattern in noun_patterns:
                if pattern in word:
                    contextual_keywords.append(word)
                    break

        # 전체 핵심 키워드 (중복 제거)
        all_keywords = list(set(
            emotional_keywords + temporal_keywords + contextual_keywords +
            [w for w in words if len(w) > 1][:10]  # 상위 10개 단어
        ))

        return {
            'all': all_keywords[:15],
            'emotional': emotional_keywords,
            'temporal': temporal_keywords,
            'contextual': contextual_keywords
        }

    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """감정 분석"""
        try:
            results = self.emotion_analyzer(text)
            emotion_scores = {}
            for result_list in results:
                for item in result_list:
                    emotion_scores[item['label']] = item['score']
            return emotion_scores
        except:
            # 폴백: 기본 감정 점수
            return {
                'neutral': 0.5,
                'joy': 0.2,
                'surprise': 0.1,
                'sadness': 0.1,
                'anger': 0.05,
                'fear': 0.05
            }

    def process(self, user_input: str) -> StructuredInput:
        """
        메인 처리 함수
        사용자 입력을 받아 구조화된 데이터로 변환
        """
        # 전처리
        cleaned_text = self.preprocess_text(user_input)

        # 의도 분류
        intent, confidence = self.classify_intent(user_input)

        # 키워드 추출
        keywords_dict = self.extract_keywords(user_input)

        # 감정 분석
        emotion_scores = self.analyze_emotion(user_input)

        # 구조화된 입력 생성
        structured_input = StructuredInput(
            intent=intent,
            keywords=keywords_dict['all'],
            cleaned_text=cleaned_text,
            emotional_keywords=keywords_dict['emotional'],
            contextual_keywords=keywords_dict['contextual'],
            temporal_keywords=keywords_dict['temporal'],
            confidence_score=confidence
        )

        return structured_input

    def to_json(self, structured_input: StructuredInput) -> str:
        """StructuredInput을 JSON으로 변환"""
        return json.dumps({
            'intent': structured_input.intent.value,
            'keywords': structured_input.keywords,
            'cleaned_text': structured_input.cleaned_text,
            'emotional_keywords': structured_input.emotional_keywords,
            'contextual_keywords': structured_input.contextual_keywords,
            'temporal_keywords': structured_input.temporal_keywords,
            'confidence_score': structured_input.confidence_score
        }, ensure_ascii=False, indent=2)


# 싱글톤 인스턴스
_linguistic_receptor_instance = None

def get_linguistic_receptor() -> LinguisticReceptorAI:
    """싱글톤 LinguisticReceptorAI 인스턴스 반환"""
    global _linguistic_receptor_instance
    if _linguistic_receptor_instance is None:
        _linguistic_receptor_instance = LinguisticReceptorAI()
    return _linguistic_receptor_instance