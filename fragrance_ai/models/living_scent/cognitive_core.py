"""
Living Scent - Cognitive Core AI
시스템의 '뇌' - 사용자의 복합적인 감정과 심상을 해독하는 인지 센터

LinguisticReceptorAI로부터 받은 정제된 정보를 바탕으로
사용자의 깊은 감정과 의도를 파악하고 창조적 브리프를 생성합니다.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@dataclass
class CreativeBrief:
    """CognitiveCoreAI의 출력 - 창조적 브리프"""
    theme: str  # 전체적인 테마 (예: "A Nostalgic Attic of Comfort")
    core_emotion: str  # 핵심 감정
    story: str  # 스토리텔링
    emotional_palette: Dict[str, float]  # 감정 팔레트
    sensory_map: Dict[str, float]  # 감각 지도
    memory_triggers: List[str]  # 기억 트리거
    cultural_context: str  # 문화적 맥락
    archetype: str  # 원형 (예: "The Comforter", "The Explorer")


class EmotionalInterpreterAgent:
    """감정 해석 하위 에이전트"""

    def __init__(self):
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

        # 감정 원형 매핑
        self.emotion_archetypes = {
            'nostalgia': {
                'primary': ['그리움', '추억', '향수'],
                'secondary': ['comfort', 'melancholy', 'warmth'],
                'palette': {'warm': 0.8, 'soft': 0.7, 'vintage': 0.9}
            },
            'romance': {
                'primary': ['사랑', '열정', '친밀'],
                'secondary': ['tenderness', 'desire', 'intimacy'],
                'palette': {'sweet': 0.9, 'floral': 0.8, 'sensual': 0.7}
            },
            'adventure': {
                'primary': ['모험', '자유', '도전'],
                'secondary': ['excitement', 'freedom', 'courage'],
                'palette': {'fresh': 0.9, 'energetic': 0.8, 'bold': 0.7}
            },
            'serenity': {
                'primary': ['평온', '안정', '명상'],
                'secondary': ['peace', 'calm', 'meditation'],
                'palette': {'clean': 0.8, 'soft': 0.9, 'balanced': 0.7}
            }
        }

    def interpret(self, keywords: List[str], emotional_keywords: List[str]) -> Dict[str, Any]:
        """감정 해석"""
        # 키워드 기반 감정 매칭
        matched_archetype = 'serenity'  # 기본값
        max_score = 0

        for archetype, data in self.emotion_archetypes.items():
            score = 0
            for keyword in keywords + emotional_keywords:
                if keyword in str(data['primary']) or keyword in str(data['secondary']):
                    score += 1
            if score > max_score:
                max_score = score
                matched_archetype = archetype

        return {
            'archetype': matched_archetype,
            'palette': self.emotion_archetypes[matched_archetype]['palette'],
            'intensity': min(max_score / 5, 1.0)  # 0-1 정규화
        }


class MemoryCartographerAgent:
    """기억 지도 제작 하위 에이전트"""

    def __init__(self):
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 기억의 범주
        self.memory_categories = {
            'childhood': ['어린시절', '유년', '아이', '놀이터', '학교'],
            'home': ['집', '가정', '가족', '안방', '거실'],
            'nature': ['자연', '숲', '바다', '산', '들판'],
            'urban': ['도시', '거리', '카페', '빌딩', '지하철'],
            'special_moments': ['기념일', '생일', '결혼', '졸업', '첫'],
        }

        # 감각 매핑
        self.sensory_associations = {
            'visual': ['보이는', '색', '빛', '어두운', '밝은'],
            'tactile': ['촉감', '부드러운', '거친', '따뜻한', '차가운'],
            'auditory': ['소리', '음악', '조용한', '시끄러운', '울림'],
            'gustatory': ['맛', '달콤한', '쓴', '신', '짠'],
            'olfactory': ['냄새', '향기', '향', '내음', '향수']
        }

    def map_memories(self, keywords: List[str], contextual_keywords: List[str]) -> Dict[str, Any]:
        """기억과 감각의 지도 제작"""
        # 기억 범주 매칭
        memory_scores = {}
        for category, patterns in self.memory_categories.items():
            score = sum(1 for kw in keywords + contextual_keywords if any(p in kw for p in patterns))
            if score > 0:
                memory_scores[category] = score

        # 감각 연상 매칭
        sensory_scores = {}
        for sense, patterns in self.sensory_associations.items():
            score = sum(1 for kw in keywords if any(p in kw for p in patterns))
            if score > 0:
                sensory_scores[sense] = score / 10  # 정규화

        # 기억 트리거 생성
        memory_triggers = []
        for category in memory_scores.keys():
            if category == 'childhood':
                memory_triggers.append("첫 번째 자전거를 탔던 날")
            elif category == 'home':
                memory_triggers.append("엄마가 끓여주던 된장찌개 냄새")
            elif category == 'nature':
                memory_triggers.append("여름날 숲속의 피톤치드")

        return {
            'memory_categories': memory_scores,
            'sensory_map': sensory_scores,
            'triggers': memory_triggers
        }


class NarrativeWeaverAgent:
    """내러티브 직조 하위 에이전트"""

    def __init__(self):
        self.story_templates = {
            'nostalgia': "사용자는 {time}의 {place}에서 느꼈던 {emotion}을 담은 향기를 찾고 있습니다. 이것은 단순한 향수가 아닌, 시간을 거슬러 올라가는 후각적 타임머신입니다.",
            'romance': "사용자는 {emotion}의 순간을 영원히 간직하고 싶어합니다. {place}에서의 {time}, 그 특별한 순간을 향기로 각인시키려 합니다.",
            'adventure': "새로운 {place}로의 여정을 앞둔 사용자는 {emotion}으로 가득 찬 향기를 원합니다. {time}의 도전 정신을 담은 향수입니다.",
            'serenity': "{time}의 {place}에서 찾은 평온함. 사용자는 {emotion}을 일상 속에서 언제나 느끼고 싶어합니다."
        }

        self.theme_generator = {
            'nostalgic_comfort': "A Nostalgic Attic of Comfort",
            'romantic_garden': "Secret Garden of Love",
            'urban_adventure': "Metropolitan Explorer",
            'serene_sanctuary': "Inner Sanctuary of Peace",
            'mysterious_night': "Enigmatic Midnight",
            'fresh_morning': "Dawn's First Light",
            'warm_embrace': "Eternal Embrace"
        }

    def weave_story(self, emotion_data: Dict, memory_data: Dict, keywords: List[str]) -> Dict[str, str]:
        """스토리 직조"""
        # 핵심 요소 추출
        archetype = emotion_data.get('archetype', 'serenity')
        memory_categories = list(memory_data.get('memory_categories', {}).keys())

        # 시간, 장소, 감정 추출
        time_element = "그 시절" if memory_categories else "지금 이 순간"
        place_element = memory_categories[0] if memory_categories else "마음 속"
        emotion_element = archetype

        # 스토리 템플릿 선택 및 채우기
        template = self.story_templates.get(archetype, self.story_templates['serenity'])
        story = template.format(
            time=time_element,
            place=place_element,
            emotion=emotion_element
        )

        # 테마 생성
        theme_key = f"{archetype}_{memory_categories[0] if memory_categories else 'general'}"
        theme = self.theme_generator.get(
            theme_key,
            f"The {archetype.title()} Journey"
        )

        return {
            'theme': theme,
            'story': story,
            'archetype_narrative': archetype
        }


class CognitiveCoreAI:
    """
    시스템의 인지 중추 - 3개의 하위 에이전트를 통해
    사용자의 복합적인 감정과 의도를 심층 분석
    """

    def __init__(self):
        # 하위 에이전트 초기화
        self.emotional_interpreter = EmotionalInterpreterAgent()
        self.memory_cartographer = MemoryCartographerAgent()
        self.narrative_weaver = NarrativeWeaverAgent()

        # 문화적 맥락 분석기
        self.cultural_patterns = {
            'korean': ['한국', '김치', '한복', '된장', '고추장'],
            'western': ['서양', '와인', '치즈', '커피', '브런치'],
            'asian': ['아시아', '차', '대나무', '연꽃', '선'],
            'mediterranean': ['지중해', '올리브', '라벤더', '허브', '바다']
        }

    def analyze_cultural_context(self, keywords: List[str]) -> str:
        """문화적 맥락 분석"""
        for culture, patterns in self.cultural_patterns.items():
            if any(p in ' '.join(keywords) for p in patterns):
                return culture
        return 'universal'

    def synthesize(self, structured_input: Any) -> CreativeBrief:
        """
        모든 하위 에이전트의 분석을 종합하여
        창조적 브리프 생성
        """
        # 1. 감정 해석
        emotion_analysis = self.emotional_interpreter.interpret(
            structured_input.keywords,
            structured_input.emotional_keywords
        )

        # 2. 기억 지도 제작
        memory_map = self.memory_cartographer.map_memories(
            structured_input.keywords,
            structured_input.contextual_keywords
        )

        # 3. 내러티브 직조
        narrative = self.narrative_weaver.weave_story(
            emotion_analysis,
            memory_map,
            structured_input.keywords
        )

        # 4. 문화적 맥락
        cultural_context = self.analyze_cultural_context(structured_input.keywords)

        # 5. 감정 팔레트 구성
        emotional_palette = emotion_analysis['palette'].copy()

        # 감각 지도가 비어있을 경우 기본값 설정
        sensory_map = memory_map.get('sensory_map', {})
        if not sensory_map:
            sensory_map = {
                'olfactory': 1.0,
                'tactile': 0.5,
                'visual': 0.3
            }

        # CreativeBrief 생성
        creative_brief = CreativeBrief(
            theme=narrative['theme'],
            core_emotion=narrative['archetype_narrative'],
            story=narrative['story'],
            emotional_palette=emotional_palette,
            sensory_map=sensory_map,
            memory_triggers=memory_map.get('triggers', []),
            cultural_context=cultural_context,
            archetype=emotion_analysis['archetype']
        )

        return creative_brief

    def to_json(self, brief: CreativeBrief) -> str:
        """CreativeBrief를 JSON으로 변환"""
        return json.dumps({
            'theme': brief.theme,
            'core_emotion': brief.core_emotion,
            'story': brief.story,
            'emotional_palette': brief.emotional_palette,
            'sensory_map': brief.sensory_map,
            'memory_triggers': brief.memory_triggers,
            'cultural_context': brief.cultural_context,
            'archetype': brief.archetype
        }, ensure_ascii=False, indent=2)


# 싱글톤 인스턴스
_cognitive_core_instance = None

def get_cognitive_core() -> CognitiveCoreAI:
    """싱글톤 CognitiveCoreAI 인스턴스 반환"""
    global _cognitive_core_instance
    if _cognitive_core_instance is None:
        _cognitive_core_instance = CognitiveCoreAI()
    return _cognitive_core_instance