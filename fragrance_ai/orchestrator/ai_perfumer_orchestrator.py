"""
AI Perfumer Orchestrator - 공감각 조향사 시스템
세상의 모든 개념을 향으로 번역하는 예술가
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import json

logger = logging.getLogger(__name__)

class AIPerfumerOrchestrator:
    """
    공감각 조향사 오케스트레이터

    역할:
    - 추상적 개념을 향으로 번역
    - 감정과 기억을 향수로 변환
    - 시공간적 경험을 향으로 재현
    """

    def __init__(self):
        self.sensory_dictionary = self._initialize_sensory_dictionary()
        self.fragrance_database = self._initialize_fragrance_database()
        self.conversation_memory = {}

    def _initialize_sensory_dictionary(self) -> Dict:
        """감각 사전 초기화"""
        return {
            '정체성': {
                'colors': ['거울의 은빛', '변화하는 무지개빛'],
                'temperatures': ['체온', '변동하는'],
                'textures': ['유동적', '다층적'],
                'emotions': ['자아인식', '소속감'],
                'concepts': ['나', '우리', '경계']
            },
            '시간': {
                'colors': ['모래시계의 금빛', '석양의 주황'],
                'temperatures': ['서늘함에서 따뜻함으로'],
                'textures': ['흐르는', '쌓이는'],
                'emotions': ['흐름', '정지', '영원'],
                'concepts': ['과거', '현재', '미래']
            },
            '고독': {
                'colors': ['깊은 남빛', '차가운 회색'],
                'temperatures': ['서늘함'],
                'textures': ['비어있는', '울림이 있는'],
                'emotions': ['외로움', '평화', '자기성찰'],
                'concepts': ['내면', '거리', '독립']
            }
        }

    def _initialize_fragrance_database(self) -> Dict:
        """향료 데이터베이스 초기화"""
        return {
            'existential': ['Void Accord', 'Quantum Musk', 'Paradox'],
            'temporal': ['Vintage Leather', 'Future Metal', 'Nostalgia'],
            'emotional': ['Melancholia', 'Euphoria', 'Serenity'],
            'cultural': ['Temple Incense', 'Library Dust', 'Digital Static']
        }

    def generate_response(self, message: str, context: List[str]) -> str:
        """
        대화형 응답 생성

        Args:
            message: 사용자 메시지
            context: 대화 맥락

        Returns:
            시적이고 깊이 있는 AI 응답
        """
        # 메시지 분석
        attributes = self._analyze_message(message)
        message_count = len(context)

        # 특별한 개념에 대한 심층 응답
        if '정체성' in message or 'identity' in message.lower():
            return """정체성... 그것은 향수의 본질과 같습니다.
변하지만 변하지 않는, 나이면서 동시에 나를 넘어서는.
당신의 정체성은 어떤 층위를 가지고 있나요?
표면의 당신, 내면의 당신, 그리고 아직 발견하지 못한 당신."""

        if '시간' in message or 'time' in message.lower():
            return """시간을 향으로 포착한다는 것...
과거의 먼지 냄새, 현재의 생생함, 미래의 금속성 향.
당신이 담고 싶은 시간은 어느 순간인가요?"""

        if '고독' in message or 'solitude' in message.lower():
            return """고독 속에서 당신은 무엇을 발견하셨나요?
그것은 차가운 진실이었나요, 따뜻한 위로였나요?
고독이 당신에게 속삭인 것은 무엇인가요?"""

        # 대화 단계별 응답
        progressive_responses = [
            f'흥미롭습니다. "{message}"라는 표현에서 저는 여러 층위의 의미를 봅니다.',
            '점점 더 선명해지고 있어요. 당신의 이야기가 이미 향기를 품기 시작했습니다.',
            '거의 다 왔습니다. 마지막으로, 이 향수를 처음 맡는 사람이 느꼈으면 하는 첫 인상은?',
            '완벽합니다. 이제 당신의 영혼의 지문을 향으로 새길 준비가 되었습니다.'
        ]

        if message_count < len(progressive_responses):
            return progressive_responses[message_count]

        return "당신의 이야기는 이미 그 자체로 하나의 향수입니다. 이제 그것을 병에 담을 시간입니다..."

    def execute_creative_process(self, input_text: str) -> Dict[str, Any]:
        """
        창작 프로세스 실행 - 5단계

        Args:
            input_text: 전체 대화 내용

        Returns:
            생성된 향수 데이터
        """
        # 1. 속성 추출
        attributes = self._extract_attributes(input_text)

        # 2. 공감각적 번역
        synesthetic_map = self._translate_to_synesthesia(attributes)

        # 3. 감정적 매핑
        emotional_landscape = self._map_emotional_terrain(attributes)

        # 4. 시간적 층위 구성
        temporal_structure = self._construct_temporal_layers(emotional_landscape)

        # 5. 향수 합성
        fragrance = self._synthesize_fragrance(
            input_text, attributes, temporal_structure, emotional_landscape
        )

        return fragrance

    def _analyze_message(self, message: str) -> Dict:
        """메시지 분석"""
        attributes = {
            'emotions': [],
            'concepts': [],
            'temporalMarkers': [],
            'spatialMarkers': []
        }

        # 감정 키워드 추출
        emotion_keywords = ['사랑', '슬픔', '기쁨', '고독', '평화', '불안']
        for emotion in emotion_keywords:
            if emotion in message:
                attributes['emotions'].append(emotion)

        # 개념 추출
        concept_keywords = ['정체성', '시간', '공간', '기억', '꿈', '자유']
        for concept in concept_keywords:
            if concept in message:
                attributes['concepts'].append(concept)

        # 시간 마커
        time_markers = ['새벽', '아침', '정오', '황혼', '저녁', '밤', '봄', '여름', '가을', '겨울']
        for marker in time_markers:
            if marker in message:
                attributes['temporalMarkers'].append(marker)

        # 공간 마커
        space_markers = ['하늘', '땅', '바다', '산', '도시', '방', '집']
        for marker in space_markers:
            if marker in message:
                attributes['spatialMarkers'].append(marker)

        return attributes

    def _extract_attributes(self, text: str) -> Dict:
        """텍스트에서 속성 추출"""
        return self._analyze_message(text)

    def _translate_to_synesthesia(self, attributes: Dict) -> Dict:
        """공감각적 번역"""
        synesthetic_map = {}

        # 감정을 향으로
        if attributes.get('emotions'):
            emotion_scents = []
            emotion_to_scent = {
                '사랑': 'Rose de Mai',
                '슬픔': 'Iris Concrete',
                '기쁨': 'Champagne Accord',
                '고독': 'Vetiver Root',
                '평화': 'Lavender Absolute',
                '불안': 'Petrichor'
            }
            for emotion in attributes['emotions']:
                if emotion in emotion_to_scent:
                    emotion_scents.append(emotion_to_scent[emotion])
            synesthetic_map['emotion_scents'] = emotion_scents

        # 시간을 향으로
        if attributes.get('temporalMarkers'):
            time_scents = []
            time_to_scent = {
                '새벽': 'Dew Accord',
                '아침': 'Fresh Citrus',
                '황혼': 'Amber Sunset',
                '밤': 'Dark Woods',
                '봄': 'Spring Blossom',
                '여름': 'Solar Accord',
                '가을': 'Autumn Leaves',
                '겨울': 'Cold Spice'
            }
            for time in attributes['temporalMarkers']:
                if time in time_to_scent:
                    time_scents.append(time_to_scent[time])
            synesthetic_map['time_scents'] = time_scents

        return synesthetic_map

    def _map_emotional_terrain(self, attributes: Dict) -> Dict:
        """감정적 지형 매핑"""
        return {
            'primary_emotion': attributes.get('emotions', ['neutral'])[0],
            'emotional_progression': attributes.get('emotions', []),
            'intensity': len(attributes.get('emotions', [])) * 0.3,
            'resonance': 0.7
        }

    def _construct_temporal_layers(self, emotional_landscape: Dict) -> Dict:
        """시간적 층위 구성"""
        return {
            'opening': {
                'duration': '0-15 minutes',
                'character': 'gentle awakening',
                'notes': ['light', 'ethereal']
            },
            'development': {
                'duration': '15-120 minutes',
                'character': 'gradual unfolding',
                'notes': ['complex', 'layered']
            },
            'climax': {
                'duration': '2-4 hours',
                'character': emotional_landscape['primary_emotion'],
                'notes': ['powerful', 'memorable']
            },
            'resolution': {
                'duration': '4-8 hours',
                'character': 'gentle descent',
                'notes': ['warm', 'comforting']
            }
        }

    def _synthesize_fragrance(
        self, input_text: str, attributes: Dict,
        temporal_structure: Dict, emotional_landscape: Dict
    ) -> Dict:
        """최종 향수 합성"""

        # 이름 생성
        name_elements = []
        if attributes.get('concepts'):
            name_elements.append(attributes['concepts'][0])
        if attributes.get('temporalMarkers'):
            name_elements.append(attributes['temporalMarkers'][0])

        name = ' '.join(name_elements) if name_elements else 'Unnamed Essence'
        korean_name = f"{name_elements[0]}의 향" if name_elements else '이름 없는 향'

        # 구성 생성
        composition = {
            'top_notes': [
                {'name': 'Bergamot', 'description': '첫 인상의 설렘'},
                {'name': 'Lemon Zest', 'description': '활기찬 시작'}
            ],
            'heart_notes': [
                {'name': 'Damascus Rose', 'description': f"{emotional_landscape['primary_emotion']}의 중심'},
                {'name': 'Jasmine Absolute', 'description': '감정의 깊이'}
            ],
            'base_notes': [
                {'name': 'Ambergris', 'description': '영원한 기억'},
                {'name': 'Sandalwood', 'description': '따뜻한 마무리'}
            ]
        }

        # 특성 결정
        characteristics = {
            'intensity': self._calculate_intensity(attributes),
            'longevity': '8-12 hours',
            'sillage': 'Moderate to Heavy',
            'season': self._determine_season(attributes),
            'gender': 'Unisex',
            'keywords': attributes.get('emotions', []) + attributes.get('concepts', [])
        }

        # 스토리 생성
        story = f"""이것은 "{input_text[:50]}..."에서 시작된 향기의 이야기입니다.

{composition['top_notes'][0]['description']}로 시작하여,
{composition['heart_notes'][0]['description']}를 거쳐,
{composition['base_notes'][0]['description']}로 마무리됩니다.

당신만의 특별한 순간을 영원히 간직할 향수입니다."""

        return {
            'name': name,
            'korean_name': korean_name,
            'story': story,
            'composition': composition,
            'characteristics': characteristics,
            'creation_context': {
                'user_input': input_text,
                'extracted_concepts': attributes.get('concepts', []),
                'dominant_theme': attributes.get('concepts', ['essence'])[0] if attributes.get('concepts') else 'essence',
                'creation_timestamp': datetime.now().isoformat()
            }
        }

    def _calculate_intensity(self, attributes: Dict) -> str:
        """강도 계산"""
        total_elements = (
            len(attributes.get('emotions', [])) +
            len(attributes.get('concepts', []))
        )

        if total_elements <= 2:
            return 'Light'
        elif total_elements <= 4:
            return 'Moderate'
        elif total_elements <= 6:
            return 'Strong'
        else:
            return 'Intense'

    def _determine_season(self, attributes: Dict) -> str:
        """계절 결정"""
        temporal_markers = attributes.get('temporalMarkers', [])

        season_map = {
            '봄': 'Spring',
            '여름': 'Summer',
            '가을': 'Autumn',
            '겨울': 'Winter'
        }

        for marker in temporal_markers:
            if marker in season_map:
                return season_map[marker]

        return 'All Seasons'


# 싱글톤 인스턴스
_ai_perfumer_orchestrator = None

def get_ai_perfumer_orchestrator() -> AIPerfumerOrchestrator:
    """AI Perfumer 오케스트레이터 싱글톤 인스턴스 반환"""
    global _ai_perfumer_orchestrator
    if _ai_perfumer_orchestrator is None:
        _ai_perfumer_orchestrator = AIPerfumerOrchestrator()
        logger.info("AI Perfumer Orchestrator initialized")
    return _ai_perfumer_orchestrator