"""
Perfume Description LLM - 고객 향수 설명 전문 AI
- 대형 언어 모델 사용 (120B급 또는 유사)
- 고객의 추상적인 설명을 구체적인 향수 레시피로 변환
- 감정, 기억, 분위기를 향료로 해석
"""

import asyncio
import logging
import json
from typing import Optional, List, Dict, Any
import aiohttp
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerfumeInterpretation:
    """향수 해석 결과"""
    fragrance_families: List[str]
    key_notes: List[str]
    mood: str
    season: str
    intensity: str
    occasion: str
    style: str
    color_impression: str
    emotional_profile: Dict[str, float]
    memory_triggers: List[str]
    confidence: float


class PerfumeDescriptionLLM:
    """
    고객 향수 설명 전문 LLM
    - 추상적 설명을 구체적 향료로 변환
    - 감정과 기억을 노트로 해석
    - 시적인 표현을 기술적 레시피로 변환
    """
    
    def __init__(self, config: Optional[dict] = None):
        """초기화"""
        self.config = config or self._load_config()
        self.provider = self.config.get('provider', 'ollama')
        self.model_name = self.config.get('model_name_or_path', 'qwen:32b')  # Qwen 32B as alternative
        self.api_base = self.config.get('api_base', 'http://localhost:11434')
        self.session = None
        
        # 향수 전문 시스템 프롬프트
        self.system_prompt = """You are a master perfumer and fragrance psychologist with decades of experience.
Your expertise includes:
- Translating emotions, memories, and abstract concepts into specific fragrance notes
- Understanding cultural and personal associations with scents
- Creating harmonious fragrance compositions from poetic descriptions
- Identifying the technical perfume components that evoke specific feelings

When a customer describes their ideal fragrance using abstract terms, emotions, or memories,
you expertly interpret these into:
1. Specific fragrance notes (top, heart, base)
2. Fragrance families and accords
3. Technical perfume characteristics
4. Emotional and sensory profiles

Always respond with deep understanding of both the emotional and technical aspects of perfumery."""
        
        # 노트 매핑 데이터베이스
        self.emotion_to_notes = {
            "happiness": ["citrus", "ylang-ylang", "vanilla", "peach"],
            "nostalgia": ["powder", "violet", "heliotrope", "almond"],
            "romance": ["rose", "jasmine", "sandalwood", "musk"],
            "energy": ["mint", "ginger", "black pepper", "grapefruit"],
            "calm": ["lavender", "chamomile", "white tea", "bamboo"],
            "mystery": ["incense", "oud", "patchouli", "black orchid"],
            "freshness": ["bergamot", "green tea", "cucumber", "ocean notes"],
            "warmth": ["amber", "benzoin", "tonka bean", "cinnamon"],
            "elegance": ["iris", "white musk", "cashmere wood", "magnolia"],
            "power": ["leather", "tobacco", "vetiver", "cedarwood"]
        }
        
    def _load_config(self) -> dict:
        """설정 로드"""
        try:
            with open('configs/local.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('llm_perfume_description', {
                'provider': 'ollama',
                'model_name_or_path': 'qwen:32b',  # 32B model for better understanding
                'api_base': 'http://localhost:11434'
            })
        except:
            return {
                'provider': 'ollama',
                'model_name_or_path': 'qwen:32b',
                'api_base': 'http://localhost:11434'
            }
    
    async def _ensure_session(self):
        """세션 확인"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def interpret_description(
        self,
        description: str,
        customer_profile: Optional[Dict[str, Any]] = None
    ) -> PerfumeInterpretation:
        """
        고객 설명을 향수 레시피로 해석
        
        Args:
            description: 고객의 향수 설명 (추상적/시적 표현 포함)
            customer_profile: 고객 프로필 (선호도, 이전 구매 등)
            
        Returns:
            PerfumeInterpretation: 해석된 향수 특성
        """
        try:
            await self._ensure_session()
            
            # 프롬프트 구성
            prompt = self._build_interpretation_prompt(description, customer_profile)
            
            # LLM 호출
            if self.provider == 'ollama':
                interpretation_text = await self._ollama_interpret(prompt)
            else:
                # OpenAI API 등 다른 provider 지원 가능
                interpretation_text = self._rule_based_interpret(description)
                
            # 결과 파싱
            return self._parse_interpretation(interpretation_text, description)
            
        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            return self._fallback_interpretation(description)
    
    def _build_interpretation_prompt(self, description: str, profile: Optional[Dict]) -> str:
        """해석 프롬프트 구성"""
        prompt = f"""Analyze this fragrance description and provide a detailed perfume interpretation:

Customer Description: "{description}"
"""
        
        if profile:
            prompt += f"""
Customer Profile:
- Age Group: {profile.get('age_group', 'unknown')}
- Gender: {profile.get('gender', 'unisex')}
- Previous Favorites: {', '.join(profile.get('favorite_perfumes', []))}
- Style Preference: {profile.get('style', 'modern')}
"""
        
        prompt += """
Provide a comprehensive interpretation including:

1. FRAGRANCE FAMILIES (primary and secondary)
2. KEY NOTES:
   - Top Notes (3-5 specific ingredients)
   - Heart Notes (3-5 specific ingredients)  
   - Base Notes (3-5 specific ingredients)
3. MOOD & CHARACTER (romantic, mysterious, fresh, etc.)
4. SEASON (best suited for)
5. INTENSITY (light, moderate, intense)
6. OCCASION (daily, evening, special events)
7. STYLE (modern, classic, avant-garde)
8. COLOR IMPRESSION (what colors this scent evokes)
9. EMOTIONAL PROFILE (percentages of different emotions)
10. MEMORY TRIGGERS (what memories or places it might evoke)

Be specific with actual perfume ingredients, not generic terms."""
        
        return prompt
    
    async def _ollama_interpret(self, prompt: str) -> str:
        """Ollama로 해석"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            async with self.session.post(
                f"{self.api_base}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('message', {}).get('content', '')
                else:
                    logger.error(f"Ollama interpretation failed with status {resp.status}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Ollama interpretation error: {e}")
            return ""
    
    def _rule_based_interpret(self, description: str) -> str:
        """규칙 기반 해석 (폴백)"""
        description_lower = description.lower()
        
        # 감정 감지
        detected_emotions = []
        for emotion, notes in self.emotion_to_notes.items():
            if emotion in description_lower or any(synonym in description_lower for synonym in self._get_emotion_synonyms(emotion)):
                detected_emotions.append(emotion)
        
        # 계절 감지
        season = "all seasons"
        if any(word in description_lower for word in ["summer", "여름", "hot", "beach"]):
            season = "summer"
        elif any(word in description_lower for word in ["winter", "겨울", "cold", "cozy"]):
            season = "winter"
        elif any(word in description_lower for word in ["spring", "봄", "bloom", "fresh"]):
            season = "spring"
        elif any(word in description_lower for word in ["autumn", "fall", "가을", "warm"]):
            season = "autumn"
            
        # 강도 감지
        intensity = "moderate"
        if any(word in description_lower for word in ["light", "subtle", "가벼운", "delicate"]):
            intensity = "light"
        elif any(word in description_lower for word in ["strong", "powerful", "intense", "강한"]):
            intensity = "intense"
            
        # 해석 텍스트 생성
        interpretation = f"""
FRAGRANCE FAMILIES: {', '.join(['Floral', 'Fresh'] if detected_emotions else ['Woody', 'Oriental'])}

KEY NOTES:
- Top Notes: {', '.join(['Bergamot', 'Lemon', 'Green Apple'])}
- Heart Notes: {', '.join(['Rose', 'Jasmine', 'Peony'] if 'romance' in detected_emotions else ['Lavender', 'Geranium'])}
- Base Notes: {', '.join(['Sandalwood', 'Musk', 'Amber'])}

MOOD: {detected_emotions[0] if detected_emotions else 'balanced'}
SEASON: {season}
INTENSITY: {intensity}
OCCASION: {'daily wear' if intensity == 'light' else 'special events'}
STYLE: modern
COLOR IMPRESSION: {self._get_color_impression(detected_emotions)}
"""
        return interpretation
    
    def _get_emotion_synonyms(self, emotion: str) -> List[str]:
        """감정 동의어 가져오기"""
        synonyms = {
            "happiness": ["happy", "joy", "cheerful", "기쁜", "행복"],
            "romance": ["romantic", "love", "passion", "로맨틱", "사랑"],
            "calm": ["peaceful", "serene", "tranquil", "평화", "고요"],
            "mystery": ["mysterious", "enigmatic", "dark", "신비", "미스터리"],
            "freshness": ["fresh", "clean", "crisp", "신선", "상쾌"]
        }
        return synonyms.get(emotion, [])
    
    def _get_color_impression(self, emotions: List[str]) -> str:
        """감정에서 색상 인상 도출"""
        if "romance" in emotions:
            return "Pink, Red, Soft Purple"
        elif "freshness" in emotions:
            return "Green, Blue, White"
        elif "mystery" in emotions:
            return "Deep Purple, Black, Gold"
        elif "happiness" in emotions:
            return "Yellow, Orange, Bright Pink"
        else:
            return "Neutral tones, Beige, Soft Grey"
    
    def _parse_interpretation(self, text: str, original_description: str) -> PerfumeInterpretation:
        """해석 텍스트 파싱"""
        try:
            # 간단한 텍스트 파싱 (실제로는 더 정교한 파싱 필요)
            lines = text.strip().split('\n')
            
            # 기본값
            families = ["Floral", "Fresh"]
            key_notes = ["Bergamot", "Rose", "Sandalwood"]
            mood = "balanced"
            season = "all seasons"
            intensity = "moderate"
            occasion = "versatile"
            style = "modern"
            color = "Neutral"
            
            # 파싱
            for line in lines:
                line_lower = line.lower()
                if "families:" in line_lower:
                    families = [f.strip() for f in line.split(':')[1].split(',')]
                elif "mood:" in line_lower:
                    mood = line.split(':')[1].strip()
                elif "season:" in line_lower:
                    season = line.split(':')[1].strip()
                elif "intensity:" in line_lower:
                    intensity = line.split(':')[1].strip()
                elif "occasion:" in line_lower:
                    occasion = line.split(':')[1].strip()
                elif "style:" in line_lower:
                    style = line.split(':')[1].strip()
                elif "color impression:" in line_lower:
                    color = line.split(':')[1].strip()
                elif "top notes:" in line_lower:
                    top_notes = [n.strip() for n in line.split(':')[1].split(',')]
                elif "heart notes:" in line_lower or "middle notes:" in line_lower:
                    heart_notes = [n.strip() for n in line.split(':')[1].split(',')]
                elif "base notes:" in line_lower:
                    base_notes = [n.strip() for n in line.split(':')[1].split(',')]
            
            # 모든 노트 합치기
            if 'top_notes' in locals() and 'heart_notes' in locals() and 'base_notes' in locals():
                key_notes = top_notes[:3] + heart_notes[:3] + base_notes[:3]
            
            return PerfumeInterpretation(
                fragrance_families=families,
                key_notes=key_notes,
                mood=mood,
                season=season,
                intensity=intensity,
                occasion=occasion,
                style=style,
                color_impression=color,
                emotional_profile=self._analyze_emotions(original_description),
                memory_triggers=self._extract_memories(original_description),
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return self._fallback_interpretation(original_description)
    
    def _analyze_emotions(self, description: str) -> Dict[str, float]:
        """감정 분석"""
        emotions = {
            "joy": 0.0,
            "romance": 0.0,
            "nostalgia": 0.0,
            "serenity": 0.0,
            "mystery": 0.0,
            "energy": 0.0
        }
        
        description_lower = description.lower()
        
        # 키워드 기반 감정 점수
        if any(word in description_lower for word in ["happy", "joy", "bright", "기쁜"]):
            emotions["joy"] = 0.8
        if any(word in description_lower for word in ["love", "romantic", "사랑", "로맨틱"]):
            emotions["romance"] = 0.9
        if any(word in description_lower for word in ["memory", "remember", "추억", "기억"]):
            emotions["nostalgia"] = 0.7
        if any(word in description_lower for word in ["calm", "peaceful", "평화", "고요"]):
            emotions["serenity"] = 0.8
        if any(word in description_lower for word in ["mysterious", "dark", "신비", "미스터리"]):
            emotions["mystery"] = 0.7
        if any(word in description_lower for word in ["energetic", "vibrant", "활기", "에너지"]):
            emotions["energy"] = 0.8
            
        # 정규화
        total = sum(emotions.values())
        if total > 0:
            for key in emotions:
                emotions[key] = emotions[key] / total
                
        return emotions
    
    def _extract_memories(self, description: str) -> List[str]:
        """기억/연상 추출"""
        memories = []
        
        description_lower = description.lower()
        
        # 장소
        if any(word in description_lower for word in ["beach", "ocean", "sea", "바다", "해변"]):
            memories.append("Seaside vacation")
        if any(word in description_lower for word in ["garden", "flower", "정원", "꽃밭"]):
            memories.append("Blooming garden")
        if any(word in description_lower for word in ["forest", "woods", "숲", "나무"]):
            memories.append("Forest walk")
            
        # 시간
        if any(word in description_lower for word in ["childhood", "young", "어린시절", "유년"]):
            memories.append("Childhood memories")
        if any(word in description_lower for word in ["first love", "첫사랑"]):
            memories.append("First love")
            
        # 계절/날씨
        if any(word in description_lower for word in ["rain", "rainy", "비", "빗속"]):
            memories.append("Rainy day")
        if any(word in description_lower for word in ["snow", "winter", "눈", "겨울"]):
            memories.append("Winter wonderland")
            
        if not memories:
            memories = ["Personal moments"]
            
        return memories
    
    def _fallback_interpretation(self, description: str) -> PerfumeInterpretation:
        """폴백 해석"""
        return PerfumeInterpretation(
            fragrance_families=["Floral", "Fresh"],
            key_notes=["Bergamot", "Rose", "Jasmine", "Sandalwood", "Musk"],
            mood="balanced",
            season="all seasons",
            intensity="moderate",
            occasion="versatile",
            style="modern classic",
            color_impression="Soft pastels",
            emotional_profile={
                "joy": 0.3,
                "romance": 0.3,
                "serenity": 0.4
            },
            memory_triggers=["Pleasant memories"],
            confidence=0.5
        )
    
    async def enhance_with_storytelling(
        self,
        interpretation: PerfumeInterpretation,
        original_description: str
    ) -> str:
        """
        해석을 스토리텔링으로 강화
        """
        story = f"""
🌸 Your Personalized Fragrance Journey 🌸

Based on your beautiful description: "{original_description[:100]}..."

We've crafted a fragrance that captures:

✨ **The Essence**: A {interpretation.mood} composition that blends {', '.join(interpretation.fragrance_families)} harmonies

🎭 **The Character**: {interpretation.style.title()} in spirit, with {interpretation.intensity} presence that speaks to {interpretation.occasion}

🎨 **The Palette**: Evoking {interpretation.color_impression}, this scent paints your aura with:
  • Opening Notes: {', '.join(interpretation.key_notes[:3])} - The first impression
  • Heart Notes: {', '.join(interpretation.key_notes[3:6] if len(interpretation.key_notes) > 3 else ['Rose'])} - The soul of your fragrance
  • Base Notes: {', '.join(interpretation.key_notes[6:9] if len(interpretation.key_notes) > 6 else ['Sandalwood'])} - The lasting memory

💭 **The Emotions**: 
"""
        
        # 감정 프로필 추가
        for emotion, value in interpretation.emotional_profile.items():
            if value > 0.2:
                percentage = int(value * 100)
                story += f"  • {emotion.title()}: {'■' * (percentage // 10)}{'□' * (10 - percentage // 10)} {percentage}%\n"
        
        story += f"""
🌟 **Memory Triggers**: {', '.join(interpretation.memory_triggers)}

🌺 **Best Worn**: During {interpretation.season}, when you want to feel {interpretation.mood}

This is more than a fragrance - it's your story in scent form.
"""
        
        return story


# 전역 인스턴스
_description_llm_instance = None

def get_description_llm() -> PerfumeDescriptionLLM:
    """향수 설명 LLM 싱글톤"""
    global _description_llm_instance
    if _description_llm_instance is None:
        _description_llm_instance = PerfumeDescriptionLLM()
    return _description_llm_instance


async def interpret_customer_description(
    description: str,
    enhance_story: bool = True
) -> Dict[str, Any]:
    """
    고객 설명 해석 메인 함수
    
    Args:
        description: 고객의 향수 설명
        enhance_story: 스토리텔링 강화 여부
        
    Returns:
        해석 결과와 스토리
    """
    llm = get_description_llm()
    
    # 해석
    interpretation = await llm.interpret_description(description)
    
    result = {
        'interpretation': interpretation,
        'confidence': interpretation.confidence
    }
    
    # 스토리텔링 추가
    if enhance_story:
        story = await llm.enhance_with_storytelling(interpretation, description)
        result['story'] = story
        
    return result
