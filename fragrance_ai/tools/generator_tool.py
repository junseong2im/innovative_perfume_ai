"""
향수 레시피 생성 도구
- 전문 LLM을 사용한 창의적 레시피 생성
- 기존 generator.py 모델과 인터페이스
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import json
import torch
from datetime import datetime
from ..utils.season_mapper import get_season_mapper

logger = logging.getLogger(__name__)

# Pydantic 스키마
class GenerationRequest(BaseModel):
    """레시피 생성 요청"""
    description: str = Field(..., description="원하는 향수 설명")
    style: Optional[str] = Field(default=None, description="스타일: modern, classic, avant-garde")
    fragrance_family: Optional[str] = Field(default="floral", description="향수 계열")
    mood: Optional[str] = Field(default="romantic", description="무드/분위기")
    season: Optional[str] = Field(default=None, description="계절: spring, summer, fall, winter")
    gender: Optional[str] = Field(default="unisex", description="성별: masculine, feminine, unisex")
    intensity: Optional[str] = Field(default="moderate", description="강도: light, moderate, strong")

    # 고급 옵션
    key_notes: Optional[List[str]] = Field(default=None, description="반드시 포함할 노트")
    avoid_notes: Optional[List[str]] = Field(default=None, description="제외할 노트")
    inspiration: Optional[str] = Field(default=None, description="영감/참조")
    price_range: Optional[str] = Field(default="premium", description="가격대: budget, mid, premium, luxury")

class NoteComponent(BaseModel):
    """향료 구성 요소"""
    name: str = Field(..., description="향료 이름")
    percentage: float = Field(..., description="비율 (%)")
    role: str = Field(default="accent", description="역할: primary, accent, modifier")
    description: Optional[str] = Field(default=None, description="노트 설명")

class GeneratedRecipe(BaseModel):
    """생성된 레시피"""
    recipe_id: str = Field(..., description="레시피 ID")
    name: str = Field(..., description="향수 이름")
    description: str = Field(..., description="향수 설명")

    # 노트 구성
    top_notes: List[NoteComponent] = Field(..., description="탑 노트")
    heart_notes: List[NoteComponent] = Field(..., description="하트/미들 노트")
    base_notes: List[NoteComponent] = Field(..., description="베이스 노트")

    # 특성
    fragrance_family: str = Field(..., description="향수 계열")
    character: str = Field(..., description="향수 캐릭터")
    longevity: str = Field(..., description="지속시간")
    sillage: str = Field(..., description="확산성")

    # 메타데이터
    total_ingredients: int = Field(..., description="총 재료 수")
    complexity_level: str = Field(..., description="복잡도: simple, moderate, complex")
    estimated_cost: str = Field(..., description="예상 비용")

    # 제조 노트
    special_instructions: Optional[str] = Field(default=None, description="특별 제조 지침")
    maturation_time: Optional[str] = Field(default="4-6 weeks", description="숙성 기간")

    # 크리에이티브 노트
    story: Optional[str] = Field(default=None, description="향수 스토리")
    wearing_occasions: Optional[List[str]] = Field(default=None, description="착용 상황")

class SpecialistGenerator:
    """전문 향수 생성 모델"""

    def __init__(self, config_path: str = "configs/local.json"):
        """생성기 초기화"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.season_mapper = get_season_mapper()
        self._initialize_model()

    def _load_config(self, config_path: str) -> dict:
        """설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('llm_specialist_generator', {})
        except Exception as e:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            return {
                "model_name_or_path": "beomi/KoAlpaca-Polyglot-5.8B",
                "quantization": "4bit"
            }

    def _initialize_model(self):
        """모델 초기화"""
        try:
            # Transformers 모델 사용 시도
            from ..llm.transformers_loader import (
                KoreanPerfumeGenerator,
                check_transformers_availability
            )

            if check_transformers_availability():
                logger.info("Initializing Korean Perfume Generator with Transformers...")
                self.model = KoreanPerfumeGenerator()
                self.use_llm = True
                logger.info("LLM-based generator initialized")
            else:
                logger.warning("Transformers not available. Using rule-based fallback.")
                self.model = None
                self.use_llm = False

        except ImportError as e:
            logger.warning(f"Model initialization failed: {e}. Using fallback.")
            self.model = None
            self.use_llm = False

    def generate(self, request: GenerationRequest) -> GeneratedRecipe:
        """레시피 생성"""
        try:
            if self.model is not None:
                # 실제 모델 사용
                return self._generate_with_model(request)
            else:
                # 폴백: 규칙 기반 생성
                return self._generate_with_rules(request)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._generate_fallback(request)

    def _generate_with_model(self, request: GenerationRequest) -> GeneratedRecipe:
        """실제 모델로 생성"""
        if hasattr(self.model, 'generate_perfume_recipe'):
            # 한국어 향수 생성 모델 사용
            recipe_data = self.model.generate_perfume_recipe(
                request=request.description,
                style=request.style,
                intensity=request.intensity
            )

            # 데이터를 GeneratedRecipe 형식으로 변환
            import uuid
            recipe_id = str(uuid.uuid4())[:8]

            return GeneratedRecipe(
                recipe_id=recipe_id,
                name=recipe_data.get("name", f"Artisan Creation #{recipe_id}"),
                description=recipe_data.get("description", request.description),
                top_notes=[
                    NoteComponent(name=note, percentage=10.0, role="primary")
                    for note in recipe_data.get("top_notes", ["Bergamot"])[:3]
                ],
                heart_notes=[
                    NoteComponent(name=note, percentage=15.0, role="primary")
                    for note in recipe_data.get("heart_notes", ["Rose"])[:3]
                ],
                base_notes=[
                    NoteComponent(name=note, percentage=12.0, role="primary")
                    for note in recipe_data.get("base_notes", ["Sandalwood"])[:3]
                ],
                fragrance_family=request.fragrance_family or "floral",
                character=recipe_data.get("mood", "Elegant"),
                longevity="6-8 hours",
                sillage="Moderate",
                total_ingredients=9,
                complexity_level="moderate",
                estimated_cost="Premium",
                story=recipe_data.get("description", "")
            )
        else:
            # 기존 방식
            prompt = self._build_prompt(request)
            raw_output = self.model.generate(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.8
            )
            return self._parse_model_output(raw_output, request)

    def _build_prompt(self, request: GenerationRequest) -> str:
        """프롬프트 생성"""
        prompt_parts = [
            f"Create a perfume recipe with the following requirements:",
            f"Description: {request.description}",
            f"Style: {request.style or 'modern'}",
            f"Fragrance Family: {request.fragrance_family}",
            f"Mood: {request.mood}",
            f"Season: {request.season or 'all seasons'}",
            f"Gender: {request.gender}",
            f"Intensity: {request.intensity}"
        ]

        if request.key_notes:
            prompt_parts.append(f"Must include: {', '.join(request.key_notes)}")

        if request.avoid_notes:
            prompt_parts.append(f"Must avoid: {', '.join(request.avoid_notes)}")

        if request.inspiration:
            prompt_parts.append(f"Inspired by: {request.inspiration}")

        prompt_parts.append("\nGenerate a detailed perfume formula with exact percentages:")

        return "\n".join(prompt_parts)

    def _parse_model_output(self, output: str, request: GenerationRequest) -> GeneratedRecipe:
        """모델 출력 파싱"""
        # 여기서는 구조화된 출력을 가정
        # 실제로는 더 복잡한 파싱 로직 필요

        import uuid
        recipe_id = str(uuid.uuid4())[:8]

        # 기본 구조로 파싱 시도
        return GeneratedRecipe(
            recipe_id=recipe_id,
            name=f"Artisan Creation #{recipe_id}",
            description=request.description,
            top_notes=[
                NoteComponent(name="Bergamot", percentage=15.0, role="primary"),
                NoteComponent(name="Lemon", percentage=10.0, role="accent"),
                NoteComponent(name="Pink Pepper", percentage=5.0, role="modifier")
            ],
            heart_notes=[
                NoteComponent(name="Rose", percentage=20.0, role="primary"),
                NoteComponent(name="Jasmine", percentage=15.0, role="accent"),
                NoteComponent(name="Iris", percentage=10.0, role="accent")
            ],
            base_notes=[
                NoteComponent(name="Sandalwood", percentage=15.0, role="primary"),
                NoteComponent(name="Musk", percentage=5.0, role="accent"),
                NoteComponent(name="Amber", percentage=5.0, role="modifier")
            ],
            fragrance_family=request.fragrance_family or "floral",
            character="Elegant and sophisticated",
            longevity="6-8 hours",
            sillage="Moderate to strong",
            total_ingredients=9,
            complexity_level="moderate",
            estimated_cost="Premium",
            story=f"A fragrance inspired by {request.description}"
        )

    def _generate_with_rules(self, request: GenerationRequest) -> GeneratedRecipe:
        """규칙 기반 생성"""
        import uuid
        recipe_id = str(uuid.uuid4())[:8]

        # 계절 확인 및 적용
        season = None
        if request.season:
            season = self.season_mapper.identify_season_from_text(request.season)
        elif request.description:
            season = self.season_mapper.identify_season_from_text(request.description)

        # 계절별 템플릿 우선, 아니면 향수 계열별 기본 템플릿
        if season and season.value == "winter":
            # 겨울 향수 템플릿
            template = {
                "top": ["Bergamot", "Black Pepper", "Pink Pepper"],
                "heart": ["Rose", "Jasmine", "Tuberose"],
                "base": ["Vanilla", "Amber", "Musk", "Oud", "Sandalwood"]
            }
            family = "oriental"
        elif season and season.value == "summer":
            # 여름 향수 템플릿
            template = {
                "top": ["Lemon", "Lime", "Grapefruit"],
                "heart": ["Neroli", "Orange Blossom", "Coconut"],
                "base": ["Light Musk", "Driftwood", "Sea Salt"]
            }
            family = "citrus"
        else:
            # 기존 템플릿 로직
            templates = {
                "floral": {
                    "top": ["Bergamot", "Neroli", "Peach"],
                    "heart": ["Rose", "Jasmine", "Ylang-ylang"],
                    "base": ["Sandalwood", "Musk", "Vanilla"]
                },
                "woody": {
                    "top": ["Bergamot", "Black Pepper", "Cardamom"],
                    "heart": ["Cedar", "Vetiver", "Iris"],
                    "base": ["Sandalwood", "Patchouli", "Amber"]
                },
                "citrus": {
                    "top": ["Lemon", "Bergamot", "Grapefruit"],
                    "heart": ["Neroli", "Petitgrain", "Green Tea"],
                    "base": ["White Musk", "Cedar", "Amber"]
                },
                "oriental": {
                    "top": ["Bergamot", "Cinnamon", "Cardamom"],
                    "heart": ["Rose", "Jasmine", "Orchid"],
                    "base": ["Vanilla", "Amber", "Benzoin"]
                }
            }

            # 템플릿 선택
            family = request.fragrance_family or "floral"
            if family not in templates:
                family = "floral"
            template = templates[family]

        # 노트 구성
        top_notes = []
        for i, note in enumerate(template["top"]):
            if request.avoid_notes and note in request.avoid_notes:
                continue
            percentage = 15.0 - (i * 3.0)
            role = "primary" if i == 0 else "accent"
            top_notes.append(NoteComponent(name=note, percentage=percentage, role=role))

        heart_notes = []
        for i, note in enumerate(template["heart"]):
            if request.avoid_notes and note in request.avoid_notes:
                continue
            percentage = 20.0 - (i * 3.0)
            role = "primary" if i == 0 else "accent"
            heart_notes.append(NoteComponent(name=note, percentage=percentage, role=role))

        base_notes = []
        for i, note in enumerate(template["base"]):
            if request.avoid_notes and note in request.avoid_notes:
                continue
            percentage = 15.0 - (i * 3.0)
            role = "primary" if i == 0 else "modifier"
            base_notes.append(NoteComponent(name=note, percentage=percentage, role=role))

        # 계절에 맞는 강도 특성 적용
        if season:
            season_intensity = self.season_mapper.get_season_intensity(season)
            longevity = season_intensity["longevity"]
            sillage = season_intensity["sillage"]
        else:
            # 강도별 특성
            intensity_map = {
                "light": ("4-6 hours", "Light to moderate"),
                "moderate": ("6-8 hours", "Moderate to strong"),
                "strong": ("8-12 hours", "Strong to heavy")
            }
            longevity, sillage = intensity_map.get(request.intensity or "moderate", ("6-8 hours", "Moderate"))

        # 추천 계절 결정
        if season:
            if season.value == "winter":
                recommended_season = "가을/겨울"
            elif season.value == "summer":
                recommended_season = "봄/여름"
            elif season.value == "spring":
                recommended_season = "봄/초여름"
            elif season.value == "autumn" or season.value == "fall":
                recommended_season = "가을"
            else:
                recommended_season = "사계절"
        else:
            # 노트 구성에 따라 추천 계절 결정
            notes_dict = {"top": [n.name for n in top_notes], "heart": [n.name for n in heart_notes], "base": [n.name for n in base_notes]}
            recommended_seasons = self.season_mapper.get_recommended_season_for_notes(notes_dict)
            if recommended_seasons:
                if any(s.value in ["winter", "autumn"] for s in recommended_seasons):
                    recommended_season = "가을/겨울"
                elif any(s.value in ["spring", "summer"] for s in recommended_seasons):
                    recommended_season = "봄/여름"
                else:
                    recommended_season = "사계절"
            else:
                recommended_season = "사계절"

        return GeneratedRecipe(
            recipe_id=recipe_id,
            name=f"Artisan {family.title()} #{recipe_id}",
            description=f"{request.description} - A {request.mood} {family} fragrance for {recommended_season}",
            top_notes=top_notes,
            heart_notes=heart_notes,
            base_notes=base_notes,
            fragrance_family=family,
            character=f"{request.mood.title() if request.mood else 'Elegant'} and {request.style or 'modern'}",
            longevity=longevity,
            sillage=sillage,
            total_ingredients=len(top_notes) + len(heart_notes) + len(base_notes),
            complexity_level="moderate",
            estimated_cost=request.price_range or "premium",
            story=f"Inspired by {request.description}, this {recommended_season} fragrance captures the essence of {request.mood or 'sophistication'} in a {family} composition.",
            wearing_occasions=self._suggest_occasions(request)
        )

    def _suggest_occasions(self, request: GenerationRequest) -> List[str]:
        """착용 상황 제안"""
        occasions = []

        mood_occasions = {
            "romantic": ["Date night", "Anniversary", "Evening events"],
            "fresh": ["Daily wear", "Office", "Casual outings"],
            "mysterious": ["Special events", "Night out", "Gala"],
            "energetic": ["Sports", "Outdoor activities", "Daytime"],
            "sophisticated": ["Business meetings", "Formal events", "Dinner parties"]
        }

        # 계절별 정확한 상황 매칭
        season_occasions = {
            "spring": ["Garden parties", "Brunch", "Weddings"],
            "summer": ["Beach", "Vacation", "Outdoor festivals"],
            "fall": ["Cozy gatherings", "Wine tasting", "Theater"],
            "autumn": ["Cozy gatherings", "Wine tasting", "Theater"],
            "winter": ["Holiday parties", "Indoor events", "Fireplace evenings"],
            "겨울": ["Holiday parties", "Indoor events", "Fireplace evenings"],
            "여름": ["Beach", "Vacation", "Outdoor festivals"],
            "봄": ["Garden parties", "Brunch", "Weddings"],
            "가을": ["Cozy gatherings", "Wine tasting", "Theater"]
        }

        if request.mood in mood_occasions:
            occasions.extend(mood_occasions[request.mood])

        if request.season in season_occasions:
            occasions.extend(season_occasions[request.season])

        if not occasions:
            occasions = ["Versatile - suitable for various occasions"]

        return occasions[:5]  # 최대 5개

    def _generate_fallback(self, request: GenerationRequest) -> GeneratedRecipe:
        """최종 폴백 레시피"""
        import uuid
        recipe_id = str(uuid.uuid4())[:8]

        return GeneratedRecipe(
            recipe_id=recipe_id,
            name=f"Simple Creation #{recipe_id}",
            description=request.description,
            top_notes=[
                NoteComponent(name="Citrus Blend", percentage=30.0, role="primary")
            ],
            heart_notes=[
                NoteComponent(name="Floral Bouquet", percentage=40.0, role="primary")
            ],
            base_notes=[
                NoteComponent(name="Woody Base", percentage=30.0, role="primary")
            ],
            fragrance_family="universal",
            character="Balanced and harmonious",
            longevity="6 hours",
            sillage="Moderate",
            total_ingredients=3,
            complexity_level="simple",
            estimated_cost="Budget-friendly",
            story="A simple yet elegant fragrance"
        )

# 전역 생성기 인스턴스
generator_instance = None

def get_generator():
    """생성기 인스턴스 가져오기"""
    global generator_instance
    if generator_instance is None:
        generator_instance = SpecialistGenerator()
    return generator_instance

async def create_recipe(request: GenerationRequest) -> GeneratedRecipe:
    """
    # LLM TOOL DESCRIPTION (FOR ORCHESTRATOR)
    # Use this tool to generate creative perfume recipes.
    # This interfaces with our specialized fragrance generation model.
    # Provide detailed requirements for best results.

    Args:
        request: 레시피 생성 요청 정보

    Returns:
        GeneratedRecipe: 생성된 향수 레시피
    """
    generator = get_generator()
    return generator.generate(request)