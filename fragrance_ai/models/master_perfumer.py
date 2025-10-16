"""
마스터 조향사 AI 모델
딥러닝 기반 향수 레시피 생성 및 대화형 AI 시스템
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class FragranceKnowledgeBase:
    """향수 지식 베이스 - 딥러닝 훈련 데이터"""

    def __init__(self):
        self.fragrance_notes = {
            "citrus": ["bergamot", "lemon", "orange", "grapefruit", "lime", "mandarin", "yuzu"],
            "floral": ["rose", "jasmine", "ylang-ylang", "neroli", "tuberose", "lily", "peony", "violet"],
            "woody": ["sandalwood", "cedarwood", "patchouli", "vetiver", "oak", "pine", "birch"],
            "oriental": ["vanilla", "amber", "musk", "oud", "frankincense", "myrrh", "benzoin"],
            "fresh": ["mint", "eucalyptus", "lavender", "rosemary", "basil", "sage", "thyme"],
            "spicy": ["cinnamon", "clove", "nutmeg", "pepper", "ginger", "cardamom", "star anise"],
            "fruity": ["apple", "pear", "peach", "apricot", "blackcurrant", "strawberry", "plum"],
            "green": ["grass", "leaves", "stems", "moss", "fern", "ivy", "petitgrain"]
        }

        self.perfume_compositions = self._load_master_compositions()
        self.seasonal_preferences = self._load_seasonal_data()
        self.cultural_preferences = self._load_cultural_data()

    def _load_master_compositions(self):
        """마스터 조향사들의 명작 향수 구성 데이터"""
        return {
            "classic_chypre": {
                "top": ["bergamot", "lemon", "aldehydes"],
                "middle": ["rose", "jasmine", "ylang-ylang", "geranium"],
                "base": ["oakmoss", "patchouli", "amber", "musk"],
                "style": "sophisticated, elegant, timeless",
                "creator": "François Coty"
            },
            "oriental_spicy": {
                "top": ["cinnamon", "orange", "bergamot"],
                "middle": ["rose", "clove", "nutmeg", "carnation"],
                "base": ["vanilla", "amber", "oud", "sandalwood"],
                "style": "warm, mysterious, powerful",
                "creator": "Jacques Guerlain"
            },
            "fresh_marine": {
                "top": ["sea breeze", "lemon", "mint"],
                "middle": ["geranium", "rosemary", "lavender"],
                "base": ["white musk", "cedar", "ambergris"],
                "style": "clean, invigorating, modern",
                "creator": "Olivier Cresp"
            },
            "gourmand_modern": {
                "top": ["pink pepper", "mandarin", "coffee"],
                "middle": ["chocolate", "caramel", "rose"],
                "base": ["vanilla", "praline", "white musk"],
                "style": "edible, comforting, innovative",
                "creator": "Thierry Mugler"
            }
        }

    def _load_seasonal_data(self):
        """계절별 선호도 데이터"""
        return {
            "spring": {
                "preferred_families": ["floral", "fresh", "green"],
                "intensity": "light to medium",
                "popular_notes": ["cherry blossom", "lily of the valley", "green leaves", "rain"]
            },
            "summer": {
                "preferred_families": ["citrus", "fresh", "marine"],
                "intensity": "light",
                "popular_notes": ["sea breeze", "coconut", "citrus", "mint"]
            },
            "autumn": {
                "preferred_families": ["woody", "spicy", "oriental"],
                "intensity": "medium to strong",
                "popular_notes": ["amber", "cinnamon", "apple", "leather"]
            },
            "winter": {
                "preferred_families": ["oriental", "woody", "gourmand"],
                "intensity": "strong",
                "popular_notes": ["vanilla", "oud", "tobacco", "chocolate"]
            }
        }

    def _load_cultural_data(self):
        """문화적 선호도 데이터"""
        return {
            "korean": {
                "traditional_notes": ["pine", "bamboo", "chrysanthemum", "ginseng"],
                "modern_preferences": ["clean", "subtle", "sophisticated"],
                "popular_styles": ["minimalist", "elegant", "natural"]
            },
            "japanese": {
                "traditional_notes": ["cherry blossom", "green tea", "bamboo", "incense"],
                "modern_preferences": ["zen", "pure", "refined"],
                "popular_styles": ["minimalist", "harmony", "seasonal"]
            },
            "french": {
                "traditional_notes": ["lavender", "rose", "iris", "vetiver"],
                "modern_preferences": ["sophisticated", "complex", "artistic"],
                "popular_styles": ["classic", "elegant", "luxurious"]
            }
        }

class MasterPerfumerAI(nn.Module):
    """마스터 조향사 AI - GPT-2 기반 대화형 향수 전문가"""

    def __init__(self, model_name="gpt2", use_quantization=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.knowledge_base = FragranceKnowledgeBase()

        # 모델 설정
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.model.to(self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 향수 전문 프롬프트 템플릿
        self.system_prompt = self._create_system_prompt()

        logger.info(f"마스터 조향사 AI 초기화 완료 - 모델: {model_name}, 디바이스: {self.device}")

    def _create_system_prompt(self):
        """향수 전문가 시스템 프롬프트"""
        return """당신은 세계적인 마스터 조향사입니다. 50년 경력의 향수 전문가로서:

전문 지식:
- 3000가지 이상의 향료 성분과 조합 방법
- 계절별, 문화별, 개인별 맞춤 향수 제조
- 향의 발향 과정과 지속성 조절
- 조화로운 블렌딩과 혁신적인 창조

대화 스타일:
- 고객의 요청을 깊이 이해하고 분석
- 전문적이면서도 친근한 조언
- 창의적이고 예술적인 접근
- 과학적 근거와 감성적 표현의 조화

향수 제작 시:
- 탑/미들/베이스 노트의 완벽한 균형
- 고객의 라이프스타일과 개성 반영
- 독창적이면서도 착용 가능한 조합
- 시간에 따른 향의 변화 설계"""

    async def generate_fragrance(self, customer_input: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        """대화형 향수 생성"""
        try:
            # 대화 맥락 구성
            context = self._build_conversation_context(customer_input, conversation_history)

            # LLM을 통한 향수 분석 및 생성
            fragrance_analysis = await self._analyze_customer_needs(context)

            # 딥러닝 기반 레시피 생성
            recipe = await self._generate_recipe(fragrance_analysis)

            # 전문가급 품질 검증 및 개선
            refined_recipe = await self._refine_recipe(recipe, customer_input)

            # 상세한 설명 생성
            detailed_explanation = await self._create_detailed_explanation(refined_recipe, customer_input)

            return {
                "recipe": refined_recipe,
                "explanation": detailed_explanation,
                "confidence": refined_recipe.get("confidence", 0.85),
                "conversation_ready": True
            }

        except Exception as e:
            logger.error(f"향수 생성 실패: {e}")
            # 실패 시 목업 대답 대신 예외 발생으로 처리
            raise Exception("응답 실패했습니다")

    def _build_conversation_context(self, current_input: str, history: List[str] = None) -> str:
        """대화 맥락 구성"""
        context = f"{self.system_prompt}\n\n"

        if history:
            context += "이전 대화:\n"
            for msg in history[-3:]:  # 최근 3개만 유지
                context += f"{msg}\n"

        context += f"\n고객 요청: {current_input}\n\n"
        context += "마스터 조향사로서 이 고객을 위한 완벽한 향수를 만들어주세요:"

        return context

    async def _analyze_customer_needs(self, context: str) -> Dict[str, Any]:
        """고객 니즈 분석"""
        analysis_prompt = f"{context}\n\n먼저 고객의 요청을 분석해주세요:\n1. 원하는 향의 스타일\n2. 선호하는 계절/상황\n3. 개성과 라이프스타일\n4. 문화적 배경\n\n분석:"

        # LLM 추론
        inputs = self.tokenizer.encode(analysis_prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 150,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = analysis_text[len(analysis_prompt):]

        # 구조화된 분석 결과 생성
        return self._parse_analysis(analysis)

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """분석 텍스트를 구조화된 데이터로 변환"""
        # 키워드 기반 분석
        analysis = {
            "style_preferences": [],
            "intensity": "medium",
            "season": "all-season",
            "cultural_context": "universal",
            "mood": "balanced",
            "occasion": "daily"
        }

        text_lower = analysis_text.lower()

        # 스타일 분석
        if any(word in text_lower for word in ["fresh", "clean", "light"]):
            analysis["style_preferences"].append("fresh")
        if any(word in text_lower for word in ["warm", "cozy", "comfort"]):
            analysis["style_preferences"].append("warm")
        if any(word in text_lower for word in ["elegant", "sophisticated", "luxury"]):
            analysis["style_preferences"].append("elegant")
        if any(word in text_lower for word in ["bold", "strong", "powerful"]):
            analysis["style_preferences"].append("bold")

        # 강도 분석
        if any(word in text_lower for word in ["light", "subtle", "gentle"]):
            analysis["intensity"] = "light"
        elif any(word in text_lower for word in ["strong", "bold", "intense"]):
            analysis["intensity"] = "strong"

        return analysis

    async def _generate_recipe(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """딥러닝 기반 레시피 생성"""
        # 지식 베이스에서 적합한 조합 찾기
        base_composition = self._select_base_composition(analysis)

        # AI 모델을 통한 창의적 조합
        creative_prompt = self._create_recipe_prompt(analysis, base_composition)

        inputs = self.tokenizer.encode(creative_prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 200,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        recipe_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        recipe_part = recipe_text[len(creative_prompt):]

        # 구조화된 레시피 생성
        return self._parse_recipe(recipe_part, base_composition, analysis)

    def _select_base_composition(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 바탕으로 기본 구성 선택"""
        compositions = self.knowledge_base.perfume_compositions

        # 스타일에 따른 기본 구성 선택
        if "elegant" in analysis["style_preferences"]:
            return compositions["classic_chypre"]
        elif "bold" in analysis["style_preferences"]:
            return compositions["oriental_spicy"]
        elif "fresh" in analysis["style_preferences"]:
            return compositions["fresh_marine"]
        else:
            return compositions["gourmand_modern"]

    def _create_recipe_prompt(self, analysis: Dict[str, Any], base: Dict[str, Any]) -> str:
        """레시피 생성 프롬프트"""
        return f"""기본 구성: {base['style']}

고객 분석:
- 스타일: {', '.join(analysis['style_preferences'])}
- 강도: {analysis['intensity']}
- 분위기: {analysis['mood']}

이 정보를 바탕으로 창의적이고 조화로운 향수 레시피를 만들어주세요:

탑 노트 (첫 15분):
미들 노트 (15분-4시간):
베이스 노트 (4시간 이상):

각 노트의 구체적인 성분과 비율을 제시해주세요:"""

    def _parse_recipe(self, recipe_text: str, base_composition: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """생성된 레시피 텍스트를 구조화"""
        # 기본 구성에서 시작
        recipe = {
            "concept": f"{analysis.get('mood', 'balanced')} {analysis.get('intensity', 'medium')} fragrance",
            "top_notes": base_composition["top"][:2],  # 기본 2개
            "middle_notes": base_composition["middle"][:3],  # 기본 3개
            "base_notes": base_composition["base"][:3],  # 기본 3개
            "proportions": {
                "top_notes": "30%",
                "middle_notes": "50%",
                "base_notes": "20%"
            },
            "intensity": analysis.get("intensity", "medium"),
            "longevity": "6-8 hours",
            "sillage": "moderate",
            "confidence": 0.85
        }

        # AI 생성 텍스트에서 추가 노트 추출
        additional_notes = self._extract_notes_from_text(recipe_text)

        # 창의적 조합 적용
        if additional_notes["top"]:
            recipe["top_notes"].extend(additional_notes["top"][:2])
        if additional_notes["middle"]:
            recipe["middle_notes"].extend(additional_notes["middle"][:2])
        if additional_notes["base"]:
            recipe["base_notes"].extend(additional_notes["base"][:2])

        # 중복 제거
        recipe["top_notes"] = list(set(recipe["top_notes"]))[:4]
        recipe["middle_notes"] = list(set(recipe["middle_notes"]))[:5]
        recipe["base_notes"] = list(set(recipe["base_notes"]))[:4]

        return recipe

    def _extract_notes_from_text(self, text: str) -> Dict[str, List[str]]:
        """텍스트에서 향료 노트 추출"""
        notes = {"top": [], "middle": [], "base": []}

        # 모든 향료 리스트
        all_notes = []
        for category in self.knowledge_base.fragrance_notes.values():
            all_notes.extend(category)

        text_lower = text.lower()

        # 텍스트에서 언급된 향료 찾기
        for note in all_notes:
            if note.lower() in text_lower:
                # 간단한 분류 로직 (위치 기반)
                note_pos = text_lower.find(note.lower())
                if note_pos < len(text_lower) * 0.3:
                    notes["top"].append(note)
                elif note_pos < len(text_lower) * 0.7:
                    notes["middle"].append(note)
                else:
                    notes["base"].append(note)

        return notes

    async def _refine_recipe(self, recipe: Dict[str, Any], customer_input: str) -> Dict[str, Any]:
        """전문가급 품질로 레시피 개선"""
        # 조화성 검증
        harmony_score = self._calculate_harmony(recipe)

        # 독창성 검증
        originality_score = self._calculate_originality(recipe)

        # 실제 제작 가능성 검증
        feasibility_score = self._calculate_feasibility(recipe)

        # 전체 품질 점수
        overall_confidence = (harmony_score + originality_score + feasibility_score) / 3
        recipe["confidence"] = overall_confidence

        # 품질이 낮으면 개선
        if overall_confidence < 0.7:
            recipe = self._improve_recipe(recipe)

        return recipe

    def _calculate_harmony(self, recipe: Dict[str, Any]) -> float:
        """조화성 계산"""
        # 향료 조합의 전통적 조화성 검증
        harmony_score = 0.7  # 기본 점수

        # 시트러스 + 플로럴 조합 보너스
        if any("citrus" in str(note) for note in recipe["top_notes"]) and \
           any("floral" in str(note) for note in recipe["middle_notes"]):
            harmony_score += 0.1

        # 우디 + 오리엔탈 베이스 보너스
        if any("wood" in str(note) for note in recipe["base_notes"]) and \
           any("oriental" in str(note) for note in recipe["base_notes"]):
            harmony_score += 0.1

        return min(harmony_score, 1.0)

    def _calculate_originality(self, recipe: Dict[str, Any]) -> float:
        """독창성 계산"""
        # 일반적이지 않은 조합일수록 높은 점수
        uniqueness = 0.6

        # 노트 다양성 검증
        all_notes = recipe["top_notes"] + recipe["middle_notes"] + recipe["base_notes"]
        unique_families = set()

        for note in all_notes:
            for family, notes in self.knowledge_base.fragrance_notes.items():
                if note in notes:
                    unique_families.add(family)

        # 3개 이상 패밀리 조합 시 보너스
        if len(unique_families) >= 3:
            uniqueness += 0.2

        return min(uniqueness, 1.0)

    def _calculate_feasibility(self, recipe: Dict[str, Any]) -> float:
        """실제 제작 가능성 계산"""
        # 실제 존재하는 향료인지 확인
        feasible_notes = 0
        total_notes = 0

        all_recipe_notes = recipe["top_notes"] + recipe["middle_notes"] + recipe["base_notes"]

        for note in all_recipe_notes:
            total_notes += 1
            # 지식 베이스에 있는 노트인지 확인
            for category in self.knowledge_base.fragrance_notes.values():
                if note in category:
                    feasible_notes += 1
                    break

        return feasible_notes / total_notes if total_notes > 0 else 0.5

    def _improve_recipe(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """레시피 개선"""
        # 안전한 대체 향료 추가
        safe_additions = {
            "top_notes": ["bergamot", "lemon"],
            "middle_notes": ["rose", "jasmine"],
            "base_notes": ["sandalwood", "musk"]
        }

        for layer, notes in safe_additions.items():
            if len(recipe[layer]) < 3:
                for note in notes:
                    if note not in recipe[layer]:
                        recipe[layer].append(note)
                        break

        recipe["confidence"] = 0.75  # 개선 후 신뢰도
        return recipe

    async def _create_detailed_explanation(self, recipe: Dict[str, Any], customer_input: str) -> str:
        """상세한 설명 생성"""
        explanation_prompt = f"""
고객 요청: {customer_input}

생성된 레시피:
- 탑 노트: {', '.join(recipe['top_notes'])}
- 미들 노트: {', '.join(recipe['middle_notes'])}
- 베이스 노트: {', '.join(recipe['base_notes'])}

마스터 조향사로서 이 향수에 대한 아름답고 전문적인 설명을 해주세요:
"""

        inputs = self.tokenizer.encode(explanation_prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 250,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )

        explanation_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation_text[len(explanation_prompt):]

        return explanation.strip()

    # _fallback_response 삭제 - 실패 시 예외 발생으로 처리

# 전역 인스턴스
master_perfumer = None

def get_master_perfumer():
    """마스터 조향사 AI 인스턴스 반환"""
    global master_perfumer
    if master_perfumer is None:
        master_perfumer = MasterPerfumerAI()
    return master_perfumer