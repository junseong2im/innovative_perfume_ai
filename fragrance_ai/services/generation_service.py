from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
import json
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.generator import FragranceRecipeGenerator
from ..core.config import settings

logger = logging.getLogger(__name__)

class GenerationService:
    """향수 레시피 생성을 위한 통합 서비스"""
    
    def __init__(self):
        self.generator = FragranceRecipeGenerator()
        self.generation_cache = {}
        self.cache_ttl = 1800  # 30분
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.generation_queue = asyncio.Queue(maxsize=100)
        
    async def initialize(self):
        """서비스 초기화"""
        try:
            await self.generator.initialize()
            logger.info("GenerationService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GenerationService: {e}")
            raise
    
    async def generate_recipe(
        self,
        request_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        향수 레시피 생성
        
        Args:
            request_data: 생성 요청 데이터
            use_cache: 캐시 사용 여부
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # 입력 검증
            self._validate_generation_request(request_data)
            
            # 캐시 확인
            if use_cache:
                cache_key = self._generate_cache_key(request_data)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for generation request: {request_id}")
                    return cached_result
            
            # 생성 타입에 따른 처리
            generation_type = request_data.get("generation_type", "basic_recipe")
            
            if generation_type == "basic_recipe":
                result = await self._generate_basic_recipe(request_data)
            elif generation_type == "detailed_recipe":
                result = await self._generate_detailed_recipe(request_data)
            elif generation_type == "premium_recipe":
                result = await self._generate_premium_recipe(request_data)
            elif generation_type == "variation":
                result = await self._generate_recipe_variation(request_data)
            else:
                raise ValueError(f"Unsupported generation type: {generation_type}")
            
            # 생성 시간 계산
            generation_time = time.time() - start_time
            
            # 품질 평가
            quality_score = await self._evaluate_recipe_quality(result)
            
            # 최종 응답 구성
            response = {
                "request_id": request_id,
                "generation_type": generation_type,
                "recipe": result,
                "quality_score": quality_score,
                "metadata": {
                    "generation_time": round(generation_time, 3),
                    "model_version": self.generator.model_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "parameters": request_data.get("generation_params", {})
                }
            }
            
            # 결과 캐싱
            if use_cache and cache_key:
                self._cache_result(cache_key, response)
            
            logger.info(f"Recipe generated successfully: {request_id} in {generation_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {e}")
            raise
    
    async def _generate_basic_recipe(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본 레시피 생성"""
        prompt_template = """
향수 레시피를 생성해주세요.

요구사항:
- 향조: {fragrance_family}
- 무드: {mood}
- 강도: {intensity}
- 성별: {gender}
- 계절: {season}

다음 형식으로 응답해주세요:
{{
    "name": "향수 이름",
    "description": "향수 설명",
    "fragrance_family": "향조",
    "notes": {{
        "top": ["톱 노트 리스트"],
        "middle": ["미들 노트 리스트"],  
        "base": ["베이스 노트 리스트"]
    }},
    "formula": {{
        "노트명": "농도(%)"
    }},
    "characteristics": {{
        "sillage": "확산성 (1-10)",
        "longevity": "지속성 (1-10)",
        "complexity": "복합성 (1-10)"
    }}
}}
"""
        
        prompt = prompt_template.format(**request_data)
        
        generation_params = request_data.get("generation_params", {})
        generation_params.update({
            "max_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9
        })
        
        result = await self.generator.generate(
            prompt=prompt,
            generation_config=generation_params
        )
        
        return self._parse_recipe_output(result)
    
    async def _generate_detailed_recipe(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """상세 레시피 생성"""
        prompt_template = """
상세한 향수 레시피를 생성해주세요.

요구사항:
- 향조: {fragrance_family}
- 무드: {mood}
- 강도: {intensity}
- 성별: {gender}
- 계절: {season}
- 특별 요청: {special_requirements}

다음 형식으로 상세한 응답을 제공해주세요:
{{
    "name": "향수 이름",
    "description": "상세한 향수 설명",
    "concept": "컨셉트 설명",
    "target_audience": "타겟 고객",
    "fragrance_family": "향조",
    "notes": {{
        "top": [
            {{"name": "노트명", "percentage": 농도, "description": "노트 설명"}}
        ],
        "middle": [
            {{"name": "노트명", "percentage": 농도, "description": "노트 설명"}}
        ],
        "base": [
            {{"name": "노트명", "percentage": 농도, "description": "노트 설명"}}
        ]
    }},
    "formula": {{
        "total_oil_concentration": "총 오일 농도(%)",
        "alcohol_percentage": "알코올 비율(%)",
        "water_percentage": "물 비율(%)",
        "detailed_formula": {{
            "노트명": {{"percentage": 농도, "supplier": "공급업체", "grade": "등급"}}
        }}
    }},
    "production_notes": {{
        "maceration_time": "숙성 시간",
        "aging_requirements": "숙성 조건",
        "stability_notes": "안정성 주의사항",
        "quality_control": "품질 관리 포인트"
    }},
    "characteristics": {{
        "sillage": "확산성 (1-10)",
        "longevity": "지속성 (1-10)", 
        "complexity": "복합성 (1-10)",
        "uniqueness": "독창성 (1-10)",
        "commercial_viability": "상업성 (1-10)"
    }},
    "development_timeline": "개발 일정",
    "estimated_cost": "예상 제조 비용"
}}
"""
        
        prompt = prompt_template.format(**request_data)
        
        generation_params = request_data.get("generation_params", {})
        generation_params.update({
            "max_tokens": 1500,
            "temperature": 0.6,
            "top_p": 0.85
        })
        
        result = await self.generator.generate(
            prompt=prompt,
            generation_config=generation_params
        )
        
        return self._parse_recipe_output(result)
    
    async def _generate_premium_recipe(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """프리미엄 레시피 생성"""
        # 프리미엄 재료 데이터베이스 활용
        premium_ingredients = await self._get_premium_ingredients(
            request_data.get("fragrance_family", "")
        )
        
        prompt_template = """
프리미엄 향수 레시피를 생성해주세요. 고급 원료와 독창적인 조합을 사용하세요.

요구사항:
- 향조: {fragrance_family}
- 무드: {mood}
- 강도: {intensity}
- 성별: {gender}
- 계절: {season}
- 예산 범위: {budget_range}
- 프리미엄 재료 활용: {premium_ingredients}

다음 형식으로 프리미엄 레시피를 제공해주세요:
{{
    "name": "럭셔리 향수 이름",
    "brand_positioning": "브랜드 포지셔닝",
    "description": "프리미엄 향수 설명",
    "artisan_concept": "아티장 컨셉트",
    "heritage_story": "헤리티지 스토리",
    "fragrance_family": "향조",
    "signature_notes": ["시그니처 노트들"],
    "notes": {{
        "top": [
            {{"name": "프리미엄 노트명", "origin": "원산지", "percentage": 농도, "rarity_score": 희귀도, "description": "설명"}}
        ],
        "middle": [
            {{"name": "프리미엄 노트명", "origin": "원산지", "percentage": 농도, "rarity_score": 희귀도, "description": "설명"}}
        ],
        "base": [
            {{"name": "프리미엄 노트명", "origin": "원산지", "percentage": 농도, "rarity_score": 희귀도, "description": "설명"}}
        ]
    }},
    "exclusive_formula": {{
        "total_concentration": "총 농도(%)",
        "oil_quality": "오일 품질 등급",
        "extraction_method": "추출 방법",
        "premium_components": {{
            "노트명": {{"percentage": 농도, "supplier": "독점 공급업체", "certification": "인증", "cost_per_ml": "ml당 비용"}}
        }}
    }},
    "artisan_process": {{
        "maceration_period": "장기 숙성 기간",
        "handcrafted_elements": "수공예 요소",
        "quality_testing": "품질 테스트 과정",
        "limited_production": "한정 생산 수량"
    }},
    "luxury_characteristics": {{
        "sillage": "확산성 (1-10)",
        "longevity": "지속성 (1-10)",
        "complexity": "복합성 (1-10)",
        "exclusivity": "독점성 (1-10)",
        "artisanal_quality": "장인 품질 (1-10)"
    }},
    "packaging_concept": "패키징 컨셉트",
    "retail_strategy": "판매 전략",
    "target_price_range": "목표 가격대",
    "production_timeline": "생산 일정"
}}
"""
        
        request_data["premium_ingredients"] = ", ".join(premium_ingredients)
        prompt = prompt_template.format(**request_data)
        
        generation_params = request_data.get("generation_params", {})
        generation_params.update({
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 0.8
        })
        
        result = await self.generator.generate(
            prompt=prompt,
            generation_config=generation_params
        )
        
        return self._parse_recipe_output(result)
    
    async def _generate_recipe_variation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """기존 레시피의 변형 생성"""
        base_recipe = request_data.get("base_recipe", {})
        variation_type = request_data.get("variation_type", "seasonal")
        
        prompt_template = """
기존 향수 레시피의 {variation_type} 변형을 생성해주세요.

원본 레시피:
{base_recipe}

변형 요구사항:
- 변형 타입: {variation_type}
- 변형 강도: {variation_intensity}
- 유지할 특성: {preserve_characteristics}

변형된 레시피를 다음 형식으로 제공해주세요:
{{
    "original_name": "원본 향수 이름",
    "variation_name": "변형 향수 이름",
    "variation_concept": "변형 컨셉트",
    "changes_summary": "주요 변경 사항",
    "notes": {{
        "top": [변형된 톱 노트들],
        "middle": [변형된 미들 노트들],
        "base": [변형된 베이스 노트들]
    }},
    "formula_changes": {{
        "added_notes": "추가된 노트들",
        "removed_notes": "제거된 노트들", 
        "modified_notes": "수정된 노트들"
    }},
    "characteristics": {{
        "sillage": "확산성 (1-10)",
        "longevity": "지속성 (1-10)",
        "complexity": "복합성 (1-10)"
    }},
    "variation_rationale": "변형 근거"
}}
"""
        
        prompt = prompt_template.format(
            base_recipe=json.dumps(base_recipe, ensure_ascii=False, indent=2),
            **request_data
        )
        
        generation_params = request_data.get("generation_params", {})
        generation_params.update({
            "max_tokens": 1200,
            "temperature": 0.6,
            "top_p": 0.9
        })
        
        result = await self.generator.generate(
            prompt=prompt,
            generation_config=generation_params
        )
        
        return self._parse_recipe_output(result)
    
    async def batch_generate(
        self,
        batch_requests: List[Dict[str, Any]],
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """배치 레시피 생성"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting batch generation: {batch_id} ({len(batch_requests)} requests)")
            
            # 동시 생성 수 제한을 위한 세마포어
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def generate_single(request_data: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    return await self.generate_recipe(request_data)
            
            # 모든 요청을 병렬로 처리
            tasks = [generate_single(req) for req in batch_requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            successful_results = []
            failed_results = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append({
                        "request_index": i,
                        "error": str(result),
                        "request_data": batch_requests[i]
                    })
                else:
                    successful_results.append({
                        "request_index": i,
                        "result": result
                    })
            
            batch_time = time.time() - start_time
            
            logger.info(f"Batch generation completed: {batch_id} "
                       f"({len(successful_results)} success, {len(failed_results)} failed) "
                       f"in {batch_time:.3f}s")
            
            return {
                "batch_id": batch_id,
                "total_requests": len(batch_requests),
                "successful_results": successful_results,
                "failed_results": failed_results,
                "batch_time": round(batch_time, 3),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise
    
    async def _evaluate_recipe_quality(self, recipe: Dict[str, Any]) -> float:
        """레시피 품질 평가"""
        try:
            quality_factors = []
            
            # 구조 완성도 (0-1)
            structure_score = self._evaluate_recipe_structure(recipe)
            quality_factors.append(("structure", structure_score, 0.3))
            
            # 노트 균형성 (0-1)
            balance_score = self._evaluate_note_balance(recipe)
            quality_factors.append(("balance", balance_score, 0.25))
            
            # 창의성 (0-1)
            creativity_score = self._evaluate_creativity(recipe)
            quality_factors.append(("creativity", creativity_score, 0.2))
            
            # 실현 가능성 (0-1)
            feasibility_score = self._evaluate_feasibility(recipe)
            quality_factors.append(("feasibility", feasibility_score, 0.15))
            
            # 일관성 (0-1)
            consistency_score = self._evaluate_consistency(recipe)
            quality_factors.append(("consistency", consistency_score, 0.1))
            
            # 가중 평균 계산
            total_score = sum(score * weight for _, score, weight in quality_factors)
            
            # 0-100 점수로 변환
            final_score = max(0, min(100, total_score * 100))
            
            logger.debug(f"Recipe quality evaluation: {final_score:.1f}")
            return round(final_score, 1)
            
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 75.0  # 기본 점수
    
    def _evaluate_recipe_structure(self, recipe: Dict[str, Any]) -> float:
        """레시피 구조 평가"""
        required_fields = ["name", "description", "notes", "formula"]
        present_fields = sum(1 for field in required_fields if field in recipe)
        structure_score = present_fields / len(required_fields)
        
        # 노트 구조 확인
        if "notes" in recipe:
            note_types = ["top", "middle", "base"]
            present_note_types = sum(1 for note_type in note_types 
                                   if note_type in recipe["notes"] 
                                   and recipe["notes"][note_type])
            structure_score = (structure_score + (present_note_types / len(note_types))) / 2
        
        return structure_score
    
    def _evaluate_note_balance(self, recipe: Dict[str, Any]) -> float:
        """노트 균형성 평가"""
        if "notes" not in recipe:
            return 0.5
        
        notes = recipe["notes"]
        
        # 각 층별 노트 수 확인
        top_count = len(notes.get("top", []))
        middle_count = len(notes.get("middle", []))
        base_count = len(notes.get("base", []))
        
        # 이상적인 비율: 톱(3-5), 미들(3-6), 베이스(2-4)
        ideal_ratios = {"top": (3, 5), "middle": (3, 6), "base": (2, 4)}
        
        balance_scores = []
        for note_type, (min_count, max_count) in ideal_ratios.items():
            actual_count = len(notes.get(note_type, []))
            if min_count <= actual_count <= max_count:
                balance_scores.append(1.0)
            elif actual_count == 0:
                balance_scores.append(0.0)
            else:
                # 범위를 벗어난 정도에 따라 점수 감점
                distance = min(abs(actual_count - min_count), abs(actual_count - max_count))
                balance_scores.append(max(0, 1 - distance * 0.2))
        
        return sum(balance_scores) / len(balance_scores) if balance_scores else 0.5
    
    def _evaluate_creativity(self, recipe: Dict[str, Any]) -> float:
        """창의성 평가 - 완전한 구현"""
        creativity_score = 0.0
        max_score = 100.0
        
        if "notes" not in recipe:
            return 0.0
        
        # 1. 노트 조합의 독창성 (25점)
        novelty_score = self._calculate_note_novelty(recipe["notes"])
        creativity_score += novelty_score * 25
        
        # 2. 향조 간 융합의 혁신성 (20점)
        fusion_score = self._calculate_family_fusion_score(recipe)
        creativity_score += fusion_score * 20
        
        # 3. 농도 배합의 예술성 (15점)
        if "formula" in recipe:
            artistry_score = self._calculate_formula_artistry(recipe["formula"])
            creativity_score += artistry_score * 15
        
        # 4. 테마와 컨셉트의 독창성 (15점)
        concept_score = self._calculate_concept_originality(recipe)
        creativity_score += concept_score * 15
        
        # 5. 계절성/상황별 적용의 창의성 (10점)
        seasonal_score = self._calculate_seasonal_creativity(recipe)
        creativity_score += seasonal_score * 10
        
        # 6. 스토리텔링과 감정적 연결 (10점)
        storytelling_score = self._calculate_storytelling_score(recipe)
        creativity_score += storytelling_score * 10
        
        # 7. 기술적 혁신도 (5점)
        technical_score = self._calculate_technical_innovation(recipe)
        creativity_score += technical_score * 5
        
        return min(1.0, creativity_score / max_score)
    
    def _calculate_note_novelty(self, notes: Dict[str, Any]) -> float:
        """노트 조합의 참신함 계산"""
        # 일반적이지 않은 노트 조합에 높은 점수
        unusual_combinations = [
            (["rose", "jasmine"], ["tobacco", "leather"]),  # 플로럴 + 스모키
            (["citrus", "lemon"], ["vanilla", "amber"]),    # 시트러스 + 오리엔탈
            (["marine", "ozone"], ["oud", "incense"]),      # 프레시 + 신비로운
            (["apple", "pear"], ["sandalwood", "cedar"]),   # 프루티 + 우디
            (["lavender", "mint"], ["chocolate", "coffee"]) # 허브 + 구르망
        ]
        
        all_notes = []
        for note_list in notes.values():
            if isinstance(note_list, list):
                for note in note_list:
                    note_name = note.get("name", note) if isinstance(note, dict) else note
                    all_notes.append(note_name.lower())
        
        novelty_score = 0.0
        for combo1, combo2 in unusual_combinations:
            has_combo1 = any(note for note in all_notes if any(c1 in note for c1 in combo1))
            has_combo2 = any(note for note in all_notes if any(c2 in note for c2 in combo2))
            if has_combo1 and has_combo2:
                novelty_score += 0.3
        
        # 희귀한 노트들에 대한 보너스
        rare_notes = ["oud", "ambergris", "civet", "castoreum", "frankincense", "myrrh", 
                     "saffron", "cardamom", "pink pepper", "black tea", "fig leaf"]
        rare_count = sum(1 for note in all_notes if any(rare in note for rare in rare_notes))
        novelty_score += min(0.4, rare_count * 0.1)
        
        # 노트 수의 다양성
        unique_notes = len(set(all_notes))
        if unique_notes >= 12:
            novelty_score += 0.3
        elif unique_notes >= 8:
            novelty_score += 0.2
        elif unique_notes >= 5:
            novelty_score += 0.1
        
        return min(1.0, novelty_score)
    
    def _calculate_family_fusion_score(self, recipe: Dict[str, Any]) -> float:
        """향조 융합의 혁신성"""
        if "fragrance_family" not in recipe or "notes" not in recipe:
            return 0.0
        
        family = recipe["fragrance_family"].lower()
        notes = recipe["notes"]
        
        # 예상치 못한 향조 융합 패턴 감지
        fusion_patterns = {
            "oriental-fresh": (["oriental", "amber", "vanilla"], ["marine", "ozone", "mint"]),
            "woody-gourmand": (["woody", "cedar", "sandalwood"], ["chocolate", "vanilla", "caramel"]),
            "floral-spicy": (["floral", "rose", "jasmine"], ["pepper", "ginger", "cardamom"]),
            "citrus-smoky": (["citrus", "lemon", "bergamot"], ["tobacco", "leather", "smoke"])
        }
        
        all_notes = []
        for note_list in notes.values():
            if isinstance(note_list, list):
                for note in note_list:
                    note_name = note.get("name", note) if isinstance(note, dict) else note
                    all_notes.append(note_name.lower())
        
        fusion_score = 0.0
        for pattern_name, (group1, group2) in fusion_patterns.items():
            has_group1 = any(note for note in all_notes if any(g1 in note for g1 in group1))
            has_group2 = any(note for note in all_notes if any(g2 in note for g2 in group2))
            
            if has_group1 and has_group2:
                fusion_score += 0.4
                
                # 융합이 조화롭게 이루어졌는지 확인
                if self._check_fusion_harmony(group1, group2, all_notes):
                    fusion_score += 0.2
        
        return min(1.0, fusion_score)
    
    def _check_fusion_harmony(self, group1: List[str], group2: List[str], all_notes: List[str]) -> bool:
        """융합의 조화로움 확인"""
        # 브릿지 노트가 있는지 확인 (두 그룹을 연결해주는 중간 노트)
        bridge_notes = {
            ("oriental", "fresh"): ["bergamot", "lavender", "geranium"],
            ("woody", "gourmand"): ["tonka", "benzoin", "honey"],
            ("floral", "spicy"): ["iris", "violet", "neroli"],
            ("citrus", "smoky"): ["tea", "herbs", "aromatic"]
        }
        
        for bridge_group, bridges in bridge_notes.items():
            has_bridge = any(note for note in all_notes if any(bridge in note for bridge in bridges))
            if has_bridge:
                return True
        
        return False
    
    def _calculate_formula_artistry(self, formula: Dict[str, Any]) -> float:
        """포뮬러 배합의 예술성"""
        if not formula:
            return 0.0
        
        artistry_score = 0.0
        concentrations = []
        
        # 농도값 추출
        for ingredient, concentration in formula.items():
            try:
                if isinstance(concentration, (int, float)):
                    concentrations.append(concentration)
                elif isinstance(concentration, str):
                    conc_val = float(concentration.replace('%', ''))
                    concentrations.append(conc_val)
            except:
                continue
        
        if not concentrations:
            return 0.0
        
        # 1. 황금비율 근사성 (피보나치 수열 비율)
        fibonacci_ratios = [1, 1.618, 2.618, 4.236]  # 황금비율 기반
        sorted_concs = sorted(concentrations, reverse=True)
        
        if len(sorted_concs) >= 2:
            actual_ratios = [sorted_concs[i] / sorted_concs[i+1] for i in range(len(sorted_concs)-1)]
            for actual_ratio in actual_ratios:
                for fib_ratio in fibonacci_ratios:
                    if abs(actual_ratio - fib_ratio) < 0.3:
                        artistry_score += 0.2
        
        # 2. 농도 분포의 우아함 (너무 극단적이지 않은 분포)
        std_dev = np.std(concentrations)
        mean_conc = np.mean(concentrations)
        coefficient_of_variation = std_dev / mean_conc if mean_conc > 0 else 0
        
        if 0.3 <= coefficient_of_variation <= 0.8:  # 적절한 변동성
            artistry_score += 0.3
        
        # 3. 계층적 구조 (top-middle-base의 농도 감소 패턴)
        if len(concentrations) >= 3:
            # 일반적으로 베이스 노트가 가장 높은 농도를 가짐
            descending_order = sorted(concentrations, reverse=True)
            if concentrations == descending_order:
                artistry_score += 0.2
        
        # 4. 특별한 배합 기법 (마이크로 도징)
        micro_doses = [c for c in concentrations if c < 1.0]
        if len(micro_doses) >= 2:  # 정교한 마이크로 도징
            artistry_score += 0.3
        
        return min(1.0, artistry_score)
    
    def _calculate_concept_originality(self, recipe: Dict[str, Any]) -> float:
        """컨셉트의 독창성"""
        originality_score = 0.0
        
        # 설명문의 창의성 분석
        description = recipe.get("description", "").lower()
        
        # 독창적 키워드 체크
        creative_keywords = [
            "시간여행", "기억", "꿈", "환상", "신화", "우주", "별", "달빛",
            "비밀", "금지된", "잊혀진", "숨겨진", "영원한", "순간의",
            "변화", "변신", "마법", "연금술", "철학", "예술", "음악",
            "time travel", "memory", "dream", "fantasy", "myth", "cosmic",
            "forbidden", "hidden", "eternal", "moment", "alchemy", "art"
        ]
        
        creative_count = sum(1 for keyword in creative_keywords if keyword in description)
        originality_score += min(0.4, creative_count * 0.1)
        
        # 특별한 컨셉트 필드가 있는지 확인
        concept_fields = ["concept", "inspiration", "story", "artisan_concept", "heritage_story"]
        has_concept = any(field in recipe for field in concept_fields)
        if has_concept:
            originality_score += 0.3
        
        # 이름의 창의성
        name = recipe.get("name", "").lower()
        if len(name) > 0:
            # 숫자나 특수문자가 포함된 창의적 네이밍
            if any(char.isdigit() or char in "+-*/=()[]{}@#$%^&*" for char in name):
                originality_score += 0.1
            
            # 다국어 조합 (로마자가 아닌 문자 포함)
            if any(ord(char) > 127 for char in name):
                originality_score += 0.2
        
        return min(1.0, originality_score)
    
    def _calculate_seasonal_creativity(self, recipe: Dict[str, Any]) -> float:
        """계절적 창의성"""
        seasonal_score = 0.0
        
        # 계절별 예상치 못한 노트 조합
        seasonal_surprises = {
            "winter": ["fresh", "citrus", "marine"],  # 겨울에 신선한 노트
            "summer": ["warm", "spicy", "amber"],     # 여름에 따뜻한 노트
            "spring": ["heavy", "oud", "leather"],    # 봄에 무거운 노트
            "autumn": ["light", "airy", "ozone"]      # 가을에 가벼운 노트
        }
        
        description = recipe.get("description", "").lower()
        notes_text = str(recipe.get("notes", {})).lower()
        
        for season, unexpected_notes in seasonal_surprises.items():
            if season in description:
                surprise_count = sum(1 for note in unexpected_notes if note in notes_text)
                if surprise_count > 0:
                    seasonal_score += 0.3
        
        # 시간대별 독창성 (밤/새벽/정오 등)
        time_creativity = ["midnight", "dawn", "dusk", "noon", "twilight", "aurora"]
        time_count = sum(1 for time_word in time_creativity if time_word in description)
        seasonal_score += min(0.4, time_count * 0.2)
        
        return min(1.0, seasonal_score)
    
    def _calculate_storytelling_score(self, recipe: Dict[str, Any]) -> float:
        """스토리텔링 점수"""
        story_score = 0.0
        
        description = recipe.get("description", "")
        
        if len(description) > 100:  # 충분한 길이의 설명
            story_score += 0.3
        
        # 감정적 단어들
        emotional_words = [
            "사랑", "그리움", "추억", "열정", "평화", "자유", "희망", "꿈",
            "love", "longing", "memory", "passion", "peace", "freedom", "hope", "dream"
        ]
        emotion_count = sum(1 for word in emotional_words if word in description.lower())
        story_score += min(0.4, emotion_count * 0.1)
        
        # 서사적 구조 (시작-중간-끝)
        narrative_indicators = ["처음", "그리고", "마침내", "결국", "finally", "then", "eventually"]
        narrative_count = sum(1 for indicator in narrative_indicators if indicator in description.lower())
        if narrative_count >= 2:
            story_score += 0.3
        
        return min(1.0, story_score)
    
    def _calculate_technical_innovation(self, recipe: Dict[str, Any]) -> float:
        """기술적 혁신도"""
        innovation_score = 0.0
        
        # 특별한 기술적 요소들
        technical_elements = [
            "encapsulation", "molecular", "biotechnology", "sustainable", "eco-friendly",
            "캡슐화", "분자", "생명공학", "지속가능", "친환경", "나노", "smart"
        ]
        
        full_text = str(recipe).lower()
        tech_count = sum(1 for element in technical_elements if element in full_text)
        innovation_score += min(0.6, tech_count * 0.2)
        
        # 특별한 추출 방법이나 공정 언급
        extraction_methods = [
            "cold extraction", "supercritical", "distillation", "enfluerage",
            "저온추출", "초임계", "증류", "앙플뢰라주"
        ]
        
        extraction_count = sum(1 for method in extraction_methods if method in full_text)
        if extraction_count > 0:
            innovation_score += 0.4
        
        return min(1.0, innovation_score)
    
    def _evaluate_feasibility(self, recipe: Dict[str, Any]) -> float:
        """실현 가능성 평가"""
        feasibility_score = 1.0
        
        # 노트 농도 확인
        if "formula" in recipe:
            formula = recipe["formula"]
            if isinstance(formula, dict):
                total_percentage = 0
                for note, percentage in formula.items():
                    if isinstance(percentage, (int, float)):
                        total_percentage += percentage
                    elif isinstance(percentage, str) and percentage.replace('.', '').isdigit():
                        total_percentage += float(percentage)
                
                # 총 농도가 100%를 크게 초과하면 감점
                if total_percentage > 120:
                    feasibility_score -= 0.3
                elif total_percentage < 80:
                    feasibility_score -= 0.2
        
        # 실제 존재하지 않을 가능성이 높은 노트들 체크 (간단한 예시)
        unrealistic_notes = ["unicorn", "dragon", "magic", "시간의흐름", "영혼의향기"]
        if "notes" in recipe:
            for note_list in recipe["notes"].values():
                if isinstance(note_list, list):
                    for note in note_list:
                        note_name = note if isinstance(note, str) else note.get("name", "")
                        if any(unrealistic in note_name.lower() for unrealistic in unrealistic_notes):
                            feasibility_score -= 0.1
        
        return max(0, feasibility_score)
    
    def _evaluate_consistency(self, recipe: Dict[str, Any]) -> float:
        """일관성 평가"""
        consistency_score = 1.0
        
        # 향조와 노트의 일관성 체크 (간단한 예시)
        fragrance_family = recipe.get("fragrance_family", "").lower()
        
        if fragrance_family and "notes" in recipe:
            # 향조별 예상 노트들
            family_notes = {
                "citrus": ["lemon", "orange", "bergamot", "grapefruit", "lime"],
                "floral": ["rose", "jasmine", "lily", "peony", "lavender"],
                "woody": ["sandalwood", "cedar", "oak", "pine", "birch"],
                "oriental": ["vanilla", "amber", "musk", "oud", "incense"]
            }
            
            expected_notes = []
            for family, notes in family_notes.items():
                if family in fragrance_family:
                    expected_notes.extend(notes)
            
            if expected_notes:
                all_recipe_notes = []
                for note_list in recipe["notes"].values():
                    if isinstance(note_list, list):
                        for note in note_list:
                            note_name = note if isinstance(note, str) else note.get("name", "")
                            all_recipe_notes.append(note_name.lower())
                
                # 예상 노트와의 일치율 확인
                matching_notes = sum(1 for note in all_recipe_notes 
                                   if any(expected in note for expected in expected_notes))
                if all_recipe_notes:
                    match_ratio = matching_notes / len(all_recipe_notes)
                    if match_ratio < 0.3:  # 30% 미만 일치시 감점
                        consistency_score -= 0.2
        
        return max(0, consistency_score)
    
    def _parse_recipe_output(self, raw_output: str) -> Dict[str, Any]:
        """생성된 레시피 파싱"""
        try:
            # JSON 추출 시도
            import re
            
            # JSON 블록 찾기
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # 파싱 실패시 기본 구조 반환
            return {
                "name": "Generated Fragrance",
                "description": raw_output[:200] + "..." if len(raw_output) > 200 else raw_output,
                "notes": {"top": [], "middle": [], "base": []},
                "formula": {},
                "characteristics": {"sillage": 5, "longevity": 5, "complexity": 5},
                "parse_error": "Failed to parse structured output"
            }
            
        except Exception as e:
            logger.error(f"Failed to parse recipe output: {e}")
            return {
                "name": "Generated Fragrance",
                "description": "Recipe generation completed but parsing failed",
                "notes": {"top": [], "middle": [], "base": []},
                "formula": {},
                "characteristics": {"sillage": 5, "longevity": 5, "complexity": 5},
                "parse_error": str(e)
            }
    
    async def _get_premium_ingredients(self, fragrance_family: str) -> List[str]:
        """프리미엄 재료 목록 조회"""
        # 실제로는 데이터베이스나 외부 API에서 조회
        premium_ingredients_db = {
            "oriental": [
                "Oud Assam", "Ambergris", "Saffron", "Rose Absolute", 
                "Sandalwood Mysore", "Frankincense", "Cardamom"
            ],
            "woody": [
                "Agarwood", "Sandalwood Australian", "Cedar Atlas", 
                "Palo Santo", "Hinoki", "Birch Tar"
            ],
            "floral": [
                "Rose Bulgare", "Jasmine Grandiflorum", "Tuberose Absolute",
                "Ylang-Ylang Comoro", "Iris Absolute", "Neroli Bigarade"
            ],
            "citrus": [
                "Bergamot Calabria", "Lemon Sicily", "Yuzu", 
                "Lime Distilled", "Grapefruit Pink", "Mandarin Red"
            ]
        }
        
        return premium_ingredients_db.get(fragrance_family.lower(), 
                                        ["Premium Essential Oils", "Rare Botanicals"])
    
    def _validate_generation_request(self, request_data: Dict[str, Any]):
        """생성 요청 데이터 검증"""
        required_fields = ["fragrance_family", "mood", "intensity"]
        
        for field in required_fields:
            if field not in request_data:
                raise ValueError(f"Required field missing: {field}")
        
        # 값 범위 검증
        valid_intensities = ["light", "moderate", "strong", "very_strong"]
        if request_data.get("intensity") not in valid_intensities:
            raise ValueError(f"Invalid intensity. Must be one of: {valid_intensities}")
        
        valid_genders = ["masculine", "feminine", "unisex"]
        if request_data.get("gender", "unisex") not in valid_genders:
            raise ValueError(f"Invalid gender. Must be one of: {valid_genders}")
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        import hashlib
        
        # 캐시에 영향을 주는 필드만 선별
        cache_fields = {
            "fragrance_family": request_data.get("fragrance_family"),
            "mood": request_data.get("mood"),
            "intensity": request_data.get("intensity"),
            "gender": request_data.get("gender"),
            "season": request_data.get("season"),
            "generation_type": request_data.get("generation_type"),
            "special_requirements": request_data.get("special_requirements")
        }
        
        cache_string = json.dumps(cache_fields, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """캐시된 결과 조회"""
        if cache_key in self.generation_cache:
            cached_item = self.generation_cache[cache_key]
            if time.time() - cached_item["timestamp"] < self.cache_ttl:
                return cached_item["result"]
            else:
                del self.generation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """결과 캐싱"""
        self.generation_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # 캐시 크기 제한
        if len(self.generation_cache) > 500:
            oldest_key = min(self.generation_cache.keys(),
                           key=lambda k: self.generation_cache[k]["timestamp"])
            del self.generation_cache[oldest_key]
    
    async def reload_models(self):
        """생성 모델 재로드"""
        try:
            # 생성 모델 재초기화
            if hasattr(self, 'generator'):
                del self.generator
            
            # 모델 재로드
            await self.initialize()
            
            # 캐시 클리어
            self.generation_cache.clear()
            
            logger.info("Successfully reloaded generation service models")
            
        except Exception as e:
            logger.error(f"Failed to reload generation service models: {e}")
            raise