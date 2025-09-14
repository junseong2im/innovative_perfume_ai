"""
향수 데이터 정제 및 검증 유틸리티
스크래핑된 데이터의 품질 향상과 일관성 확보
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib

import pandas as pd
import numpy as np
from textdistance import jaro_winkler
from langdetect import detect, DetectorFactory
from googletrans import Translator
import unicodedata

# 언어 감지 결과 일관성을 위한 시드 고정
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """데이터 품질 메트릭"""
    total_items: int
    duplicates_removed: int
    incomplete_items: int
    language_mismatches: int
    normalized_fields: int
    overall_score: float
    field_completeness: Dict[str, float]
    data_consistency: Dict[str, float]

class FragranceDataCleaner:
    """향수 데이터 전문 정제기"""

    def __init__(
        self,
        remove_duplicates: bool = True,
        validate_schema: bool = True,
        normalize_text: bool = True,
        translate_descriptions: bool = False,
        min_description_length: int = 10,
        similarity_threshold: float = 0.85
    ):
        self.remove_duplicates = remove_duplicates
        self.validate_schema = validate_schema
        self.normalize_text = normalize_text
        self.translate_descriptions = translate_descriptions
        self.min_description_length = min_description_length
        self.similarity_threshold = similarity_threshold

        # 번역기 초기화 (필요시)
        self.translator = Translator() if translate_descriptions else None

        # 향수 노트 표준화 사전
        self.note_standardization = self._init_note_standardization()

        # 브랜드명 표준화 사전
        self.brand_standardization = self._init_brand_standardization()

        # 성별 표준화
        self.gender_standardization = {
            'men': 'men', 'male': 'men', 'masculine': 'men', '남성': 'men',
            'women': 'women', 'female': 'women', 'feminine': 'women', '여성': 'women',
            'unisex': 'unisex', 'gender-neutral': 'unisex', '유니섹스': 'unisex',
            'both': 'unisex'
        }

        logger.info("FragranceDataCleaner initialized")

    def _init_note_standardization(self) -> Dict[str, str]:
        """향수 노트 표준화 사전 초기화"""
        return {
            # 시트러스
            'lemon': '레몬', '레몬': '레몬',
            'bergamot': '베르가못', '베르가못': '베르가못',
            'orange': '오렌지', '오렌지': '오렌지',
            'grapefruit': '자몽', '자몽': '자몽',
            'lime': '라임', '라임': '라임',
            'yuzu': '유자', '유자': '유자',

            # 플로럴
            'rose': '장미', '장미': '장미', 'rosa': '장미',
            'jasmine': '자스민', '자스민': '자스민',
            'lavender': '라벤더', '라벤더': '라벤더',
            'lily': '백합', '백합': '백합',
            'violet': '바이올렛', '바이올렛': '바이올렛',
            'peony': '피오니', '피오니': '피오니',

            # 우디
            'sandalwood': '샌달우드', '백단향': '샌달우드', '샌달우드': '샌달우드',
            'cedarwood': '시더우드', '삼나무': '시더우드', '시더우드': '시더우드',
            'patchouli': '패츌리', '패츌리': '패츌리',
            'vetiver': '베티버', '베티버': '베티버',

            # 오리엔탈
            'vanilla': '바닐라', '바닐라': '바닐라',
            'amber': '앰버', '앰버': '앰버',
            'musk': '머스크', '사향': '머스크', '머스크': '머스크',
            'oud': '우드', '아가우드': '우드', '우드': '우드',

            # 스파이시
            'cinnamon': '계피', '계피': '계피',
            'pepper': '후추', '후추': '후추',
            'ginger': '생강', '생강': '생강',
            'cardamom': '카다몬', '카다몬': '카다몬',
            'clove': '정향', '정향': '정향',

            # 프루티
            'apple': '사과', '사과': '사과',
            'peach': '복숭아', '복숭아': '복숭아',
            'strawberry': '딸기', '딸기': '딸기',
            'blackcurrant': '블랙커런트', '까시스': '블랙커런트', '블랙커런트': '블랙커런트'
        }

    def _init_brand_standardization(self) -> Dict[str, str]:
        """브랜드명 표준화 사전 초기화"""
        return {
            # 고급 브랜드
            'chanel': 'Chanel', 'CHANEL': 'Chanel', '샤넬': 'Chanel',
            'dior': 'Dior', 'DIOR': 'Dior', '디올': 'Dior',
            'tom ford': 'Tom Ford', 'TOM FORD': 'Tom Ford', '톰포드': 'Tom Ford',
            'creed': 'Creed', 'CREED': 'Creed', '크리드': 'Creed',
            'jo malone': 'Jo Malone', 'JO MALONE': 'Jo Malone', '조말론': 'Jo Malone',

            # 한국 브랜드
            'amore pacific': 'Amore Pacific', '아모레퍼시픽': 'Amore Pacific',
            'laneige': 'Laneige', '라네즈': 'Laneige',
            'sulwhasoo': 'Sulwhasoo', '설화수': 'Sulwhasoo',
            'innisfree': 'Innisfree', '이니스프리': 'Innisfree',

            # 대중 브랜드
            'calvin klein': 'Calvin Klein', 'CALVIN KLEIN': 'Calvin Klein', '캘빈클라인': 'Calvin Klein',
            'hugo boss': 'Hugo Boss', 'HUGO BOSS': 'Hugo Boss', '휴고보스': 'Hugo Boss',
            'versace': 'Versace', 'VERSACE': 'Versace', '베르사체': 'Versace',
            'armani': 'Giorgio Armani', 'giorgio armani': 'Giorgio Armani', '아르마니': 'Giorgio Armani'
        }

    def clean_fragrance_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """메인 데이터 정제 함수"""

        logger.info(f"Starting data cleaning for {len(raw_data)} items")

        try:
            # 1. 스키마 검증
            if self.validate_schema:
                raw_data = self._validate_schema(raw_data)

            # 2. 텍스트 정규화
            if self.normalize_text:
                raw_data = self._normalize_text_fields(raw_data)

            # 3. 중복 제거
            if self.remove_duplicates:
                raw_data = self._remove_duplicates(raw_data)

            # 4. 데이터 보강
            raw_data = self._enrich_data(raw_data)

            # 5. 최종 검증
            cleaned_data = self._final_validation(raw_data)

            logger.info(f"Data cleaning completed: {len(cleaned_data)} items remaining")
            return cleaned_data

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise

    def _validate_schema(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """스키마 검증 및 수정"""

        logger.info("Validating data schema")
        valid_items = []

        required_fields = ['name', 'brand']
        optional_fields = [
            'description', 'top_notes', 'heart_notes', 'base_notes',
            'price', 'rating', 'data_source', 'scraped_at'
        ]

        for item in data:
            try:
                # 필수 필드 검증
                if not all(field in item and item[field] for field in required_fields):
                    logger.debug(f"Skipping item due to missing required fields: {item.get('name', 'Unknown')}")
                    continue

                # 필드 타입 검증 및 수정
                validated_item = {}

                # 문자열 필드들
                for field in ['id', 'name', 'brand', 'description', 'perfumer', 'gender', 'data_source']:
                    if field in item:
                        value = item[field]
                        validated_item[field] = str(value).strip() if value is not None else ""

                # 리스트 필드들 (노트)
                for field in ['top_notes', 'heart_notes', 'base_notes', 'season', 'occasion']:
                    if field in item:
                        value = item[field]
                        if isinstance(value, list):
                            validated_item[field] = [str(note).strip() for note in value if note]
                        elif isinstance(value, str):
                            # 쉼표로 분리된 문자열을 리스트로 변환
                            validated_item[field] = [note.strip() for note in value.split(',') if note.strip()]
                        else:
                            validated_item[field] = []

                # 숫자 필드들
                for field in ['price', 'rating', 'launch_year', 'reviews_count']:
                    if field in item and item[field] is not None:
                        try:
                            if field == 'rating':
                                validated_item[field] = float(item[field])
                            elif field in ['launch_year', 'reviews_count']:
                                validated_item[field] = int(item[field])
                            else:  # price
                                validated_item[field] = float(item[field])
                        except (ValueError, TypeError):
                            validated_item[field] = None

                # 날짜 필드
                if 'scraped_at' in item:
                    validated_item['scraped_at'] = item['scraped_at']
                else:
                    validated_item['scraped_at'] = datetime.utcnow().isoformat()

                # ID 생성 (없는 경우)
                if 'id' not in validated_item or not validated_item['id']:
                    validated_item['id'] = self._generate_id(validated_item)

                valid_items.append(validated_item)

            except Exception as e:
                logger.warning(f"Failed to validate item: {e}")
                continue

        logger.info(f"Schema validation completed: {len(valid_items)} valid items")
        return valid_items

    def _normalize_text_fields(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """텍스트 필드 정규화"""

        logger.info("Normalizing text fields")

        for item in data:
            try:
                # 브랜드명 표준화
                if item.get('brand'):
                    brand_lower = item['brand'].lower().strip()
                    if brand_lower in self.brand_standardization:
                        item['brand'] = self.brand_standardization[brand_lower]
                    else:
                        item['brand'] = self._clean_brand_name(item['brand'])

                # 제품명 정규화
                if item.get('name'):
                    item['name'] = self._clean_product_name(item['name'])

                # 설명 정규화
                if item.get('description'):
                    item['description'] = self._clean_description(item['description'])

                # 노트 정규화
                for note_field in ['top_notes', 'heart_notes', 'base_notes']:
                    if item.get(note_field):
                        item[note_field] = self._normalize_notes(item[note_field])

                # 성별 정규화
                if item.get('gender'):
                    gender_lower = item['gender'].lower().strip()
                    item['gender'] = self.gender_standardization.get(gender_lower, item['gender'])

                # 퍼퓨머명 정규화
                if item.get('perfumer'):
                    item['perfumer'] = self._clean_perfumer_name(item['perfumer'])

            except Exception as e:
                logger.warning(f"Failed to normalize item {item.get('name', 'Unknown')}: {e}")
                continue

        logger.info("Text normalization completed")
        return data

    def _clean_brand_name(self, brand: str) -> str:
        """브랜드명 정리"""
        if not brand:
            return ""

        # 기본 정리
        cleaned = brand.strip()

        # 특수문자 및 불필요한 문자 제거
        cleaned = re.sub(r'[™®©]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 대소문자 정규화 (첫 글자만 대문자)
        if cleaned.isalpha():
            cleaned = cleaned.title()

        return cleaned.strip()

    def _clean_product_name(self, name: str) -> str:
        """제품명 정리"""
        if not name:
            return ""

        cleaned = name.strip()

        # HTML 태그 제거
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # 불필요한 문구 제거
        unwanted_phrases = [
            'eau de toilette', 'eau de parfum', 'edt', 'edp',
            'for men', 'for women', 'unisex',
            'ml', 'oz', '향수', 'perfume', 'fragrance'
        ]

        for phrase in unwanted_phrases:
            cleaned = re.sub(rf'\b{re.escape(phrase)}\b', '', cleaned, flags=re.IGNORECASE)

        # 여러 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 괄호 안의 용량 정보 제거
        cleaned = re.sub(r'\(\d+\s*(ml|oz)\)', '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    def _clean_description(self, description: str) -> str:
        """설명 정리"""
        if not description:
            return ""

        cleaned = description.strip()

        # HTML 태그 및 엔티티 제거
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)

        # 유니코드 정규화
        cleaned = unicodedata.normalize('NFKC', cleaned)

        # 여러 공백/줄바꿈을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 너무 짧거나 긴 설명 처리
        if len(cleaned) < self.min_description_length:
            return ""

        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "..."

        return cleaned.strip()

    def _normalize_notes(self, notes: List[str]) -> List[str]:
        """노트 리스트 정규화"""
        if not notes:
            return []

        normalized_notes = []

        for note in notes:
            if not note or not note.strip():
                continue

            note_cleaned = note.strip().lower()

            # 표준화 사전에서 찾기
            if note_cleaned in self.note_standardization:
                standardized_note = self.note_standardization[note_cleaned]
            else:
                # 기본 정리
                standardized_note = self._clean_note_name(note.strip())

            if standardized_note and standardized_note not in normalized_notes:
                normalized_notes.append(standardized_note)

        return normalized_notes

    def _clean_note_name(self, note: str) -> str:
        """개별 노트명 정리"""
        if not note:
            return ""

        cleaned = note.strip()

        # 불필요한 문구 제거
        unwanted_words = ['note', '노트', 'accord', '어코드']
        for word in unwanted_words:
            cleaned = re.sub(rf'\b{re.escape(word)}\b', '', cleaned, flags=re.IGNORECASE)

        # 여러 공백을 하나로
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _clean_perfumer_name(self, perfumer: str) -> str:
        """퍼퓨머명 정리"""
        if not perfumer:
            return ""

        cleaned = perfumer.strip()

        # 불필요한 접두사 제거
        prefixes = ['by ', 'created by ', 'nose: ', 'perfumer: ']
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):]

        # 대소문자 정규화
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())

        return cleaned.strip()

    def _remove_duplicates(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """중복 제거"""

        logger.info("Removing duplicates")

        if not data:
            return data

        unique_items = []
        seen_signatures = set()

        for item in data:
            try:
                # 고유 서명 생성
                signature = self._generate_signature(item)

                if signature not in seen_signatures:
                    unique_items.append(item)
                    seen_signatures.add(signature)
                else:
                    # 중복된 경우 더 완전한 데이터를 선택
                    existing_item = self._find_existing_item(unique_items, signature)
                    if existing_item:
                        merged_item = self._merge_duplicate_items(existing_item, item)
                        # 기존 항목을 병합된 항목으로 교체
                        for i, ui in enumerate(unique_items):
                            if self._generate_signature(ui) == signature:
                                unique_items[i] = merged_item
                                break

            except Exception as e:
                logger.warning(f"Failed to process duplicate check for item: {e}")
                continue

        duplicates_removed = len(data) - len(unique_items)
        logger.info(f"Removed {duplicates_removed} duplicate items")

        return unique_items

    def _generate_signature(self, item: Dict[str, Any]) -> str:
        """아이템의 고유 서명 생성"""
        name = self._normalize_for_comparison(item.get('name', ''))
        brand = self._normalize_for_comparison(item.get('brand', ''))

        signature_text = f"{brand}_{name}"
        return hashlib.md5(signature_text.encode()).hexdigest()

    def _normalize_for_comparison(self, text: str) -> str:
        """비교를 위한 텍스트 정규화"""
        if not text:
            return ""

        # 소문자 변환
        normalized = text.lower()

        # 특수문자 제거
        normalized = re.sub(r'[^\w\s가-힣]', '', normalized)

        # 공백 정규화
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _find_existing_item(self, items: List[Dict[str, Any]], signature: str) -> Optional[Dict[str, Any]]:
        """기존 아이템 찾기"""
        for item in items:
            if self._generate_signature(item) == signature:
                return item
        return None

    def _merge_duplicate_items(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Dict[str, Any]:
        """중복 아이템 병합"""
        merged = item1.copy()

        # 더 완전한 데이터를 선택
        for field, value in item2.items():
            if field not in merged or not merged[field]:
                merged[field] = value
            elif field in ['description'] and len(str(value)) > len(str(merged[field])):
                merged[field] = value
            elif field in ['top_notes', 'heart_notes', 'base_notes']:
                # 노트들은 합치기
                if isinstance(value, list) and isinstance(merged[field], list):
                    combined_notes = list(set(merged[field] + value))
                    merged[field] = combined_notes
            elif field in ['price', 'rating'] and value and not merged[field]:
                merged[field] = value

        return merged

    def _enrich_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 보강"""

        logger.info("Enriching data")

        for item in data:
            try:
                # 누락된 필드 추론
                if not item.get('gender'):
                    item['gender'] = self._infer_gender(item)

                if not item.get('season'):
                    item['season'] = self._infer_season(item)

                if not item.get('category'):
                    item['category'] = self._infer_category(item)

                # 설명이 너무 짧은 경우 노트 기반으로 생성
                if not item.get('description') or len(item.get('description', '')) < self.min_description_length:
                    item['description'] = self._generate_description_from_notes(item)

                # 언어 감지 및 번역 (선택적)
                if self.translate_descriptions and item.get('description'):
                    item['description'] = self._translate_if_needed(item['description'])

            except Exception as e:
                logger.warning(f"Failed to enrich item {item.get('name', 'Unknown')}: {e}")
                continue

        logger.info("Data enrichment completed")
        return data

    def _infer_gender(self, item: Dict[str, Any]) -> str:
        """성별 추론"""
        name = item.get('name', '').lower()
        description = item.get('description', '').lower()

        # 키워드 기반 추론
        men_keywords = ['men', 'homme', 'masculine', 'for him', '남성', '옴므']
        women_keywords = ['women', 'femme', 'feminine', 'for her', '여성', '펨므']

        text_to_check = f"{name} {description}"

        men_score = sum(1 for keyword in men_keywords if keyword in text_to_check)
        women_score = sum(1 for keyword in women_keywords if keyword in text_to_check)

        if men_score > women_score:
            return 'men'
        elif women_score > men_score:
            return 'women'
        else:
            return 'unisex'

    def _infer_season(self, item: Dict[str, Any]) -> List[str]:
        """계절 추론"""
        description = item.get('description', '').lower()
        notes = []
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            if item.get(note_type):
                notes.extend([note.lower() for note in item[note_type]])

        seasons = []

        # 계절별 키워드
        season_keywords = {
            '봄': ['fresh', 'light', 'floral', 'green', 'spring', '봄', '신선', '가벼운'],
            '여름': ['citrus', 'aquatic', 'fresh', 'cooling', 'summer', '여름', '시원한'],
            '가을': ['warm', 'spicy', 'woody', 'amber', 'autumn', '가을', '따뜻한'],
            '겨울': ['heavy', 'rich', 'vanilla', 'musk', 'winter', '겨울', '진한']
        }

        text_to_check = f"{description} {' '.join(notes)}"

        for season, keywords in season_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                seasons.append(season)

        return seasons if seasons else ['사계절']

    def _infer_category(self, item: Dict[str, Any]) -> str:
        """카테고리 추론"""
        notes = []
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            if item.get(note_type):
                notes.extend([note.lower() for note in item[note_type]])

        note_text = ' '.join(notes)
        description = item.get('description', '').lower()

        # 카테고리별 키워드
        if any(keyword in note_text for keyword in ['citrus', 'lemon', 'orange', 'bergamot']):
            return 'Citrus'
        elif any(keyword in note_text for keyword in ['rose', 'jasmine', 'lavender', 'lily']):
            return 'Floral'
        elif any(keyword in note_text for keyword in ['sandalwood', 'cedar', 'pine', 'woody']):
            return 'Woody'
        elif any(keyword in note_text for keyword in ['vanilla', 'amber', 'musk', 'oriental']):
            return 'Oriental'
        else:
            return 'Fresh'

    def _generate_description_from_notes(self, item: Dict[str, Any]) -> str:
        """노트 기반 설명 생성"""
        parts = []

        if item.get('top_notes'):
            parts.append(f"톱노트: {', '.join(item['top_notes'])}")

        if item.get('heart_notes'):
            parts.append(f"미들노트: {', '.join(item['heart_notes'])}")

        if item.get('base_notes'):
            parts.append(f"베이스노트: {', '.join(item['base_notes'])}")

        if not parts:
            return f"{item.get('brand', '')} {item.get('name', '')}의 향수"

        return '. '.join(parts) + '.'

    def _translate_if_needed(self, text: str) -> str:
        """필요시 번역"""
        try:
            # 언어 감지
            detected_lang = detect(text)

            # 한국어가 아닌 경우 번역
            if detected_lang != 'ko' and len(text) > 20:
                translated = self.translator.translate(text, dest='ko')
                return translated.text
        except Exception as e:
            logger.debug(f"Translation failed: {e}")

        return text

    def _final_validation(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """최종 검증"""

        logger.info("Performing final validation")
        validated_items = []

        for item in data:
            try:
                # 기본 품질 검사
                if not self._meets_quality_criteria(item):
                    continue

                # 데이터 타입 재검증
                if not self._validate_data_types(item):
                    continue

                validated_items.append(item)

            except Exception as e:
                logger.warning(f"Final validation failed for item {item.get('name', 'Unknown')}: {e}")
                continue

        logger.info(f"Final validation completed: {len(validated_items)} items passed")
        return validated_items

    def _meets_quality_criteria(self, item: Dict[str, Any]) -> bool:
        """품질 기준 확인"""
        # 필수 필드 확인
        if not item.get('name') or not item.get('brand'):
            return False

        # 최소 노트 개수
        total_notes = 0
        for note_field in ['top_notes', 'heart_notes', 'base_notes']:
            if item.get(note_field):
                total_notes += len(item[note_field])

        if total_notes < 1:  # 최소 1개 노트는 있어야 함
            return False

        # 이름 길이 확인
        if len(item.get('name', '')) < 2:
            return False

        return True

    def _validate_data_types(self, item: Dict[str, Any]) -> bool:
        """데이터 타입 검증"""
        try:
            # 문자열 필드
            string_fields = ['id', 'name', 'brand', 'description']
            for field in string_fields:
                if field in item and item[field] is not None:
                    if not isinstance(item[field], str):
                        return False

            # 리스트 필드
            list_fields = ['top_notes', 'heart_notes', 'base_notes']
            for field in list_fields:
                if field in item and item[field] is not None:
                    if not isinstance(item[field], list):
                        return False

            # 숫자 필드
            if 'price' in item and item['price'] is not None:
                if not isinstance(item['price'], (int, float)) or item['price'] < 0:
                    return False

            if 'rating' in item and item['rating'] is not None:
                if not isinstance(item['rating'], (int, float)) or not (0 <= item['rating'] <= 5):
                    return False

            return True

        except Exception:
            return False

    def calculate_quality_metrics(self, cleaned_data: List[Dict[str, Any]]) -> QualityMetrics:
        """품질 메트릭 계산"""

        logger.info("Calculating quality metrics")

        total_items = len(cleaned_data)

        if total_items == 0:
            return QualityMetrics(
                total_items=0,
                duplicates_removed=0,
                incomplete_items=0,
                language_mismatches=0,
                normalized_fields=0,
                overall_score=0.0,
                field_completeness={},
                data_consistency={}
            )

        # 필드 완성도 계산
        field_completeness = {}
        important_fields = ['name', 'brand', 'description', 'top_notes', 'heart_notes', 'base_notes', 'price', 'rating']

        for field in important_fields:
            complete_count = sum(1 for item in cleaned_data
                               if item.get(field) and
                               (not isinstance(item[field], list) or len(item[field]) > 0))
            field_completeness[field] = complete_count / total_items

        # 데이터 일관성 계산
        data_consistency = {
            'brand_standardization': self._calculate_brand_consistency(cleaned_data),
            'note_standardization': self._calculate_note_consistency(cleaned_data),
            'gender_classification': self._calculate_gender_consistency(cleaned_data)
        }

        # 전체 품질 점수 계산
        completeness_score = np.mean(list(field_completeness.values()))
        consistency_score = np.mean(list(data_consistency.values()))
        overall_score = (completeness_score * 0.6 + consistency_score * 0.4)

        return QualityMetrics(
            total_items=total_items,
            duplicates_removed=0,  # 실제 구현에서는 추적
            incomplete_items=sum(1 for item in cleaned_data if not self._is_complete(item)),
            language_mismatches=0,  # 실제 구현에서는 추적
            normalized_fields=total_items * len(important_fields),
            overall_score=overall_score,
            field_completeness=field_completeness,
            data_consistency=data_consistency
        )

    def _calculate_brand_consistency(self, data: List[Dict[str, Any]]) -> float:
        """브랜드 일관성 계산"""
        brands = [item.get('brand', '') for item in data if item.get('brand')]
        if not brands:
            return 0.0

        # 표준화된 브랜드명 비율
        standardized_count = sum(1 for brand in brands
                               if brand in self.brand_standardization.values())
        return standardized_count / len(brands)

    def _calculate_note_consistency(self, data: List[Dict[str, Any]]) -> float:
        """노트 일관성 계산"""
        all_notes = []
        for item in data:
            for note_field in ['top_notes', 'heart_notes', 'base_notes']:
                if item.get(note_field):
                    all_notes.extend(item[note_field])

        if not all_notes:
            return 0.0

        # 표준화된 노트 비율
        standardized_count = sum(1 for note in all_notes
                               if note in self.note_standardization.values())
        return standardized_count / len(all_notes)

    def _calculate_gender_consistency(self, data: List[Dict[str, Any]]) -> float:
        """성별 분류 일관성 계산"""
        gender_items = [item for item in data if item.get('gender')]
        if not gender_items:
            return 0.0

        valid_genders = {'men', 'women', 'unisex'}
        valid_count = sum(1 for item in gender_items
                         if item.get('gender') in valid_genders)
        return valid_count / len(gender_items)

    def _is_complete(self, item: Dict[str, Any]) -> bool:
        """아이템 완성도 확인"""
        required_fields = ['name', 'brand']
        important_fields = ['description', 'top_notes', 'heart_notes', 'base_notes']

        # 필수 필드 확인
        if not all(item.get(field) for field in required_fields):
            return False

        # 중요 필드 중 절반 이상 있어야 완성된 것으로 간주
        complete_important = sum(1 for field in important_fields
                               if item.get(field) and
                               (not isinstance(item[field], list) or len(item[field]) > 0))

        return complete_important >= len(important_fields) // 2

    def _generate_id(self, item: Dict[str, Any]) -> str:
        """아이템 ID 생성"""
        name = item.get('name', 'unknown')
        brand = item.get('brand', 'unknown')
        source = item.get('data_source', 'unknown')

        id_text = f"{source}_{brand}_{name}".lower()
        id_text = re.sub(r'[^\w]', '_', id_text)

        return hashlib.md5(id_text.encode()).hexdigest()[:12]