"""
Data Cleaning Operator
수집된 향수 데이터를 정제하고 지식베이스를 업데이트하는 오퍼레이터

주요 기능:
1. 데이터 정규화 및 표준화
2. 중복 제거 및 병합
3. 누락된 정보 보완
4. 데이터 품질 검증
5. 지식베이스 업데이트
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# NLTK 데이터 다운로드 (처음 실행 시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


@dataclass
class CleanedFragranceData:
    """정제된 향수 데이터"""
    # 고유 식별자
    fragrance_id: str
    canonical_name: str

    # 기본 정보
    brand: str
    category: str
    gender: str
    release_year: Optional[int]

    # 정규화된 향 구성
    top_notes: List[str]
    heart_notes: List[str]
    base_notes: List[str]
    accords: List[str]

    # 집계된 특성
    avg_longevity: float
    avg_sillage: float
    seasons: List[str]
    occasions: List[str]

    # 통합 리뷰 데이터
    overall_rating: float
    total_reviews: int
    sentiment_score: float
    key_descriptors: List[str]

    # 메타데이터
    sources: List[str]
    last_updated: str
    data_quality_score: float


class DataCleaningOperator:
    """
    수집된 데이터를 정제하고 지식베이스를 업데이트하는 오퍼레이터
    """

    def __init__(self, knowledge_base_path: str = None):
        self.knowledge_base_path = knowledge_base_path or 'data/comprehensive_fragrance_notes_database.json'
        self.knowledge_base = self._load_knowledge_base()

        # 정규화 매핑
        self.note_mappings = self._load_note_mappings()
        self.brand_mappings = self._load_brand_mappings()

        # 통계
        self.stats = {
            'total_processed': 0,
            'cleaned': 0,
            'merged': 0,
            'discarded': 0,
            'new_entries': 0,
            'updated_entries': 0
        }

        logger.info("DataCleaningOperator initialized")

    def _load_knowledge_base(self) -> Dict:
        """기존 지식베이스 로드"""
        path = Path(self.knowledge_base_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'fragrances': {},
            'ingredients': {},
            'brands': {},
            'metadata': {
                'version': '1.0',
                'last_updated': datetime.now().isoformat()
            }
        }

    def _load_note_mappings(self) -> Dict[str, str]:
        """향 노트 이름 정규화 매핑"""
        return {
            # 일반적인 변형들
            'bergamotte': 'bergamot',
            'citrus fruits': 'citrus',
            'lemon zest': 'lemon',
            'orange blossom': 'neroli',
            'ylang ylang': 'ylang-ylang',
            'patchouly': 'patchouli',
            'vetiver root': 'vetiver',
            'sandal wood': 'sandalwood',
            'cedar wood': 'cedarwood',
            'musk': 'white musk',
            'ambergris': 'amber',
            # 한국어 -> 영어
            '장미': 'rose',
            '자스민': 'jasmine',
            '라벤더': 'lavender',
            '바닐라': 'vanilla',
            '시트러스': 'citrus',
        }

    def _load_brand_mappings(self) -> Dict[str, str]:
        """브랜드 이름 정규화 매핑"""
        return {
            'chanel': 'Chanel',
            'CHANEL': 'Chanel',
            'dior': 'Dior',
            'Christian Dior': 'Dior',
            'YSL': 'Yves Saint Laurent',
            'tom ford': 'Tom Ford',
            'jo malone': 'Jo Malone London',
        }

    def clean_raw_data(self, raw_data: List[Dict]) -> List[CleanedFragranceData]:
        """원시 데이터 정제"""
        cleaned_data = []

        for item in raw_data:
            self.stats['total_processed'] += 1

            try:
                # 데이터 유효성 검사
                if not self._validate_data(item):
                    self.stats['discarded'] += 1
                    continue

                # 정규화
                normalized = self._normalize_data(item)

                # 중복 체크 및 병합
                merged = self._merge_duplicates(normalized)

                # 품질 점수 계산
                quality_score = self._calculate_quality_score(merged)

                if quality_score >= 0.5:  # 최소 품질 기준
                    cleaned = self._create_cleaned_data(merged, quality_score)
                    cleaned_data.append(cleaned)
                    self.stats['cleaned'] += 1
                else:
                    self.stats['discarded'] += 1

            except Exception as e:
                logger.error(f"Error cleaning data: {e}")
                self.stats['discarded'] += 1

        return cleaned_data

    def _validate_data(self, data: Dict) -> bool:
        """데이터 유효성 검사"""
        # 필수 필드 체크
        required_fields = ['name', 'brand']
        for field in required_fields:
            if not data.get(field):
                return False

        # 최소 정보 체크 (노트가 하나라도 있어야 함)
        has_notes = any([
            data.get('top_notes'),
            data.get('heart_notes'),
            data.get('base_notes')
        ])

        return has_notes

    def _normalize_data(self, data: Dict) -> Dict:
        """데이터 정규화"""
        normalized = data.copy()

        # 브랜드 정규화
        if normalized.get('brand'):
            brand_lower = normalized['brand'].lower().strip()
            normalized['brand'] = self.brand_mappings.get(brand_lower, normalized['brand'])

        # 노트 정규화
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            if normalized.get(note_type):
                normalized[note_type] = self._normalize_notes(normalized[note_type])

        # 성별 정규화
        if normalized.get('gender'):
            normalized['gender'] = self._normalize_gender(normalized['gender'])

        # 카테고리 정규화
        if normalized.get('category'):
            normalized['category'] = self._normalize_category(normalized['category'])

        return normalized

    def _normalize_notes(self, notes: List[str]) -> List[str]:
        """향 노트 정규화"""
        normalized = []

        for note in notes:
            if not note:
                continue

            # 소문자 변환 및 공백 정리
            note_clean = note.lower().strip()

            # 특수문자 제거
            note_clean = re.sub(r'[^\w\s-]', '', note_clean)

            # 매핑 적용
            note_final = self.note_mappings.get(note_clean, note_clean)

            if note_final and note_final not in normalized:
                normalized.append(note_final)

        return normalized

    def _normalize_gender(self, gender: str) -> str:
        """성별 정규화"""
        gender_lower = gender.lower().strip()

        gender_map = {
            'men': 'masculine',
            'male': 'masculine',
            'homme': 'masculine',
            'women': 'feminine',
            'female': 'feminine',
            'femme': 'feminine',
            'unisex': 'unisex',
            'shared': 'unisex',
            'both': 'unisex'
        }

        return gender_map.get(gender_lower, 'unisex')

    def _normalize_category(self, category: str) -> str:
        """카테고리 정규화"""
        category_lower = category.lower().strip()

        category_map = {
            'eau de parfum': 'edp',
            'eau de toilette': 'edt',
            'eau de cologne': 'edc',
            'parfum': 'parfum',
            'extrait': 'extrait',
            'cologne': 'cologne',
            'body mist': 'mist',
            'splash': 'splash'
        }

        return category_map.get(category_lower, category_lower)

    def _merge_duplicates(self, data: Dict) -> Dict:
        """중복 데이터 병합"""
        # 같은 향수의 여러 소스 데이터를 병합
        fragrance_key = f"{data.get('brand')}:{data.get('name')}"

        # 기존 데이터가 있으면 병합
        if fragrance_key in self.knowledge_base.get('fragrances', {}):
            existing = self.knowledge_base['fragrances'][fragrance_key]
            return self._merge_fragrance_data(existing, data)

        return data

    def _merge_fragrance_data(self, existing: Dict, new: Dict) -> Dict:
        """두 향수 데이터 병합"""
        merged = existing.copy()

        # 노트 병합 (합집합)
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            existing_notes = set(existing.get(note_type, []))
            new_notes = set(new.get(note_type, []))
            merged[note_type] = list(existing_notes | new_notes)

        # 평점 평균
        if existing.get('rating') and new.get('rating'):
            existing_count = existing.get('review_count', 1)
            new_count = new.get('review_count', 1)

            weighted_avg = (
                existing['rating'] * existing_count +
                new['rating'] * new_count
            ) / (existing_count + new_count)

            merged['rating'] = round(weighted_avg, 2)
            merged['review_count'] = existing_count + new_count

        # 소스 추가
        sources = set(existing.get('sources', []))
        if new.get('source'):
            sources.add(new['source'])
        merged['sources'] = list(sources)

        self.stats['merged'] += 1

        return merged

    def _calculate_quality_score(self, data: Dict) -> float:
        """데이터 품질 점수 계산"""
        score = 0.0
        weights = {
            'name': 2.0,
            'brand': 2.0,
            'notes': 3.0,
            'reviews': 1.0,
            'metadata': 2.0
        }

        # 이름과 브랜드
        if data.get('name'):
            score += weights['name']
        if data.get('brand'):
            score += weights['brand']

        # 노트 완성도
        notes_score = 0
        if data.get('top_notes') and len(data['top_notes']) > 0:
            notes_score += 1
        if data.get('heart_notes') and len(data['heart_notes']) > 0:
            notes_score += 1
        if data.get('base_notes') and len(data['base_notes']) > 0:
            notes_score += 1
        score += (notes_score / 3) * weights['notes']

        # 리뷰 데이터
        if data.get('rating') and data.get('review_count', 0) > 0:
            score += weights['reviews']

        # 메타데이터 (카테고리, 성별, 연도 등)
        metadata_score = 0
        for field in ['category', 'gender', 'release_year']:
            if data.get(field):
                metadata_score += 1
        score += (metadata_score / 3) * weights['metadata']

        # 정규화 (0-1 범위)
        max_score = sum(weights.values())
        return score / max_score

    def _create_cleaned_data(self, data: Dict, quality_score: float) -> CleanedFragranceData:
        """정제된 데이터 객체 생성"""
        # 고유 ID 생성
        fragrance_id = self._generate_fragrance_id(data)

        # 센티먼트 분석 (리뷰가 있는 경우)
        sentiment_score = 0.0
        key_descriptors = []

        if data.get('reviews'):
            sentiment_score, key_descriptors = self._analyze_reviews(data['reviews'])

        return CleanedFragranceData(
            fragrance_id=fragrance_id,
            canonical_name=f"{data.get('brand', '')} {data.get('name', '')}".strip(),
            brand=data.get('brand', ''),
            category=data.get('category', 'edp'),
            gender=data.get('gender', 'unisex'),
            release_year=data.get('release_year'),
            top_notes=data.get('top_notes', []),
            heart_notes=data.get('heart_notes', []),
            base_notes=data.get('base_notes', []),
            accords=data.get('accords', []),
            avg_longevity=data.get('longevity', 3.0),
            avg_sillage=data.get('sillage', 3.0),
            seasons=data.get('season', ['all']),
            occasions=data.get('occasion', ['daily']),
            overall_rating=data.get('rating', 0.0),
            total_reviews=data.get('review_count', 0),
            sentiment_score=sentiment_score,
            key_descriptors=key_descriptors,
            sources=data.get('sources', [data.get('source', 'unknown')]),
            last_updated=datetime.now().isoformat(),
            data_quality_score=quality_score
        )

    def _generate_fragrance_id(self, data: Dict) -> str:
        """향수 고유 ID 생성"""
        id_string = f"{data.get('brand', '')}_{data.get('name', '')}".lower()
        id_string = re.sub(r'[^\w]', '_', id_string)
        return id_string

    def _analyze_reviews(self, reviews: List[Dict]) -> tuple[float, List[str]]:
        """리뷰 센티먼트 분석 및 키워드 추출"""
        if not reviews:
            return 0.0, []

        # 간단한 센티먼트 분석 (실제로는 더 정교한 모델 사용)
        positive_words = {'love', 'amazing', 'beautiful', 'perfect', 'excellent', 'wonderful'}
        negative_words = {'hate', 'terrible', 'awful', 'bad', 'disappointing', 'weak'}

        sentiment_scores = []
        all_text = []

        for review in reviews:
            text = review.get('text', '').lower()
            all_text.append(text)

            # 단어 빈도 기반 센티먼트
            words = set(text.split())
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)

            if pos_count + neg_count > 0:
                score = (pos_count - neg_count) / (pos_count + neg_count)
                sentiment_scores.append(score)

        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0

        # 키워드 추출 (TF-IDF)
        key_descriptors = self._extract_keywords(' '.join(all_text))

        return avg_sentiment, key_descriptors

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """TF-IDF를 사용한 키워드 추출"""
        if not text:
            return []

        try:
            # 불용어 제거
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in stop_words]

            if not words:
                return []

            # 빈도 계산
            word_freq = Counter(words)

            # 상위 키워드 반환
            top_words = [word for word, _ in word_freq.most_common(max_keywords)]

            return top_words

        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def update_knowledge_base(self, cleaned_data: List[CleanedFragranceData]):
        """지식베이스 업데이트"""
        for item in cleaned_data:
            fragrance_key = f"{item.brand}:{item.canonical_name}"

            if fragrance_key in self.knowledge_base.get('fragrances', {}):
                # 기존 항목 업데이트
                self._update_existing_entry(fragrance_key, item)
                self.stats['updated_entries'] += 1
            else:
                # 새 항목 추가
                self._add_new_entry(fragrance_key, item)
                self.stats['new_entries'] += 1

        # 메타데이터 업데이트
        self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
        self.knowledge_base['metadata']['total_fragrances'] = len(
            self.knowledge_base.get('fragrances', {})
        )

        # 파일 저장
        self._save_knowledge_base()

    def _update_existing_entry(self, key: str, data: CleanedFragranceData):
        """기존 항목 업데이트"""
        existing = self.knowledge_base['fragrances'][key]

        # 더 높은 품질 점수의 데이터로 업데이트
        if data.data_quality_score > existing.get('data_quality_score', 0):
            self.knowledge_base['fragrances'][key] = asdict(data)
        else:
            # 부분 업데이트 (누락된 정보만 추가)
            for field in ['top_notes', 'heart_notes', 'base_notes']:
                if not existing.get(field) and getattr(data, field):
                    existing[field] = getattr(data, field)

    def _add_new_entry(self, key: str, data: CleanedFragranceData):
        """새 항목 추가"""
        if 'fragrances' not in self.knowledge_base:
            self.knowledge_base['fragrances'] = {}

        self.knowledge_base['fragrances'][key] = asdict(data)

    def _save_knowledge_base(self):
        """지식베이스 파일 저장"""
        path = Path(self.knowledge_base_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)

        logger.info(f"Knowledge base saved to {path}")

    def run(self, input_file: str) -> Dict[str, Any]:
        """데이터 정제 실행"""
        # 수집된 데이터 로드
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if 'data' in raw_data:
            raw_data = raw_data['data']

        # 데이터 정제
        cleaned_data = self.clean_raw_data(raw_data)

        # 지식베이스 업데이트
        self.update_knowledge_base(cleaned_data)

        # 결과 반환
        return {
            'status': 'completed',
            'stats': self.stats,
            'cleaned_count': len(cleaned_data),
            'knowledge_base_updated': True,
            'timestamp': datetime.now().isoformat()
        }


# 실행 예시
if __name__ == "__main__":
    cleaner = DataCleaningOperator()

    # 테스트 데이터
    test_data = [
        {
            'name': 'No. 5',
            'brand': 'chanel',
            'top_notes': ['bergamotte', 'lemon zest'],
            'heart_notes': ['rose', 'ylang ylang'],
            'base_notes': ['sandal wood', 'musk'],
            'rating': 4.5,
            'review_count': 100,
            'source': 'test'
        }
    ]

    # 임시 파일에 저장
    with open('test_scraped_data.json', 'w') as f:
        json.dump({'data': test_data}, f)

    # 정제 실행
    result = cleaner.run('test_scraped_data.json')
    print(f"Cleaning completed: {result}")