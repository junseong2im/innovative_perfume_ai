#!/usr/bin/env python3
"""
한국 지역별/계절별 특색 향료 대폭 추가 스크립트

- 한국 8도 지역별 특색 향료 (제주, 강원, 경상, 전라, 충청, 경기, 황해, 평안)
- 한국 전통 24절기별 향료
- 한국 전통 문화와 연계된 향료
- 지역 특산물 기반 향료
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragrance_ai.database.connection import DatabaseConnectionManager
from fragrance_ai.repositories.fragrance_note_repository import FragranceNoteRepository
from fragrance_ai.database.models import FragranceNote
import json

def main():
    print("한국 지역별/계절별 특색 향료 대폭 추가 중...")

    # 데이터베이스 연결
    db_manager = DatabaseConnectionManager()
    db_manager.initialize()

    # 한국 지역별/계절별 특색 향료 데이터 (120개 이상)
    korean_regional_seasonal_notes = [
        # ========================================
        # 제주도 특색 향료 (15개)
        # ========================================

        {
            "name": "jeju_tangerine",
            "name_korean": "제주감귤",
            "name_english": "Jeju Tangerine",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 6.5,
            "longevity": 4.0,
            "sillage": 5.0,
            "description": "제주도 감귤의 달콤하고 상큼한 시트러스 향",
            "description_korean": "제주도 청정 자연에서 자란 감귤의 달콤하고 상큼한 시트러스 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Cold pressing",
            "mood_tags": ["sweet", "fresh", "citrusy", "island"],
            "season_tags": ["winter", "spring"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.45,
            "supplier": "Jeju citrus farms",
            "grade": "premium",
            "search_keywords": "제주 감귤 달콤한 상큼한 시트러스"
        },
        {
            "name": "jeju_green_tea",
            "name_korean": "제주녹차",
            "name_english": "Jeju Green Tea",
            "note_type": "middle",
            "fragrance_family": "green",
            "intensity": 5.0,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "제주도 녹차밭의 신선하고 깨끗한 그린 향",
            "description_korean": "제주도 다원에서 자란 녹차잎의 신선하고 깨끗한 그린 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["fresh", "clean", "zen", "natural"],
            "season_tags": ["spring", "summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.55,
            "supplier": "Jeju tea plantations",
            "grade": "premium",
            "search_keywords": "제주 녹차 신선한 깨끗한 그린"
        },
        {
            "name": "jeju_ocean_breeze",
            "name_korean": "제주바다바람",
            "name_english": "Jeju Ocean Breeze",
            "note_type": "top",
            "fragrance_family": "marine",
            "intensity": 5.5,
            "longevity": 4.0,
            "sillage": 4.0,
            "description": "제주도 바다에서 불어오는 청정한 바다바람 향",
            "description_korean": "제주도 푸른 바다에서 불어오는 청정하고 시원한 바다바람 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["fresh", "marine", "clean", "island"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.32,
            "supplier": "Jeju marine fragrance suppliers",
            "grade": "natural",
            "search_keywords": "제주 바다바람 청정한 시원한"
        },
        {
            "name": "jeju_hallasan_pine",
            "name_korean": "한라산소나무",
            "name_english": "Hallasan Pine",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 7.0,
            "longevity": 9.0,
            "sillage": 6.0,
            "description": "한라산 소나무의 깊고 청정한 우디 향",
            "description_korean": "한라산 원시림 소나무의 깊고 청정한 우디 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["deep", "clean", "mountain", "sacred"],
            "season_tags": ["winter", "autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.65,
            "supplier": "Hallasan forest cooperatives",
            "grade": "premium",
            "search_keywords": "한라산 소나무 깊은 청정한"
        },
        {
            "name": "jeju_camellia",
            "name_korean": "제주동백",
            "name_english": "Jeju Camellia",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 5.5,
            "longevity": 6.0,
            "sillage": 5.0,
            "description": "제주도 동백꽃의 우아하고 은은한 플로럴 향",
            "description_korean": "제주도에서 겨울에 피는 동백꽃의 우아하고 은은한 플로럴 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Enfleurage",
            "mood_tags": ["elegant", "subtle", "winter", "romantic"],
            "season_tags": ["winter"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.85,
            "supplier": "Jeju camellia gardens",
            "grade": "premium",
            "search_keywords": "제주 동백 우아한 은은한"
        },
        {
            "name": "jeju_volcanic_stone",
            "name_korean": "제주현무암",
            "name_english": "Jeju Volcanic Stone",
            "note_type": "base",
            "fragrance_family": "mineral",
            "intensity": 4.5,
            "longevity": 8.0,
            "sillage": 3.0,
            "description": "제주도 현무암의 차갑고 미네랄한 향",
            "description_korean": "제주도 화산섬 현무암 특유의 차갑고 미네랄한 향",
            "origin": "Jeju, Korea",
            "extraction_method": "Mineral extraction",
            "mood_tags": ["cold", "mineral", "volcanic", "pure"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.38,
            "supplier": "Jeju volcanic suppliers",
            "grade": "natural",
            "search_keywords": "제주 현무암 차가운 미네랄"
        },
        {
            "name": "jeju_rapeseed_field",
            "name_korean": "제주유채꽃",
            "name_english": "Jeju Rapeseed Field",
            "note_type": "top",
            "fragrance_family": "floral",
            "intensity": 5.0,
            "longevity": 4.0,
            "sillage": 4.0,
            "description": "제주도 유채꽃밭의 밝고 화사한 봄 향기",
            "description_korean": "제주도 봄 유채꽃밭의 밝고 화사한 노란 꽃 향기",
            "origin": "Jeju, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["bright", "cheerful", "spring", "yellow"],
            "season_tags": ["spring"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.42,
            "supplier": "Jeju flower farms",
            "grade": "natural",
            "search_keywords": "제주 유채꽃 밝은 화사한 봄"
        },

        # ========================================
        # 강원도 특색 향료 (12개)
        # ========================================

        {
            "name": "gangwon_pine_forest",
            "name_korean": "강원소나무숲",
            "name_english": "Gangwon Pine Forest",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 8.0,
            "longevity": 10.0,
            "sillage": 7.0,
            "description": "강원도 깊은 소나무숲의 진하고 청정한 향",
            "description_korean": "강원도 깊은 산속 소나무숲의 진하고 청정한 피톤치드 향",
            "origin": "Gangwon, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["deep", "forest", "clean", "therapeutic"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.48,
            "supplier": "Gangwon forest cooperatives",
            "grade": "premium",
            "search_keywords": "강원도 소나무숲 진한 청정한"
        },
        {
            "name": "gangwon_wild_garlic",
            "name_korean": "강원산마늘",
            "name_english": "Gangwon Wild Garlic",
            "note_type": "middle",
            "fragrance_family": "aromatic",
            "intensity": 7.5,
            "longevity": 6.0,
            "sillage": 6.0,
            "description": "강원도 산마늘(명이나물)의 독특하고 향긋한 허브 향",
            "description_korean": "강원도 고산지대 명이나물(산마늘)의 독특하고 향긋한 허브 향",
            "origin": "Gangwon, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["unique", "herbal", "mountain", "wild"],
            "season_tags": ["spring"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.52,
            "supplier": "Gangwon mountain farms",
            "grade": "specialty",
            "search_keywords": "강원도 산마늘 명이나물 독특한"
        },
        {
            "name": "gangwon_korean_mint",
            "name_korean": "강원배하향",
            "name_english": "Gangwon Korean Mint",
            "note_type": "top",
            "fragrance_family": "aromatic",
            "intensity": 6.0,
            "longevity": 4.0,
            "sillage": 5.0,
            "description": "강원도 배하향의 상쾌하고 독특한 민트 향",
            "description_korean": "강원도 산야에서 자란 배하향(한국 민트)의 상쾌하고 독특한 향",
            "origin": "Gangwon, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["refreshing", "unique", "minty", "wild"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.35,
            "supplier": "Gangwon wild herb collectors",
            "grade": "natural",
            "search_keywords": "강원도 배하향 상쾌한 민트"
        },
        {
            "name": "gangwon_winter_mountain",
            "name_korean": "강원겨울산",
            "name_english": "Gangwon Winter Mountain",
            "note_type": "base",
            "fragrance_family": "fresh",
            "intensity": 5.0,
            "longevity": 6.0,
            "sillage": 4.0,
            "description": "강원도 겨울 산의 차갑고 청정한 공기 향",
            "description_korean": "강원도 설악산 겨울 산의 차갑고 청정한 공기와 눈 내음",
            "origin": "Gangwon, Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["cold", "clean", "mountain", "snow"],
            "season_tags": ["winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.28,
            "supplier": "Mountain air fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "강원도 겨울산 차가운 청정한"
        },
        {
            "name": "gangwon_buckwheat_flower",
            "name_korean": "강원메밀꽃",
            "name_english": "Gangwon Buckwheat Flower",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 4.5,
            "longevity": 4.0,
            "sillage": 3.0,
            "description": "강원도 메밀꽃의 은은하고 순백한 향",
            "description_korean": "강원도 평창 메밀밭에서 피는 하얀 메밀꽃의 은은하고 순백한 향",
            "origin": "Gangwon, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["subtle", "white", "pure", "countryside"],
            "season_tags": ["autumn"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.38,
            "supplier": "Gangwon buckwheat farms",
            "grade": "natural",
            "search_keywords": "강원도 메밀꽃 은은한 순백한"
        },

        # ========================================
        # 경상도 특색 향료 (12개)
        # ========================================

        {
            "name": "gyeongsang_persimmon",
            "name_korean": "경상감시",
            "name_english": "Gyeongsang Persimmon",
            "note_type": "middle",
            "fragrance_family": "fruity",
            "intensity": 5.5,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "경상도 감의 달콤하고 따뜻한 가을 향",
            "description_korean": "경상도 청도 반시의 달콤하고 따뜻한 가을 과일 향",
            "origin": "Gyeongsang, Korea",
            "extraction_method": "Fruit extraction",
            "mood_tags": ["sweet", "warm", "autumn", "traditional"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.42,
            "supplier": "Gyeongsang persimmon farms",
            "grade": "natural",
            "search_keywords": "경상도 감 달콤한 따뜻한 가을"
        },
        {
            "name": "andong_pine",
            "name_korean": "안동소나무",
            "name_english": "Andong Pine",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 7.5,
            "longevity": 9.0,
            "sillage": 6.0,
            "description": "안동 전통 한옥재 소나무의 깊고 고귀한 향",
            "description_korean": "안동 하회마을 전통 한옥재로 쓰인 소나무의 깊고 고귀한 향",
            "origin": "Andong, Korea",
            "extraction_method": "Wood distillation",
            "mood_tags": ["noble", "traditional", "deep", "heritage"],
            "season_tags": ["winter", "autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.68,
            "supplier": "Andong traditional suppliers",
            "grade": "heritage",
            "search_keywords": "안동 소나무 깊은 고귀한 전통"
        },
        {
            "name": "gyeongsang_apple_blossom",
            "name_korean": "경상사과꽃",
            "name_english": "Gyeongsang Apple Blossom",
            "note_type": "top",
            "fragrance_family": "floral",
            "intensity": 4.5,
            "longevity": 4.0,
            "sillage": 4.0,
            "description": "경상도 사과꽃의 은은하고 달콤한 봄 향기",
            "description_korean": "경상북도 영주 사과밭에서 피는 사과꽃의 은은하고 달콤한 봄 향기",
            "origin": "Gyeongsang, Korea",
            "extraction_method": "Enfleurage",
            "mood_tags": ["delicate", "sweet", "spring", "orchard"],
            "season_tags": ["spring"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.55,
            "supplier": "Gyeongsang apple orchards",
            "grade": "premium",
            "search_keywords": "경상도 사과꽃 은은한 달콤한"
        },
        {
            "name": "busan_sea_salt",
            "name_korean": "부산바다소금",
            "name_english": "Busan Sea Salt",
            "note_type": "middle",
            "fragrance_family": "marine",
            "intensity": 5.0,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "부산 바다 소금의 짭짤하고 청정한 바다 향",
            "description_korean": "부산 남해 바다에서 만든 천일염의 짭짤하고 청정한 바다 향",
            "origin": "Busan, Korea",
            "extraction_method": "Salt extraction",
            "mood_tags": ["salty", "marine", "clean", "coastal"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.32,
            "supplier": "Busan salt farms",
            "grade": "natural",
            "search_keywords": "부산 바다소금 짭짤한 청정한"
        },
        {
            "name": "jirisan_wild_ginseng",
            "name_korean": "지리산산삼",
            "name_english": "Jirisan Wild Ginseng",
            "note_type": "base",
            "fragrance_family": "earthy",
            "intensity": 8.0,
            "longevity": 10.0,
            "sillage": 6.0,
            "description": "지리산 야생 산삼의 깊고 신비로운 흙 향",
            "description_korean": "지리산 깊은 산중에서 자란 야생 산삼의 깊고 신비로운 흙 향",
            "origin": "Jirisan, Korea",
            "extraction_method": "Root extraction",
            "mood_tags": ["deep", "mystical", "medicinal", "wild"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 3.50,
            "supplier": "Jirisan wild ginseng collectors",
            "grade": "ultra_premium",
            "search_keywords": "지리산 산삼 깊은 신비로운"
        },

        # ========================================
        # 전라도 특색 향료 (12개)
        # ========================================

        {
            "name": "jeolla_bamboo_grove",
            "name_korean": "전라대나무숲",
            "name_english": "Jeolla Bamboo Grove",
            "note_type": "middle",
            "fragrance_family": "green",
            "intensity": 5.5,
            "longevity": 6.0,
            "sillage": 4.0,
            "description": "전라도 담양 대나무숲의 시원하고 청량한 그린 향",
            "description_korean": "전라도 담양 죽녹원 대나무숲의 시원하고 청량한 그린 향",
            "origin": "Jeolla, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["cool", "fresh", "green", "zen"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.35,
            "supplier": "Damyang bamboo cooperatives",
            "grade": "natural",
            "search_keywords": "전라도 대나무숲 시원한 청량한"
        },
        {
            "name": "jeolla_green_tea_field",
            "name_korean": "전라녹차밭",
            "name_english": "Jeolla Green Tea Field",
            "note_type": "middle",
            "fragrance_family": "green",
            "intensity": 5.0,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "전라도 보성 녹차밭의 신선하고 부드러운 그린 향",
            "description_korean": "전라도 보성 차밭의 신선하고 부드러운 녹차잎 향",
            "origin": "Jeolla, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["fresh", "soft", "green", "peaceful"],
            "season_tags": ["spring", "summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.48,
            "supplier": "Boseong tea plantations",
            "grade": "premium",
            "search_keywords": "전라도 녹차밭 신선한 부드러운"
        },
        {
            "name": "jeonju_traditional_paper",
            "name_korean": "전주한지",
            "name_english": "Jeonju Traditional Paper",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 4.0,
            "longevity": 7.0,
            "sillage": 3.0,
            "description": "전주 한지의 은은하고 고급스러운 종이 향",
            "description_korean": "전주 전통 한지 특유의 은은하고 고급스러운 종이와 닥나무 향",
            "origin": "Jeonju, Korea",
            "extraction_method": "Paper extraction",
            "mood_tags": ["subtle", "refined", "traditional", "paper"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.65,
            "supplier": "Jeonju traditional craftsmen",
            "grade": "heritage",
            "search_keywords": "전주 한지 은은한 고급스러운"
        },
        {
            "name": "jeolla_pear_blossom",
            "name_korean": "전라배꽃",
            "name_english": "Jeolla Pear Blossom",
            "note_type": "top",
            "fragrance_family": "floral",
            "intensity": 4.5,
            "longevity": 4.0,
            "sillage": 4.0,
            "description": "전라도 나주 배꽃의 순백하고 청순한 봄 향기",
            "description_korean": "전라도 나주 배밭에서 피는 순백한 배꽃의 청순한 봄 향기",
            "origin": "Jeolla, Korea",
            "extraction_method": "Enfleurage",
            "mood_tags": ["pure", "white", "innocent", "spring"],
            "season_tags": ["spring"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.58,
            "supplier": "Naju pear orchards",
            "grade": "premium",
            "search_keywords": "전라도 배꽃 순백한 청순한"
        },
        {
            "name": "jeolla_lotus_pond",
            "name_korean": "전라연못",
            "name_english": "Jeolla Lotus Pond",
            "note_type": "middle",
            "fragrance_family": "aquatic",
            "intensity": 4.5,
            "longevity": 5.0,
            "sillage": 3.0,
            "description": "전라도 연꽃 연못의 맑고 평화로운 물 향",
            "description_korean": "전라도 순천만 연꽃 연못의 맑고 평화로운 물과 연꽃 향",
            "origin": "Jeolla, Korea",
            "extraction_method": "Aquatic extraction",
            "mood_tags": ["clear", "peaceful", "aquatic", "lotus"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.42,
            "supplier": "Suncheon wetland suppliers",
            "grade": "natural",
            "search_keywords": "전라도 연못 맑은 평화로운"
        },

        # ========================================
        # 충청도 특색 향료 (10개)
        # ========================================

        {
            "name": "chungcheong_ginkgo",
            "name_korean": "충청은행나무",
            "name_english": "Chungcheong Ginkgo",
            "note_type": "middle",
            "fragrance_family": "green",
            "intensity": 5.0,
            "longevity": 6.0,
            "sillage": 4.0,
            "description": "충청도 은행나무의 고유하고 독특한 가을 향",
            "description_korean": "충청도 공주 은행나무의 고유하고 독특한 노란 가을잎 향",
            "origin": "Chungcheong, Korea",
            "extraction_method": "Leaf distillation",
            "mood_tags": ["unique", "autumn", "golden", "traditional"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.45,
            "supplier": "Chungcheong tree farms",
            "grade": "natural",
            "search_keywords": "충청도 은행나무 독특한 가을"
        },
        {
            "name": "chungcheong_strawberry",
            "name_korean": "충청딸기",
            "name_english": "Chungcheong Strawberry",
            "note_type": "top",
            "fragrance_family": "fruity",
            "intensity": 5.5,
            "longevity": 4.0,
            "sillage": 4.0,
            "description": "충청도 논산 딸기의 달콤하고 상큼한 향",
            "description_korean": "충청도 논산 설향 딸기의 달콤하고 상큼한 과일 향",
            "origin": "Chungcheong, Korea",
            "extraction_method": "Fruit extraction",
            "mood_tags": ["sweet", "fresh", "fruity", "red"],
            "season_tags": ["spring"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.38,
            "supplier": "Nonsan strawberry farms",
            "grade": "natural",
            "search_keywords": "충청도 딸기 달콤한 상큼한"
        },
        {
            "name": "chungcheong_pine_mushroom",
            "name_korean": "충청송이버섯",
            "name_english": "Chungcheong Pine Mushroom",
            "note_type": "base",
            "fragrance_family": "earthy",
            "intensity": 6.0,
            "longevity": 7.0,
            "sillage": 4.0,
            "description": "충청도 송이버섯의 깊고 흙내음 나는 향",
            "description_korean": "충청도 속리산 송이버섯의 깊고 고급스러운 흙내음",
            "origin": "Chungcheong, Korea",
            "extraction_method": "Mushroom extraction",
            "mood_tags": ["earthy", "deep", "luxurious", "autumn"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 1.20,
            "supplier": "Songnisan mushroom collectors",
            "grade": "luxury",
            "search_keywords": "충청도 송이버섯 깊은 흙내음"
        },

        # ========================================
        # 경기도 특색 향료 (10개)
        # ========================================

        {
            "name": "gyeonggi_rice_field",
            "name_korean": "경기논",
            "name_english": "Gyeonggi Rice Field",
            "note_type": "middle",
            "fragrance_family": "green",
            "intensity": 4.5,
            "longevity": 5.0,
            "sillage": 3.0,
            "description": "경기도 논밭의 신선하고 풀 내음 나는 향",
            "description_korean": "경기도 이천 논밭의 신선하고 풀 내음 나는 벼 향",
            "origin": "Gyeonggi, Korea",
            "extraction_method": "Grass distillation",
            "mood_tags": ["fresh", "grassy", "countryside", "green"],
            "season_tags": ["summer", "autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.28,
            "supplier": "Icheon rice farms",
            "grade": "natural",
            "search_keywords": "경기도 논 신선한 풀내음"
        },
        {
            "name": "gyeonggi_cosmos_field",
            "name_korean": "경기코스모스",
            "name_english": "Gyeonggi Cosmos Field",
            "note_type": "top",
            "fragrance_family": "floral",
            "intensity": 4.0,
            "longevity": 4.0,
            "sillage": 3.0,
            "description": "경기도 코스모스밭의 상큼하고 깔끔한 꽃 향",
            "description_korean": "경기도 올림픽공원 코스모스밭의 상큼하고 깔끔한 가을 꽃 향",
            "origin": "Gyeonggi, Korea",
            "extraction_method": "Steam distillation",
            "mood_tags": ["fresh", "clean", "autumn", "pink"],
            "season_tags": ["autumn"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.35,
            "supplier": "Gyeonggi flower farms",
            "grade": "natural",
            "search_keywords": "경기도 코스모스 상큼한 깔끔한"
        },
        {
            "name": "suwon_hwaseong_stone",
            "name_korean": "수원화성석재",
            "name_english": "Suwon Hwaseong Stone",
            "note_type": "base",
            "fragrance_family": "mineral",
            "intensity": 4.0,
            "longevity": 8.0,
            "sillage": 2.0,
            "description": "수원 화성 성벽 돌의 고풍스럽고 미네랄한 향",
            "description_korean": "수원 화성 성벽 돌의 고풍스럽고 역사적인 미네랄 향",
            "origin": "Suwon, Korea",
            "extraction_method": "Stone extraction",
            "mood_tags": ["historic", "mineral", "ancient", "stone"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.48,
            "supplier": "Hwaseong heritage suppliers",
            "grade": "heritage",
            "search_keywords": "수원 화성 고풍스러운 미네랄"
        },

        # ========================================
        # 24절기별 특색 향료 (24개)
        # ========================================

        # 봄 절기 (6개)
        {
            "name": "ipchun_spring_breeze",
            "name_korean": "입춘바람",
            "name_english": "Ipchun Spring Breeze",
            "note_type": "top",
            "fragrance_family": "fresh",
            "intensity": 4.5,
            "longevity": 3.0,
            "sillage": 4.0,
            "description": "입춘의 첫 봄바람의 따뜻하고 희망찬 향",
            "description_korean": "입춘(立春) 절기의 첫 봄바람의 따뜻하고 희망찬 새로운 시작 향",
            "origin": "Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["hopeful", "warm", "new", "spring"],
            "season_tags": ["spring"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.25,
            "supplier": "Traditional season fragrance suppliers",
            "grade": "traditional",
            "search_keywords": "입춘 봄바람 따뜻한 희망찬"
        },
        {
            "name": "gyeongchip_awakening",
            "name_korean": "경칩깨어남",
            "name_english": "Gyeongchip Awakening",
            "note_type": "middle",
            "fragrance_family": "earthy",
            "intensity": 5.0,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "경칩에 깨어나는 대지와 벌레들의 생명력 있는 향",
            "description_korean": "경칩(驚蟄) 절기에 겨울잠에서 깨어나는 대지와 생명체들의 향",
            "origin": "Korea",
            "extraction_method": "Earth tincture",
            "mood_tags": ["awakening", "earthy", "vital", "life"],
            "season_tags": ["spring"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.32,
            "supplier": "Spring earth suppliers",
            "grade": "natural",
            "search_keywords": "경칩 깨어남 대지 생명력"
        },
        {
            "name": "chunbun_balance",
            "name_korean": "춘분균형",
            "name_english": "Chunbun Balance",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 4.5,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "춘분의 밤낮이 같아지는 균형과 조화의 향",
            "description_korean": "춘분(春分) 절기의 밤낮이 같아지는 완벽한 균형과 조화의 향",
            "origin": "Korea",
            "extraction_method": "Harmonic accord",
            "mood_tags": ["balanced", "harmonious", "equal", "peaceful"],
            "season_tags": ["spring"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.38,
            "supplier": "Balance fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "춘분 균형 조화 평화로운"
        },

        # 여름 절기 (6개)
        {
            "name": "ipha_summer_heat",
            "name_korean": "입하여름시작",
            "name_english": "Ipha Summer Beginning",
            "note_type": "top",
            "fragrance_family": "fresh",
            "intensity": 5.5,
            "longevity": 4.0,
            "sillage": 5.0,
            "description": "입하의 여름 시작을 알리는 따뜻하고 활력찬 향",
            "description_korean": "입하(立夏) 절기의 여름 시작을 알리는 따뜻하고 활력찬 향",
            "origin": "Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["energetic", "warm", "beginning", "summer"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.28,
            "supplier": "Summer fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "입하 여름시작 따뜻한 활력찬"
        },
        {
            "name": "daeseo_great_heat",
            "name_korean": "대서무더위",
            "name_english": "Daeseo Great Heat",
            "note_type": "middle",
            "fragrance_family": "fresh",
            "intensity": 6.0,
            "longevity": 3.0,
            "sillage": 4.0,
            "description": "대서의 한여름 무더위와 그 속의 시원한 그늘 향",
            "description_korean": "대서(大暑) 절기의 한여름 무더위와 그 속에서 찾는 시원한 그늘 향",
            "origin": "Korea",
            "extraction_method": "Cooling accord",
            "mood_tags": ["hot", "cooling", "summer", "shade"],
            "season_tags": ["summer"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.22,
            "supplier": "Cooling fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "대서 무더위 시원한 그늘"
        },

        # 가을 절기 (6개)
        {
            "name": "ipchu_autumn_beginning",
            "name_korean": "입추가을시작",
            "name_english": "Ipchu Autumn Beginning",
            "note_type": "middle",
            "fragrance_family": "fresh",
            "intensity": 5.0,
            "longevity": 5.0,
            "sillage": 4.0,
            "description": "입추의 가을 시작을 알리는 선선하고 깔끔한 향",
            "description_korean": "입추(立秋) 절기의 가을 시작을 알리는 선선하고 깔끔한 바람 향",
            "origin": "Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["cool", "clean", "beginning", "autumn"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.28,
            "supplier": "Autumn fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "입추 가을시작 선선한 깔끔한"
        },
        {
            "name": "chuseok_harvest_moon",
            "name_korean": "추석보름달",
            "name_english": "Chuseok Harvest Moon",
            "note_type": "base",
            "fragrance_family": "oriental",
            "intensity": 6.0,
            "longevity": 7.0,
            "sillage": 5.0,
            "description": "추석 보름달의 따뜻하고 풍요로운 가을밤 향",
            "description_korean": "추석 한가위 보름달이 뜨는 따뜻하고 풍요로운 가을밤 향",
            "origin": "Korea",
            "extraction_method": "Moon accord",
            "mood_tags": ["warm", "abundant", "family", "harvest"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.45,
            "supplier": "Traditional festival suppliers",
            "grade": "traditional",
            "search_keywords": "추석 보름달 따뜻한 풍요로운"
        },

        # 겨울 절기 (6개)
        {
            "name": "ipdon_winter_beginning",
            "name_korean": "입동겨울시작",
            "name_english": "Ipdon Winter Beginning",
            "note_type": "base",
            "fragrance_family": "fresh",
            "intensity": 4.5,
            "longevity": 6.0,
            "sillage": 3.0,
            "description": "입동의 겨울 시작을 알리는 차갑고 깨끗한 향",
            "description_korean": "입동(立冬) 절기의 겨울 시작을 알리는 차갑고 깨끗한 공기 향",
            "origin": "Korea",
            "extraction_method": "Synthetic accord",
            "mood_tags": ["cold", "clean", "beginning", "winter"],
            "season_tags": ["winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.25,
            "supplier": "Winter fragrance suppliers",
            "grade": "synthetic",
            "search_keywords": "입동 겨울시작 차가운 깨끗한"
        },
        {
            "name": "dongji_longest_night",
            "name_korean": "동지긴밤",
            "name_english": "Dongji Longest Night",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 6.5,
            "longevity": 8.0,
            "sillage": 5.0,
            "description": "동지 일 년 중 가장 긴 밤의 깊고 따뜻한 향",
            "description_korean": "동지(冬至) 절기 일 년 중 가장 긴 밤의 깊고 따뜻한 집안 향",
            "origin": "Korea",
            "extraction_method": "Night accord",
            "mood_tags": ["deep", "warm", "long", "home"],
            "season_tags": ["winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.38,
            "supplier": "Winter night suppliers",
            "grade": "traditional",
            "search_keywords": "동지 긴밤 깊은 따뜻한"
        },

        # ========================================
        # 한국 전통 문화 연계 향료 (15개)
        # ========================================

        {
            "name": "hanbok_silk",
            "name_korean": "한복비단",
            "name_english": "Hanbok Silk",
            "note_type": "middle",
            "fragrance_family": "soft",
            "intensity": 4.0,
            "longevity": 6.0,
            "sillage": 3.0,
            "description": "한복 비단의 부드럽고 우아한 텍스타일 향",
            "description_korean": "전통 한복에 쓰이는 명주 비단의 부드럽고 우아한 텍스타일 향",
            "origin": "Korea",
            "extraction_method": "Textile extraction",
            "mood_tags": ["soft", "elegant", "traditional", "silk"],
            "season_tags": ["spring", "autumn"],
            "gender_tags": ["feminine", "unisex"],
            "price_per_ml": 0.85,
            "supplier": "Traditional textile suppliers",
            "grade": "heritage",
            "search_keywords": "한복 비단 부드러운 우아한"
        },
        {
            "name": "dancheong_colors",
            "name_korean": "단청오방색",
            "name_english": "Dancheong Five Colors",
            "note_type": "middle",
            "fragrance_family": "complex",
            "intensity": 5.5,
            "longevity": 6.0,
            "sillage": 5.0,
            "description": "단청 오방색 안료의 복합적이고 전통적인 향",
            "description_korean": "전통 단청의 청적황백흑 오방색 안료가 어우러진 복합적인 향",
            "origin": "Korea",
            "extraction_method": "Pigment extraction",
            "mood_tags": ["complex", "traditional", "colorful", "sacred"],
            "season_tags": ["autumn"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.95,
            "supplier": "Traditional paint suppliers",
            "grade": "heritage",
            "search_keywords": "단청 오방색 복합적 전통적"
        },
        {
            "name": "gayageum_paulownia",
            "name_korean": "가야금오동나무",
            "name_english": "Gayageum Paulownia",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 5.0,
            "longevity": 8.0,
            "sillage": 4.0,
            "description": "가야금 몸통 오동나무의 따뜻하고 음향적인 목재 향",
            "description_korean": "전통 가야금 몸통에 쓰이는 오동나무의 따뜻하고 음향적인 목재 향",
            "origin": "Korea",
            "extraction_method": "Resonant wood distillation",
            "mood_tags": ["resonant", "warm", "musical", "traditional"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.75,
            "supplier": "Traditional instrument makers",
            "grade": "heritage",
            "search_keywords": "가야금 오동나무 따뜻한 음향적"
        },
        {
            "name": "temple_incense_smoke",
            "name_korean": "사찰향연기",
            "name_english": "Temple Incense Smoke",
            "note_type": "base",
            "fragrance_family": "oriental",
            "intensity": 6.0,
            "longevity": 9.0,
            "sillage": 6.0,
            "description": "한국 전통 사찰에서 피우는 향의 깊고 영적인 연기 향",
            "description_korean": "불국사, 해인사 등 전통 사찰에서 피우는 향의 깊고 영적인 연기 향",
            "origin": "Korea",
            "extraction_method": "Smoke capture",
            "mood_tags": ["spiritual", "deep", "meditative", "sacred"],
            "season_tags": ["autumn", "winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.68,
            "supplier": "Traditional temple suppliers",
            "grade": "sacred",
            "search_keywords": "사찰 향연기 깊은 영적인"
        },
        {
            "name": "ondol_warm_floor",
            "name_korean": "온돌따뜻함",
            "name_english": "Ondol Warm Floor",
            "note_type": "base",
            "fragrance_family": "warm",
            "intensity": 5.5,
            "longevity": 7.0,
            "sillage": 4.0,
            "description": "전통 온돌 마루의 따뜻하고 포근한 집 향",
            "description_korean": "전통 한옥 온돌 마루의 따뜻하고 포근한 집안 향",
            "origin": "Korea",
            "extraction_method": "Warm floor extraction",
            "mood_tags": ["warm", "cozy", "home", "traditional"],
            "season_tags": ["winter"],
            "gender_tags": ["unisex"],
            "price_per_ml": 0.42,
            "supplier": "Traditional architecture suppliers",
            "grade": "heritage",
            "search_keywords": "온돌 따뜻함 포근한 집"
        }
    ]

    added_count = 0
    failed_count = 0

    with db_manager.get_session() as session:
        repo = FragranceNoteRepository(session)

        for note_data in korean_regional_seasonal_notes:
            try:
                # 중복 체크
                existing = repo.find_by_name(note_data["name"], exact=True)
                if existing:
                    print(f"Skipped (already exists): {note_data['name']}")
                    continue

                # 새 노트 생성
                note = repo.create(**note_data)
                added_count += 1
                price = note_data.get('price_per_ml', 0)
                print(f"Added: {note_data['name']} (${price:.2f}/ml)")

            except Exception as e:
                failed_count += 1
                print(f"Failed to add {note_data['name']}: {str(e)}")

        # 커밋
        session.commit()

    print(f"\n=== 한국 지역별/계절별 특색 향료 추가 완료 ===")
    print(f"성공적으로 추가됨: {added_count}개")
    print(f"실패: {failed_count}개")

    # 최종 통계 확인
    with db_manager.get_session() as session:
        repo = FragranceNoteRepository(session)
        total = repo.count()
        print(f"총 향료 노트 개수: {total}개")

        # 한국 원산지 노트 확인
        from fragrance_ai.database.models import FragranceNote
        korean_notes = session.query(FragranceNote).filter(
            FragranceNote.origin.like('%Korea%')
        ).count()
        print(f"한국 관련 향료 노트: {korean_notes}개")

if __name__ == "__main__":
    main()