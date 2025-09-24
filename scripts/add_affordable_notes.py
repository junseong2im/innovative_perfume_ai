#!/usr/bin/env python3
"""
저렴하고 구하기 쉬운 향료 노트 대량 추가 스크립트

일반적으로 구매하기 쉽고 가격이 저렴한 향료들을 데이터베이스에 추가합니다.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from fragrance_ai.database.connection import initialize_database, get_db_session
from fragrance_ai.repositories import FragranceNoteRepository
from fragrance_ai.database.models import FragranceNote


def get_affordable_fragrance_notes() -> List[Dict[str, Any]]:
    """저렴하고 구하기 쉬운 향료 노트 데이터"""

    notes = [
        # === 시트러스 계열 (가장 저렴하고 구하기 쉬움) ===
        {
            "name": "sweet_orange",
            "korean_name": "스위트 오렌지",
            "english_name": "Sweet Orange",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 7.0,
            "longevity": 3.0,
            "sillage": 6.0,
            "description": "달콤하고 상쾌한 오렌지 향",
            "mood": ["fresh", "happy", "energetic"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.15,  # 매우 저렴
            "grade": "standard",
            "supplier": "local_supplier"
        },
        {
            "name": "lemon_essential",
            "korean_name": "레몬 에센셜",
            "english_name": "Lemon Essential",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 8.0,
            "longevity": 2.5,
            "sillage": 7.0,
            "description": "생생한 레몬의 신선함",
            "mood": ["fresh", "clean", "invigorating"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.12,
            "grade": "standard",
            "supplier": "local_supplier"
        },
        {
            "name": "grapefruit_pink",
            "korean_name": "핑크 자몽",
            "english_name": "Pink Grapefruit",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 6.5,
            "longevity": 3.5,
            "sillage": 5.5,
            "description": "달콤하고 상큼한 핑크 자몽",
            "mood": ["fresh", "sweet", "cheerful"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.18,
            "grade": "standard",
            "supplier": "local_supplier"
        },
        {
            "name": "mandarin",
            "korean_name": "만다린",
            "english_name": "Mandarin",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 6.0,
            "longevity": 3.0,
            "sillage": 5.0,
            "description": "부드럽고 달콤한 만다린",
            "mood": ["gentle", "sweet", "comforting"],
            "season": ["spring", "summer", "fall"],
            "price_per_ml": 0.20,
            "grade": "standard",
            "supplier": "local_supplier"
        },

        # === 허브 계열 (저렴하고 재배 가능) ===
        {
            "name": "lavender_bulgarian",
            "korean_name": "불가리안 라벤더",
            "english_name": "Bulgarian Lavender",
            "note_type": "middle",
            "fragrance_family": "aromatic",
            "intensity": 7.0,
            "longevity": 6.0,
            "sillage": 6.0,
            "description": "진정 효과가 있는 클래식 라벤더",
            "mood": ["calm", "relaxing", "peaceful"],
            "season": ["spring", "summer", "fall"],
            "price_per_ml": 0.25,
            "grade": "standard",
            "supplier": "herb_farm"
        },
        {
            "name": "rosemary",
            "korean_name": "로즈마리",
            "english_name": "Rosemary",
            "note_type": "top",
            "fragrance_family": "aromatic",
            "intensity": 8.0,
            "longevity": 4.0,
            "sillage": 6.5,
            "description": "상쾌하고 허브향이 강한 로즈마리",
            "mood": ["fresh", "invigorating", "clean"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.18,
            "grade": "standard",
            "supplier": "herb_farm"
        },
        {
            "name": "peppermint",
            "korean_name": "페퍼민트",
            "english_name": "Peppermint",
            "note_type": "top",
            "fragrance_family": "aromatic",
            "intensity": 9.0,
            "longevity": 3.0,
            "sillage": 8.0,
            "description": "시원하고 상쾌한 페퍼민트",
            "mood": ["fresh", "cooling", "energetic"],
            "season": ["summer"],
            "price_per_ml": 0.15,
            "grade": "standard",
            "supplier": "herb_farm"
        },
        {
            "name": "eucalyptus",
            "korean_name": "유칼립투스",
            "english_name": "Eucalyptus",
            "note_type": "top",
            "fragrance_family": "aromatic",
            "intensity": 8.5,
            "longevity": 4.0,
            "sillage": 7.0,
            "description": "깔끔하고 시원한 유칼립투스",
            "mood": ["fresh", "clean", "medicinal"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.22,
            "grade": "standard",
            "supplier": "herb_farm"
        },

        # === 플로럴 계열 (재배 가능한 저렴한 꽃들) ===
        {
            "name": "geranium_rose",
            "korean_name": "로즈 제라늄",
            "english_name": "Rose Geranium",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 6.5,
            "longevity": 5.0,
            "sillage": 5.5,
            "description": "장미 향이 나는 제라늄",
            "mood": ["romantic", "floral", "green"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.28,
            "grade": "standard",
            "supplier": "flower_farm"
        },
        {
            "name": "jasmine_sambac",
            "korean_name": "삼박 자스민",
            "english_name": "Jasmine Sambac",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 8.0,
            "longevity": 7.0,
            "sillage": 8.0,
            "description": "이국적이고 감성적인 자스민",
            "mood": ["exotic", "romantic", "sensual"],
            "season": ["spring", "summer", "fall"],
            "price_per_ml": 0.45,
            "grade": "standard",
            "supplier": "flower_farm"
        },
        {
            "name": "ylang_ylang",
            "korean_name": "일랑일랑",
            "english_name": "Ylang Ylang",
            "note_type": "middle",
            "fragrance_family": "floral",
            "intensity": 7.5,
            "longevity": 6.5,
            "sillage": 7.5,
            "description": "달콤하고 크리미한 열대 꽃",
            "mood": ["exotic", "sweet", "romantic"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.35,
            "grade": "standard",
            "supplier": "tropical_farm"
        },

        # === 우디 계열 (저렴한 나무 향료) ===
        {
            "name": "cedar_atlas",
            "korean_name": "아틀라스 시더",
            "english_name": "Atlas Cedar",
            "note_type": "base",
            "fragrance_family": "woody",
            "intensity": 6.0,
            "longevity": 8.0,
            "sillage": 6.0,
            "description": "따뜻하고 건조한 시더우드",
            "mood": ["warm", "woody", "masculine"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.30,
            "grade": "standard",
            "supplier": "wood_distillery"
        },
        {
            "name": "pine_needle",
            "korean_name": "솔잎",
            "english_name": "Pine Needle",
            "note_type": "middle",
            "fragrance_family": "woody",
            "intensity": 7.0,
            "longevity": 5.0,
            "sillage": 6.5,
            "description": "신선하고 숲속 같은 솔잎",
            "mood": ["fresh", "natural", "woody"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.20,
            "grade": "standard",
            "supplier": "forest_co-op"
        },

        # === 스파이스 계열 (요리용 향신료에서 추출 가능) ===
        {
            "name": "cinnamon_bark",
            "korean_name": "계피",
            "english_name": "Cinnamon Bark",
            "note_type": "middle",
            "fragrance_family": "spicy",
            "intensity": 8.5,
            "longevity": 7.0,
            "sillage": 7.5,
            "description": "따뜻하고 달콤한 계피",
            "mood": ["warm", "spicy", "comforting"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.25,
            "grade": "standard",
            "supplier": "spice_merchant"
        },
        {
            "name": "black_pepper",
            "korean_name": "블랙 페퍼",
            "english_name": "Black Pepper",
            "note_type": "top",
            "fragrance_family": "spicy",
            "intensity": 7.5,
            "longevity": 4.0,
            "sillage": 6.0,
            "description": "톡 쏘는 듯한 블랙 페퍼",
            "mood": ["spicy", "bold", "energetic"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.22,
            "grade": "standard",
            "supplier": "spice_merchant"
        },
        {
            "name": "ginger_fresh",
            "korean_name": "생강",
            "english_name": "Fresh Ginger",
            "note_type": "top",
            "fragrance_family": "spicy",
            "intensity": 7.0,
            "longevity": 4.5,
            "sillage": 6.5,
            "description": "신선하고 따뜻한 생강",
            "mood": ["warm", "spicy", "invigorating"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.18,
            "grade": "standard",
            "supplier": "local_supplier"
        },

        # === 바닐라 및 발삼 계열 (합성으로 저렴하게) ===
        {
            "name": "vanilla_synthetic",
            "korean_name": "합성 바닐라",
            "english_name": "Synthetic Vanilla",
            "note_type": "base",
            "fragrance_family": "gourmand",
            "intensity": 7.0,
            "longevity": 8.5,
            "sillage": 7.0,
            "description": "달콤하고 크리미한 바닐라",
            "mood": ["sweet", "comforting", "warm"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.12,  # 합성이라 매우 저렴
            "grade": "synthetic",
            "supplier": "chemical_supplier"
        },
        {
            "name": "coconut_essence",
            "korean_name": "코코넛 에센스",
            "english_name": "Coconut Essence",
            "note_type": "middle",
            "fragrance_family": "gourmand",
            "intensity": 6.0,
            "longevity": 5.5,
            "sillage": 6.0,
            "description": "열대적이고 크리미한 코코넛",
            "mood": ["tropical", "sweet", "creamy"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.16,
            "grade": "synthetic",
            "supplier": "food_grade_supplier"
        },

        # === 프루티 계열 (과일 에센스) ===
        {
            "name": "apple_green",
            "korean_name": "그린 애플",
            "english_name": "Green Apple",
            "note_type": "top",
            "fragrance_family": "fruity",
            "intensity": 6.5,
            "longevity": 3.5,
            "sillage": 5.5,
            "description": "상큼하고 아삭한 그린 애플",
            "mood": ["fresh", "crisp", "youthful"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.14,
            "grade": "synthetic",
            "supplier": "food_grade_supplier"
        },
        {
            "name": "strawberry",
            "korean_name": "딸기",
            "english_name": "Strawberry",
            "note_type": "top",
            "fragrance_family": "fruity",
            "intensity": 6.0,
            "longevity": 3.0,
            "sillage": 5.0,
            "description": "달콤하고 상큼한 딸기",
            "mood": ["sweet", "playful", "romantic"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.15,
            "grade": "synthetic",
            "supplier": "food_grade_supplier"
        },
        {
            "name": "peach",
            "korean_name": "복숭아",
            "english_name": "Peach",
            "note_type": "middle",
            "fragrance_family": "fruity",
            "intensity": 5.5,
            "longevity": 4.0,
            "sillage": 5.0,
            "description": "부드럽고 달콤한 복숭아",
            "mood": ["sweet", "soft", "feminine"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.17,
            "grade": "synthetic",
            "supplier": "food_grade_supplier"
        },

        # === 아쿠아틱/마린 계열 (합성) ===
        {
            "name": "sea_breeze",
            "korean_name": "바다 바람",
            "english_name": "Sea Breeze",
            "note_type": "top",
            "fragrance_family": "aquatic",
            "intensity": 6.0,
            "longevity": 4.0,
            "sillage": 6.0,
            "description": "상쾌한 바다 바람의 느낌",
            "mood": ["fresh", "clean", "marine"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.20,
            "grade": "synthetic",
            "supplier": "chemical_supplier"
        },
        {
            "name": "rain_water",
            "korean_name": "빗물",
            "english_name": "Rain Water",
            "note_type": "top",
            "fragrance_family": "aquatic",
            "intensity": 5.0,
            "longevity": 3.5,
            "sillage": 4.5,
            "description": "깨끗하고 투명한 빗물의 향",
            "mood": ["clean", "pure", "refreshing"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.18,
            "grade": "synthetic",
            "supplier": "chemical_supplier"
        },

        # === 머스크 계열 (합성 머스크) ===
        {
            "name": "white_musk",
            "korean_name": "화이트 머스크",
            "english_name": "White Musk",
            "note_type": "base",
            "fragrance_family": "musky",
            "intensity": 6.0,
            "longevity": 8.0,
            "sillage": 5.5,
            "description": "깨끗하고 부드러운 화이트 머스크",
            "mood": ["clean", "soft", "intimate"],
            "season": ["all"],
            "price_per_ml": 0.25,
            "grade": "synthetic",
            "supplier": "chemical_supplier"
        },
        {
            "name": "pink_musk",
            "korean_name": "핑크 머스크",
            "english_name": "Pink Musk",
            "note_type": "base",
            "fragrance_family": "musky",
            "intensity": 5.5,
            "longevity": 7.5,
            "sillage": 5.0,
            "description": "부드럽고 파우더리한 핑크 머스크",
            "mood": ["soft", "powdery", "feminine"],
            "season": ["all"],
            "price_per_ml": 0.28,
            "grade": "synthetic",
            "supplier": "chemical_supplier"
        },

        # === 그린 계열 (잎과 줄기 향료) ===
        {
            "name": "green_leaves",
            "korean_name": "푸른 잎",
            "english_name": "Green Leaves",
            "note_type": "top",
            "fragrance_family": "green",
            "intensity": 6.5,
            "longevity": 4.0,
            "sillage": 5.5,
            "description": "신선한 녹색 잎사귀의 향",
            "mood": ["green", "natural", "fresh"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.22,
            "grade": "standard",
            "supplier": "botanical_extract"
        },
        {
            "name": "grass_cut",
            "korean_name": "갓 깎은 잔디",
            "english_name": "Cut Grass",
            "note_type": "top",
            "fragrance_family": "green",
            "intensity": 7.0,
            "longevity": 3.0,
            "sillage": 6.0,
            "description": "갓 깎은 잔디의 싱그러운 향",
            "mood": ["fresh", "green", "natural"],
            "season": ["spring", "summer"],
            "price_per_ml": 0.20,
            "grade": "standard",
            "supplier": "botanical_extract"
        },

        # === 추가 시트러스 (더 다양한 종류) ===
        {
            "name": "yuzu",
            "korean_name": "유자",
            "english_name": "Yuzu",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 7.5,
            "longevity": 3.5,
            "sillage": 6.5,
            "description": "일본의 독특한 시트러스 유자",
            "mood": ["fresh", "exotic", "zesty"],
            "season": ["winter", "spring"],
            "price_per_ml": 0.35,
            "grade": "standard",
            "supplier": "asian_supplier"
        },
        {
            "name": "tangerine",
            "korean_name": "귤",
            "english_name": "Tangerine",
            "note_type": "top",
            "fragrance_family": "citrus",
            "intensity": 6.0,
            "longevity": 3.0,
            "sillage": 5.0,
            "description": "달콤하고 부드러운 귤",
            "mood": ["sweet", "cheerful", "warm"],
            "season": ["fall", "winter"],
            "price_per_ml": 0.16,
            "grade": "standard",
            "supplier": "local_supplier"
        }
    ]

    return notes


def add_notes_to_database():
    """데이터베이스에 향료 노트 추가"""
    print("Adding affordable fragrance notes to database...")

    try:
        # 데이터베이스 초기화
        initialize_database()

        # 향료 노트 데이터 가져오기
        notes_data = get_affordable_fragrance_notes()

        added_count = 0
        skipped_count = 0

        with get_db_session() as session:
            note_repo = FragranceNoteRepository(session)

            for note_data in notes_data:
                try:
                    # 중복 확인
                    existing = note_repo.find_by_name(note_data['name'], exact=True)
                    if existing:
                        print(f"Skipping duplicate: {note_data['name']}")
                        skipped_count += 1
                        continue

                    # FragranceNote 생성
                    note = FragranceNote(
                        name=note_data['name'],
                        name_korean=note_data['korean_name'],
                        name_english=note_data['english_name'],
                        note_type=note_data['note_type'],
                        fragrance_family=note_data['fragrance_family'],
                        intensity=note_data['intensity'],
                        longevity=note_data['longevity'],
                        sillage=note_data['sillage'],
                        description=note_data['description'],
                        mood_tags=note_data['mood'],
                        season_tags=note_data['season'],
                        price_per_ml=note_data['price_per_ml'],
                        grade=note_data['grade'],
                        supplier=note_data['supplier'],
                        search_keywords=f"{note_data['name']} {note_data['korean_name']} {note_data['english_name']} {' '.join(note_data['mood'])} {' '.join(note_data['season'])}"
                    )

                    session.add(note)
                    added_count += 1
                    print(f"Added: {note_data['name']} (${note_data['price_per_ml']:.2f}/ml)")

                except Exception as e:
                    print(f"Failed to add {note_data['name']}: {str(e)}")

            # 커밋
            session.commit()

        print(f"\nSUCCESS: Added {added_count} new affordable fragrance notes")
        print(f"Skipped {skipped_count} duplicates")

        # 전체 노트 수 확인
        with get_db_session() as session:
            note_repo = FragranceNoteRepository(session)
            total_notes = note_repo.count()
            print(f"Total notes in database: {total_notes}")

            # 가격 통계
            all_notes = note_repo.get_all()
            prices = [note.price_per_ml for note in all_notes if note.price_per_ml]
            if prices:
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
                print(f"Price range: ${min_price:.2f} - ${max_price:.2f} per ml")
                print(f"Average price: ${avg_price:.2f} per ml")

    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    add_notes_to_database()