import sqlite3
import sys
import io
from datetime import datetime
import uuid

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# WebFetch로 가져온 실제 향수 데이터
WEBFETCH_PERFUMES = [
    # 1. Chanel No 5 (1921)
    {
        "brand": "Chanel",
        "name": "No 5",
        "year": 1921,
        "gender": "Women",
        "style": "floral",
        "top": {"Aldehydes": 7.5, "Bergamot": 7.5, "Lemon": 7.5, "Neroli": 7.5},
        "middle": {"Grasse Jasmine": 10, "Iris": 10, "Lily of the Valley": 10, "May Rose": 10},
        "base": {"Sandalwood": 7.5, "Vanilla": 7.5, "Amber": 7.5, "Vetiver": 7.5}
    },
    # 2. Dior Sauvage (2015)
    {
        "brand": "Dior",
        "name": "Sauvage",
        "year": 2015,
        "gender": "Men",
        "style": "fresh",
        "top": {"Calabrian Bergamot": 7.5, "Sichuan Pepper": 7.5, "Pink Pepper": 7.5, "Provençal Lavender": 7.5},
        "middle": {"Geranium": 20, "Elemi Resin": 20},
        "base": {"Ambroxan": 10, "Patchouli": 10, "Vetiver": 10}
    },
    # 3. YSL Black Opium (2014)
    {
        "brand": "Yves Saint Laurent",
        "name": "Black Opium",
        "year": 2014,
        "gender": "Women",
        "style": "oriental",
        "top": {"Orange Blossom": 15, "Pink Pepper": 15},
        "middle": {"Coffee": 13.3, "Jasmine": 13.3, "Bitter Almond": 13.3},
        "base": {"Vanilla": 10, "Patchouli": 10, "Cedar": 10}
    },
    # 4. Giorgio Armani Acqua di Gio (1995) - Women's version
    {
        "brand": "Giorgio Armani",
        "name": "Acqua di Gio",
        "year": 1995,
        "gender": "Women",
        "style": "fresh",
        "top": {"Lemon": 4.3, "Peach": 4.3, "Peony": 4.3, "Banana Leaf": 4.3, "Pineapple": 4.3, "Violet": 4.3, "Vodka": 4.3},
        "middle": {"Lily of the Valley": 6.7, "Freesia": 6.7, "Hyacinth": 6.7, "Ylang-Ylang": 6.7, "Jasmine": 6.7, "Lily": 6.7},
        "base": {"Musk": 6, "Cedarwood": 6, "Amber": 6, "Sandalwood": 6, "Styrax": 6}
    },
    # 5. Gucci Guilty (2010)
    {
        "brand": "Gucci",
        "name": "Guilty",
        "year": 2010,
        "gender": "Women",
        "style": "floral",
        "top": {"Mandarin Orange": 15, "Pink Pepper": 15},
        "middle": {"Lilac": 8, "Egyptian Geranium": 8, "Peach": 8, "Raspberry": 8},
        "base": {"Amber": 13.3, "Patchouli": 13.3, "Vanilla": 3.4}
    },
    # 6. Lancôme La Vie est Belle (2012)
    {
        "brand": "Lancôme",
        "name": "La Vie est Belle",
        "year": 2012,
        "gender": "Women",
        "style": "floral",
        "top": {"Blackcurrant": 15, "Pear": 15},
        "middle": {"Iris": 13.3, "Jasmine": 13.3, "Orange Blossom": 13.3},
        "base": {"Praliné": 7.5, "Patchouli": 7.5, "Tonka Bean": 7.5, "Vanilla": 7.5}
    },
    # 7. Viktor & Rolf Flowerbomb (2004)
    {
        "brand": "Viktor & Rolf",
        "name": "Flowerbomb",
        "year": 2004,
        "gender": "Women",
        "style": "floral",
        "top": {"Tea": 15, "Bergamot": 15},
        "middle": {"Orchid": 13.3, "Freesia": 13.3, "Jasmine": 13.3},
        "base": {"Patchouli": 15, "Rose": 15}
    },
    # 8. Rabanne 1 Million (2008)
    {
        "brand": "Paco Rabanne",
        "name": "1 Million",
        "year": 2008,
        "gender": "Men",
        "style": "oriental",
        "top": {"Red Mandarin Orange": 15, "Peppermint": 15},
        "middle": {"Cinnamon": 20, "Rose Absolute": 20},
        "base": {"Amberketal": 10, "Leather": 10, "Amber": 10}
    },
    # 9. Dolce & Gabbana Light Blue (2001)
    {
        "brand": "Dolce & Gabbana",
        "name": "Light Blue",
        "year": 2001,
        "gender": "Women",
        "style": "fresh",
        "top": {"Sicilian Citron": 10, "Apple": 10, "Bluebell": 10},
        "middle": {"Bamboo": 13.3, "Jasmine": 13.3, "Rose": 13.3},
        "base": {"Cedarwood": 10, "Musk": 10, "Amber": 10}
    },
    # 10. Versace Eros (2012)
    {
        "brand": "Versace",
        "name": "Eros",
        "year": 2012,
        "gender": "Men",
        "style": "fresh",
        "top": {"Green Apple": 10, "Mint": 10, "Italian Lemon": 10},
        "middle": {"Venezuelan Tonka Bean": 10, "Ambroxan": 10, "Geranium": 10},
        "base": {"Bourbon Vanilla": 5, "Atlas Cedar": 5, "Virginia Cedar": 5, "Oakmoss": 5, "Vetiver": 5, "Sandalwood": 5}
    },
]

def find_note_id(note_name, available_notes):
    """향료명으로 ID 찾기 - 개선된 매칭"""
    # 소문자 변환 + 공백과 하이픈을 언더스코어로
    note_name_clean = note_name.strip().lower().replace(' ', '_').replace('-', '_')

    # 정확히 일치
    for note_id, db_name in available_notes:
        db_name_clean = db_name.strip().lower()
        if note_name_clean == db_name_clean:
            return note_id

    # 부분 일치 (언더스코어 제거 후)
    note_simple = note_name_clean.replace('_', '')
    for note_id, db_name in available_notes:
        db_name_clean = db_name.strip().lower()
        db_simple = db_name_clean.replace('_', '')

        if note_simple == db_simple:
            return note_id

        # 더 관대한 매칭 (한쪽이 다른 쪽을 포함)
        if len(note_simple) >= 4 and len(db_simple) >= 4:
            if note_simple in db_simple or db_simple in note_simple:
                return note_id

    return None

def add_perfume(perfume_data, cursor, available_notes):
    """향수 1개 추가"""
    brand = perfume_data["brand"]
    name = perfume_data["name"]
    year = perfume_data["year"]
    gender = perfume_data["gender"]
    style = perfume_data["style"]

    full_name = f"{brand} {name}"

    # 레시피 생성
    recipe_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    # 복잡도 계산
    total_notes = len(perfume_data["top"]) + len(perfume_data["middle"]) + len(perfume_data["base"])
    complexity = min(10, max(1, total_notes))

    cursor.execute("""
        INSERT INTO recipes (
            id, name, name_korean, recipe_type, fragrance_family,
            complexity, estimated_cost, batch_size_ml,
            description, concept, target_audience,
            status, is_public, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        recipe_id,
        full_name,
        f"{brand} {name}",
        "premium",
        style,
        complexity,
        0.0,
        100,
        f"{brand}의 {year}년 출시 향수 {name}",
        f"{style} 계열의 {gender} 향수",
        gender,
        "approved",
        True,
        now,
        now
    ))

    # 노트 추가
    added_notes = 0
    missing_notes = []

    for position, notes_dict in [("top", perfume_data["top"]), ("middle", perfume_data["middle"]), ("base", perfume_data["base"])]:
        for i, (note_name, percentage) in enumerate(notes_dict.items()):
            note_id = find_note_id(note_name, available_notes)

            if note_id:
                ingredient_id = str(uuid.uuid4())
                note_role = 'primary' if i == 0 else 'accent'

                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO recipe_ingredients (
                            id, recipe_id, note_id, percentage,
                            role, note_position, is_optional,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ingredient_id,
                        recipe_id,
                        note_id,
                        percentage,
                        note_role,
                        position,
                        False,
                        now,
                        now
                    ))
                    added_notes += 1
                except Exception as e:
                    pass
            else:
                missing_notes.append(note_name)

    return added_notes, missing_notes

if __name__ == "__main__":
    print("=== WebFetch로 수집한 실제 향수 데이터 입력 ===\n")

    conn = sqlite3.connect('fragrance_ai.db')
    cursor = conn.cursor()

    # 향료 목록 가져오기
    cursor.execute("SELECT id, name FROM fragrance_notes")
    available_notes = [(row[0], row[1]) for row in cursor.fetchall()]
    print(f"사용 가능한 향료: {len(available_notes)}개\n")

    # 기존 레시피 확인
    cursor.execute("SELECT name FROM recipes")
    existing_names = {row[0] for row in cursor.fetchall()}

    added = 0
    skipped = 0
    total_notes_added = 0
    all_missing_notes = set()

    for perfume_data in WEBFETCH_PERFUMES:
        full_name = f"{perfume_data['brand']} {perfume_data['name']}"

        if full_name in existing_names:
            print(f"⊘ {full_name} - 이미 존재")
            skipped += 1
            continue

        try:
            notes_added, missing = add_perfume(perfume_data, cursor, available_notes)
            total_notes_added += notes_added
            all_missing_notes.update(missing)

            print(f"✓ {full_name} ({perfume_data['year']}) - {notes_added}개 노트")
            added += 1

        except Exception as e:
            print(f"✗ {full_name}: {e}")

    conn.commit()
    conn.close()

    print(f"\n=== 완료 ===")
    print(f"✓ {added}개 향수 추가")
    print(f"✓ {total_notes_added}개 노트 관계 추가")
    print(f"⊘ {skipped}개 중복")

    if all_missing_notes:
        print(f"\n매칭되지 않은 노트 ({len(all_missing_notes)}개):")
        for note in sorted(all_missing_notes)[:20]:
            print(f"  - {note}")
        if len(all_missing_notes) > 20:
            print(f"  ... 외 {len(all_missing_notes) - 20}개 더")
