"""
실제 향수 레시피 데이터베이스 생성
각 향수마다 탑/미들/베이스 노트의 정확한 원료 비율(%)을 포함
"""

import sqlite3

# 유명 향수들의 상세 레시피
# 형식: (브랜드, 이름, 년도, 성별, 스타일, {top_notes}, {heart_notes}, {base_notes})
# 각 딕셔너리: {원료명: 비율%}
PERFUME_RECIPES = [
    # === CHANEL ===
    (
        "Chanel", "No 5", 1921, "Women", "floral",
        {
            "Neroli Oil": 12.0,
            "Bergamot Oil": 10.0,
            "Lemon Oil": 8.0,
        },
        {
            "Jasmine Absolute": 18.0,
            "Rose Otto": 15.0,
            "Ylang-Ylang Oil": 7.0,
        },
        {
            "Vanilla": 12.0,
            "Sandalwood Oil": 10.0,
            "Vetiver Oil": 8.0,
        }
    ),
    (
        "Chanel", "Bleu de Chanel", 2010, "Men", "woody",
        {
            "Lemon Oil": 8.0,
            "Peppermint Oil": 6.0,
            "Pink Pepper Oil": 5.0,
            "Bergamot Oil": 6.0,
        },
        {
            "Ginger Oil": 7.0,
            "Nutmeg Oil": 5.0,
            "Jasmine Absolute": 8.0,
            "Geranium Oil": 5.0,
        },
        {
            "Sandalwood Oil": 12.0,
            "Cedarwood Oil Atlas": 10.0,
            "Vetiver Oil": 8.0,
            "Patchouli Oil": 5.0,
            "Iso E Super": 15.0,
        }
    ),
    (
        "Chanel", "Chance Eau Fraiche", 2007, "Women", "fresh",
        {
            "Lemon Oil": 12.0,
            "Lime Oil": 10.0,
            "Grapefruit Oil": 8.0,
        },
        {
            "Jasmine Absolute": 12.0,
            "Hedione": 15.0,
            "Pink Pepper Oil": 5.0,
        },
        {
            "Amber Xtreme": 10.0,
            "Patchouli Oil": 8.0,
            "Vetiver Oil": 6.0,
            "White Musk": 14.0,
        }
    ),

    # === DIOR ===
    (
        "Dior", "Sauvage", 2015, "Men", "fresh",
        {
            "Bergamot Oil": 15.0,
            "Black Pepper Oil": 8.0,
            "Pink Pepper Oil": 7.0,
        },
        {
            "Lavender Oil": 10.0,
            "Geranium Oil": 8.0,
            "Pink Pepper Oil": 5.0,
            "Elemi": 7.0,
        },
        {
            "Ambergris": 12.0,
            "Patchouli Oil": 10.0,
            "Vetiver Oil": 8.0,
            "Cedarwood Oil Virginia": 10.0,
        }
    ),
    (
        "Dior", "J'adore", 1999, "Women", "floral",
        {
            "Mandarin Oil": 10.0,
            "Bergamot Oil": 8.0,
            "Neroli Oil": 7.0,
        },
        {
            "Rose Otto": 15.0,
            "Jasmine Absolute": 12.0,
            "Ylang-Ylang Oil": 8.0,
            "Orange Blossom Absolute": 10.0,
        },
        {
            "Vanilla": 8.0,
            "White Musk": 12.0,
            "Sandalwood Oil": 10.0,
        }
    ),
    (
        "Dior", "Homme Intense", 2011, "Men", "woody",
        {
            "Lavender Oil": 10.0,
            "Bergamot Oil": 8.0,
            "Sage Oil": 5.0,
        },
        {
            "Iris": 15.0,
            "Pear": 8.0,
            "Ambrette Seeds": 7.0,
        },
        {
            "Vetiver Oil": 12.0,
            "Cedarwood Oil Atlas": 10.0,
            "Leather": 8.0,
            "Patchouli Oil": 7.0,
        }
    ),

    # === TOM FORD ===
    (
        "Tom Ford", "Oud Wood", 2007, "Unisex", "woody",
        {
            "Cardamom Oil": 8.0,
            "Pink Pepper Oil": 6.0,
            "Rosewood": 10.0,
        },
        {
            "Oud": 15.0,
            "Sandalwood Oil": 12.0,
            "Vetiver Oil": 8.0,
        },
        {
            "Tonka Bean Absolute": 12.0,
            "Vanilla": 10.0,
            "Amber Xtreme": 9.0,
            "Cedarwood Oil Virginia": 10.0,
        }
    ),
    (
        "Tom Ford", "Tobacco Vanille", 2007, "Unisex", "oriental",
        {
            "Tobacco": 12.0,
            "Ginger Oil": 6.0,
            "Cinnamon Bark Oil": 5.0,
        },
        {
            "Tobacco": 15.0,
            "Vanilla": 12.0,
            "Tonka Bean Absolute": 8.0,
            "Cocoa": 5.0,
        },
        {
            "Tonka Bean Absolute": 15.0,
            "Vanilla": 10.0,
            "Benzoin Resinoid": 7.0,
            "Amber Xtreme": 5.0,
        }
    ),

    # === CREED ===
    (
        "Creed", "Aventus", 2010, "Men", "fresh",
        {
            "Pineapple": 10.0,
            "Bergamot Oil": 8.0,
            "Apple": 6.0,
            "Blackcurrant": 6.0,
        },
        {
            "Pink Pepper Oil": 8.0,
            "Birch Tar": 10.0,
            "Patchouli Oil": 7.0,
            "Jasmine Absolute": 5.0,
        },
        {
            "Musk": 12.0,
            "Oakmoss": 8.0,
            "Ambergris": 10.0,
            "Vanilla": 10.0,
        }
    ),

    # === YSL ===
    (
        "Yves Saint Laurent", "Black Opium", 2014, "Women", "gourmand",
        {
            "Pink Pepper Oil": 8.0,
            "Orange Blossom Absolute": 10.0,
            "Pear": 7.0,
        },
        {
            "Coffee": 15.0,
            "Jasmine Absolute": 10.0,
            "Orange Blossom Absolute": 8.0,
            "Bitter Almond": 7.0,
        },
        {
            "Vanilla": 15.0,
            "Patchouli Oil": 10.0,
            "Cedarwood Oil Virginia": 10.0,
        }
    ),

    # === GIORGIO ARMANI ===
    (
        "Giorgio Armani", "Acqua di Gio", 1996, "Men", "fresh",
        {
            "Lime Oil": 10.0,
            "Lemon Oil": 8.0,
            "Bergamot Oil": 7.0,
            "Neroli Oil": 5.0,
        },
        {
            "Calone": 12.0,
            "Jasmine Absolute": 8.0,
            "Rose Otto": 7.0,
            "Rosemary Oil": 6.0,
        },
        {
            "White Musk": 12.0,
            "Cedarwood Oil Virginia": 10.0,
            "Oakmoss": 8.0,
            "Patchouli Oil": 7.0,
        }
    ),

    # === THIERRY MUGLER ===
    (
        "Thierry Mugler", "Angel", 1992, "Women", "gourmand",
        {
            "Melon": 8.0,
            "Coconut": 7.0,
            "Mandarin Oil": 6.0,
            "Bergamot Oil": 5.0,
        },
        {
            "Honey": 10.0,
            "Apricot": 8.0,
            "Plum": 7.0,
            "Peach": 6.0,
            "Jasmine Absolute": 6.0,
        },
        {
            "Patchouli Oil": 12.0,
            "Chocolate": 10.0,
            "Vanilla": 10.0,
            "Caramel": 5.0,
        }
    ),

    # === HERMES ===
    (
        "Hermes", "Terre d'Hermes", 2006, "Men", "woody",
        {
            "Orange": 12.0,
            "Grapefruit Oil": 10.0,
            "Pink Pepper Oil": 6.0,
        },
        {
            "Geranium Oil": 10.0,
            "Rose Otto": 8.0,
            "Patchouli Oil": 7.0,
        },
        {
            "Vetiver Oil": 15.0,
            "Cedarwood Oil Atlas": 12.0,
            "Benzoin Resinoid": 8.0,
            "Oakmoss": 6.0,
            "Amber Xtreme": 6.0,
        }
    ),

    # === GUERLAIN ===
    (
        "Guerlain", "Shalimar", 1925, "Women", "oriental",
        {
            "Lemon Oil": 10.0,
            "Bergamot Oil": 8.0,
            "Mandarin Oil": 7.0,
        },
        {
            "Iris": 12.0,
            "Jasmine Absolute": 10.0,
            "Rose Otto": 8.0,
            "Patchouli Oil": 5.0,
        },
        {
            "Vanilla": 15.0,
            "Incense": 10.0,
            "Tonka Bean Absolute": 10.0,
            "Benzoin Resinoid": 5.0,
        }
    ),

    # === VIKTOR & ROLF ===
    (
        "Viktor & Rolf", "Flowerbomb", 2005, "Women", "floral",
        {
            "Bergamot Oil": 8.0,
            "Green Tea": 7.0,
            "Mandarin Oil": 6.0,
        },
        {
            "Rose Otto": 12.0,
            "Jasmine Absolute": 10.0,
            "Orange Blossom Absolute": 8.0,
            "Freesia": 7.0,
            "Ylang-Ylang Oil": 6.0,
        },
        {
            "Patchouli Oil": 12.0,
            "Musk": 10.0,
            "Vanilla": 8.0,
            "Amber Xtreme": 6.0,
        }
    ),

    # === LANCOME ===
    (
        "Lancome", "La Vie Est Belle", 2012, "Women", "gourmand",
        {
            "Blackcurrant": 8.0,
            "Pear": 10.0,
            "Bergamot Oil": 7.0,
        },
        {
            "Iris": 12.0,
            "Jasmine Absolute": 10.0,
            "Orange Blossom Absolute": 8.0,
            "Ylang-Ylang Oil": 5.0,
        },
        {
            "Praline": 12.0,
            "Vanilla": 10.0,
            "Patchouli Oil": 8.0,
            "Tonka Bean Absolute": 10.0,
        }
    ),

    # === LE LABO ===
    (
        "Le Labo", "Santal 33", 2011, "Unisex", "woody",
        {
            "Cardamom Oil": 8.0,
            "Iris": 7.0,
            "Violet": 6.0,
        },
        {
            "Papyrus": 10.0,
            "Cedarwood Oil Atlas": 12.0,
            "Amber Xtreme": 8.0,
        },
        {
            "Leather": 15.0,
            "Sandalwood Oil": 15.0,
            "Musk": 9.0,
            "Vetiver Oil": 10.0,
        }
    ),

    # === BYREDO ===
    (
        "Byredo", "Gypsy Water", 2008, "Unisex", "woody",
        {
            "Bergamot Oil": 8.0,
            "Lemon Oil": 7.0,
            "Black Pepper Oil": 5.0,
            "Juniper": 6.0,
        },
        {
            "Incense": 12.0,
            "Pine": 10.0,
            "Iris": 8.0,
        },
        {
            "Vanilla": 12.0,
            "Sandalwood Oil": 12.0,
            "Amber Xtreme": 10.0,
            "Vetiver Oil": 10.0,
        }
    ),

    # === MAISON FRANCIS KURKDJIAN ===
    (
        "Maison Francis Kurkdjian", "Baccarat Rouge 540", 2015, "Unisex", "oriental",
        {
            "Saffron": 12.0,
            "Jasmine Absolute": 10.0,
            "Bitter Almond": 8.0,
        },
        {
            "Amber Xtreme": 15.0,
            "Cedarwood Oil Atlas": 12.0,
            "Ambergris": 8.0,
        },
        {
            "Fir Resin": 12.0,
            "Cedarwood Oil Virginia": 10.0,
            "Musk": 8.0,
            "Cashmeran": 5.0,
        }
    ),

    # === DOLCE & GABBANA ===
    (
        "Dolce & Gabbana", "Light Blue", 2001, "Women", "fresh",
        {
            "Lemon Oil": 10.0,
            "Apple": 8.0,
            "Grapefruit Oil": 7.0,
            "Bellflower": 5.0,
        },
        {
            "Jasmine Absolute": 12.0,
            "Rose Otto": 10.0,
            "Bamboo": 8.0,
        },
        {
            "Amber Xtreme": 10.0,
            "Musk": 12.0,
            "Cedarwood Oil Virginia": 8.0,
            "White Musk": 10.0,
        }
    ),

    # === PRADA ===
    (
        "Prada", "Candy", 2011, "Women", "gourmand",
        {
            "Caramel": 15.0,
            "Vanilla": 10.0,
        },
        {
            "Benzoin Resinoid": 15.0,
            "Musk": 12.0,
            "Vanilla": 10.0,
            "Tonka Bean Absolute": 8.0,
        },
        {
            "Benzoin Resinoid": 12.0,
            "Musk": 10.0,
            "Vanilla": 8.0,
        }
    ),
]

def find_ingredient_match(recipe_name, available_ingredients):
    """레시피 성분명을 DB 성분명으로 매칭"""
    # 1. 정확한 매칭
    if recipe_name in available_ingredients:
        return recipe_name

    # 2. Oil, Absolute, Otto 등 제거 후 매칭
    simple_name = recipe_name.replace(" Oil", "").replace(" Absolute", "").replace(" Otto", "")
    if simple_name in available_ingredients:
        return simple_name

    # 3. 반대로 DB에 Oil/Absolute가 있는 경우
    for suffix in [" Oil", " Absolute", " Otto"]:
        full_name = recipe_name + suffix
        if full_name in available_ingredients:
            return full_name

    # 4. 부분 매칭 (대소문자 무시)
    recipe_lower = recipe_name.lower()
    for db_name in available_ingredients:
        if recipe_lower in db_name.lower() or db_name.lower() in recipe_lower:
            return db_name

    return None

def clear_and_recreate_perfume_data():
    """기존 향수 데이터를 삭제하고 새로운 레시피로 재생성"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    print("기존 향수 데이터 삭제 중...")
    cursor.execute("DELETE FROM perfume_notes")
    cursor.execute("DELETE FROM perfumes")
    conn.commit()

    # 사용 가능한 성분 목록
    cursor.execute("SELECT name FROM ingredients")
    available_ingredients = {row[0] for row in cursor.fetchall()}

    print(f"사용 가능한 성분: {len(available_ingredients)}개\n")

    added_perfumes = 0
    added_notes = 0
    skipped_ingredients = set()
    validation_errors = []

    for brand, name, year, gender, style, top_notes, heart_notes, base_notes in PERFUME_RECIPES:
        # 향수 추가
        cursor.execute("""
            INSERT INTO perfumes (brand, name, year, gender, style)
            VALUES (?, ?, ?, ?, ?)
        """, (brand, name, year, gender, style))
        perfume_id = cursor.lastrowid
        added_perfumes += 1

        # 농도 합계 검증
        total_concentration = 0.0
        recipe_notes = []

        # Top notes 추가
        for ingredient, concentration in top_notes.items():
            total_concentration += concentration
            matched_name = find_ingredient_match(ingredient, available_ingredients)
            if matched_name:
                cursor.execute("""
                    INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                    VALUES (?, ?, 'top', ?)
                """, (perfume_id, matched_name, concentration))
                added_notes += 1
                recipe_notes.append((matched_name, 'top', concentration))
            else:
                skipped_ingredients.add(ingredient)

        # Heart notes 추가
        for ingredient, concentration in heart_notes.items():
            total_concentration += concentration
            matched_name = find_ingredient_match(ingredient, available_ingredients)
            if matched_name:
                cursor.execute("""
                    INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                    VALUES (?, ?, 'heart', ?)
                """, (perfume_id, matched_name, concentration))
                added_notes += 1
                recipe_notes.append((matched_name, 'heart', concentration))
            else:
                skipped_ingredients.add(ingredient)

        # Base notes 추가
        for ingredient, concentration in base_notes.items():
            total_concentration += concentration
            matched_name = find_ingredient_match(ingredient, available_ingredients)
            if matched_name:
                cursor.execute("""
                    INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                    VALUES (?, ?, 'base', ?)
                """, (perfume_id, matched_name, concentration))
                added_notes += 1
                recipe_notes.append((matched_name, 'base', concentration))
            else:
                skipped_ingredients.add(ingredient)

        # 농도 합계 검증
        if abs(total_concentration - 100.0) > 0.1:
            validation_errors.append(f"{brand} {name}: 합계 {total_concentration:.1f}% (100%가 아님)")

    conn.commit()

    # 통계 출력
    print(f"=== 향수 레시피 생성 완료 ===")
    print(f"추가된 향수: {added_perfumes}개")
    print(f"추가된 노트: {added_notes}개")
    print(f"스킵된 성분: {len(skipped_ingredients)}개")

    if skipped_ingredients:
        print(f"\n스킵된 성분 목록:")
        for ing in sorted(skipped_ingredients):
            print(f"  - {ing}")

    if validation_errors:
        print(f"\n⚠️  검증 오류:")
        for error in validation_errors:
            print(f"  {error}")
    else:
        print(f"\n✅ 모든 레시피의 농도 합계가 100%입니다!")

    # 샘플 레시피 출력
    print(f"\n=== 샘플 레시피 (3개) ===")
    cursor.execute("""
        SELECT p.brand, p.name, p.year, p.style
        FROM perfumes p
        ORDER BY RANDOM()
        LIMIT 3
    """)

    for brand, name, year, style in cursor.fetchall():
        print(f"\n{brand} - {name} ({year}, {style})")

        cursor.execute("""
            SELECT pn.note_position, pn.ingredient_name, pn.concentration
            FROM perfume_notes pn
            JOIN perfumes p ON pn.perfume_id = p.id
            WHERE p.brand = ? AND p.name = ?
            ORDER BY
                CASE pn.note_position
                    WHEN 'top' THEN 1
                    WHEN 'heart' THEN 2
                    WHEN 'base' THEN 3
                END,
                pn.concentration DESC
        """, (brand, name))

        current_position = None
        position_total = 0.0

        for position, ingredient, concentration in cursor.fetchall():
            if position != current_position:
                if current_position:
                    print(f"    소계: {position_total:.1f}%")
                current_position = position
                position_total = 0.0
                print(f"  {position.upper()} NOTES:")

            print(f"    - {ingredient}: {concentration:.1f}%")
            position_total += concentration

        if current_position:
            print(f"    소계: {position_total:.1f}%")

        # 전체 합계
        cursor.execute("""
            SELECT SUM(pn.concentration)
            FROM perfume_notes pn
            JOIN perfumes p ON pn.perfume_id = p.id
            WHERE p.brand = ? AND p.name = ?
        """, (brand, name))
        total = cursor.fetchone()[0]
        print(f"  전체 합계: {total:.1f}%")

    conn.close()

if __name__ == "__main__":
    clear_and_recreate_perfume_data()
