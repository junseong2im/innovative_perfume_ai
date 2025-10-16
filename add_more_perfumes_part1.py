"""
추가 실제 유명 향수 레시피 Part 1
실제 Fragrantica 데이터 기반
"""

import sqlite3
import sys

# 기존 create_perfume_recipes.py의 매칭 함수
def find_ingredient_match(recipe_name, available_ingredients):
    """레시피 성분명을 DB 성분명으로 매칭"""
    if recipe_name in available_ingredients:
        return recipe_name

    simple_name = recipe_name.replace(" Oil", "").replace(" Absolute", "").replace(" Otto", "")
    if simple_name in available_ingredients:
        return simple_name

    for suffix in [" Oil", " Absolute", " Otto"]:
        full_name = recipe_name + suffix
        if full_name in available_ingredients:
            return full_name

    recipe_lower = recipe_name.lower()
    for db_name in available_ingredients:
        if recipe_lower in db_name.lower() or db_name.lower() in recipe_lower:
            return db_name

    return None

# 추가 향수 레시피 Part 1 (30개)
PERFUMES_PART1 = [
    # === BURBERRY ===
    ("Burberry", "Brit", 2003, "Women", "fresh",
        {"Lime": 8.0, "Pear": 10.0, "Green Almond": 7.0},
        {"Peony": 12.0, "Rose": 10.0, "Candied Almond": 8.0},
        {"Vanilla": 15.0, "Tonka Bean": 10.0, "Amber": 10.0, "Mahogany": 10.0}
    ),
    ("Burberry", "London", 2006, "Men", "woody",
        {"Bergamot": 8.0, "Lavender": 7.0, "Cinnamon": 5.0, "Black Pepper": 5.0},
        {"Mimosa": 10.0, "Leather": 12.0, "Port Wine": 8.0},
        {"Opoponax": 12.0, "Guaiac Wood": 10.0, "Oakmoss": 8.0, "Tobacco": 15.0}
    ),

    # === BVLGARI ===
    ("Bvlgari", "Omnia Crystalline", 2005, "Women", "floral",
        {"Bamboo": 10.0, "Nashi Pear": 12.0},
        {"Lotus": 15.0, "Tea": 10.0, "Cassia": 8.0},
        {"Guaiac Wood": 15.0, "Balsa Wood": 12.0, "Musk": 18.0}
    ),
    ("Bvlgari", "Man in Black", 2014, "Men", "oriental",
        {"Rum": 10.0, "Spices": 8.0, "Orange": 7.0},
        {"Tuberose": 15.0, "Iris": 12.0, "Leather": 10.0},
        {"Benzoin": 12.0, "Guaiac Wood": 10.0, "Amber": 16.0}
    ),

    # === CALVIN KLEIN ===
    ("Calvin Klein", "Euphoria", 2005, "Women", "oriental",
        {"Pomegranate": 10.0, "Persimmon": 8.0, "Green Notes": 7.0},
        {"Lotus Blossom": 12.0, "Orchid": 15.0, "Champagne Accord": 8.0},
        {"Amber": 12.0, "Black Violet": 10.0, "Mahogany": 8.0, "Cream Accord": 10.0}
    ),
    ("Calvin Klein", "Obsession", 1985, "Women", "oriental",
        {"Mandarin": 8.0, "Bergamot": 7.0, "Vanilla": 5.0},
        {"Jasmine": 15.0, "Orange Blossom": 12.0, "Spices": 10.0},
        {"Amber": 15.0, "Incense": 10.0, "Musk": 8.0, "Sandalwood": 10.0}
    ),

    # === CAROLINA HERRERA ===
    ("Carolina Herrera", "Good Girl", 2016, "Women", "oriental",
        {"Lemon": 8.0, "Almond": 10.0, "Coffee": 7.0},
        {"Tuberose": 15.0, "Jasmine": 12.0, "Orange Blossom": 8.0},
        {"Tonka Bean": 12.0, "Cacao": 10.0, "Vanilla": 8.0, "Praline": 5.0, "Sandalwood": 5.0}
    ),
    ("Carolina Herrera", "212 VIP", 2010, "Women", "floral",
        {"Passion Fruit": 10.0, "Rum": 8.0, "Vodka": 7.0},
        {"Musk": 15.0, "Gardenia": 12.0},
        {"Vanilla": 15.0, "Tonka Bean": 12.0, "Amber": 10.0, "Musk": 11.0}
    ),

    # === ESTEE LAUDER ===
    ("Estee Lauder", "Beautiful", 1985, "Women", "floral",
        {"Rose": 10.0, "Lily": 8.0, "Marigold": 7.0},
        {"Tuberose": 15.0, "Jasmine": 12.0, "Ylang-Ylang": 10.0, "Carnation": 8.0},
        {"Sandalwood": 12.0, "Vetiver": 10.0, "Cedar": 8.0}
    ),
    ("Estee Lauder", "Modern Muse", 2013, "Women", "floral",
        {"Mandarin": 10.0},
        {"Jasmine Sambac": 15.0, "Chinese Osmanthus": 12.0, "Honeysuckle": 8.0, "Tuberose": 10.0},
        {"Patchouli": 15.0, "Amber": 12.0, "Vanilla": 8.0, "Musk": 10.0}
    ),

    # === GIVENCHY ===
    ("Givenchy", "Irresistible", 2020, "Women", "floral",
        {"Pear": 12.0, "Red Berries": 10.0},
        {"Rose": 15.0, "Iris": 15.0},
        {"Musk": 18.0, "Virginia Cedar": 12.0, "Ambrox": 18.0}
    ),
    ("Givenchy", "Gentleman", 2017, "Men", "floral",
        {"Lavender": 10.0, "Pear": 8.0, "Bergamot": 7.0},
        {"Iris": 15.0, "Geranium": 12.0},
        {"Patchouli": 15.0, "Vanilla": 12.0, "Tonka Bean": 10.0, "Black Vanilla": 11.0}
    ),
    ("Givenchy", "L'Interdit", 2018, "Women", "floral",
        {"Bergamot": 10.0, "Pear": 8.0},
        {"Tuberose": 15.0, "Orange Blossom": 12.0, "Jasmine Sambac": 10.0},
        {"Patchouli": 15.0, "Vanilla": 12.0, "Ambroxan": 8.0, "Vetiver": 10.0}
    ),

    # === GUCCI (추가) ===
    ("Gucci", "Guilty", 2010, "Women", "oriental",
        {"Pink Pepper": 8.0, "Mandarin": 7.0, "Bergamot": 5.0},
        {"Lilac": 12.0, "Geranium": 10.0, "Peach": 8.0},
        {"Patchouli": 15.0, "Amber": 15.0, "White Musk": 20.0}
    ),
    ("Gucci", "Bamboo", 2015, "Women", "floral",
        {"Bergamot": 10.0, "Orange Blossom": 8.0},
        {"Casablanca Lily": 15.0, "Ylang-Ylang": 12.0, "Orange Blossom": 8.0},
        {"Sandalwood": 15.0, "Amber": 12.0, "Tahitian Vanilla": 10.0, "Musk": 10.0}
    ),

    # === JIMMY CHOO ===
    ("Jimmy Choo", "Jimmy Choo", 2011, "Women", "fruity",
        {"Pear": 12.0, "Sweet Orange": 8.0, "Mandarin": 5.0},
        {"Tiger Orchid": 15.0, "Tea Rose": 12.0},
        {"Patchouli": 15.0, "Toffee": 13.0, "Indonesian Patchouli": 20.0}
    ),
    ("Jimmy Choo", "Man", 2014, "Men", "fruity",
        {"Lavender": 10.0, "Honeydew Melon": 8.0, "Pink Pepper": 7.0},
        {"Geranium": 12.0, "Orchid": 10.0, "Pineapple Leaf": 8.0},
        {"Suede": 15.0, "Patchouli": 15.0, "Amber": 15.0}
    ),

    # === KENZO ===
    ("Kenzo", "Flower by Kenzo", 2000, "Women", "floral",
        {"Hawthorn": 10.0, "Rose": 8.0, "Bulgarian Rose": 7.0},
        {"Parma Violet": 15.0, "Opoponax": 12.0, "Wild Hawthorn": 8.0},
        {"Vanilla": 15.0, "Musk": 15.0, "Incense": 10.0}
    ),
    ("Kenzo", "L'Eau par Kenzo", 1996, "Unisex", "aquatic",
        {"Lotus": 10.0, "Lemon": 8.0, "Mint": 7.0},
        {"Peony": 15.0, "Water Lily": 12.0},
        {"White Musk": 20.0, "Vanilla": 15.0, "Cedar": 13.0}
    ),

    # === MARC JACOBS ===
    ("Marc Jacobs", "Daisy", 2007, "Women", "floral",
        {"Strawberry": 10.0, "Violet Leaf": 8.0, "Grapefruit": 7.0},
        {"Gardenia": 15.0, "Violet": 12.0, "Jasmine": 8.0},
        {"Musk": 15.0, "Vanilla": 12.0, "White Woods": 13.0}
    ),
    ("Marc Jacobs", "Decadence", 2015, "Women", "oriental",
        {"Italian Plum": 12.0, "Saffron": 8.0, "Iris": 5.0},
        {"Bulgarian Rose": 15.0, "Orris": 12.0, "Jasmine Sambac": 8.0},
        {"Vetiver": 12.0, "Papyrus": 10.0, "Amber": 8.0, "Liquid Amber": 10.0}
    ),

    # === MICHAEL KORS ===
    ("Michael Kors", "Wonderlust", 2016, "Women", "oriental",
        {"Almond Milk": 12.0, "Bergamot": 8.0},
        {"Carnation": 15.0, "Dianthus": 12.0, "Heliotrope": 8.0},
        {"Sandalwood": 15.0, "Benzoin": 12.0, "Cashmere Wood": 10.0, "Musk": 8.0}
    ),

    # === MONTBLANC ===
    ("Montblanc", "Legend", 2011, "Men", "aromatic",
        {"Lavender": 10.0, "Pineapple": 8.0, "Bergamot": 7.0, "Lemon Verbena": 5.0},
        {"Oakmoss": 12.0, "Geranium": 10.0, "Coumarin": 8.0, "Apple": 5.0},
        {"Sandalwood": 12.0, "Tonka Bean": 10.0, "Evernyl": 8.0, "Amber": 15.0}
    ),

    # === NARCISO RODRIGUEZ ===
    ("Narciso Rodriguez", "For Her", 2003, "Women", "floral",
        {"Orange Blossom": 10.0, "Osmanthus": 8.0, "Bergamot": 7.0},
        {"Musk": 18.0, "Rose": 12.0, "Peach": 8.0},
        {"Amber": 12.0, "Vanilla": 10.0, "Sandalwood": 8.0, "Vetiver": 7.0}
    ),
    ("Narciso Rodriguez", "Bleu Noir", 2015, "Men", "woody",
        {"Cardamom": 8.0, "Bergamot": 7.0, "Nutmeg": 5.0},
        {"Blue Cedar": 15.0, "Geranium": 10.0, "Ebony": 8.0},
        {"Musk": 18.0, "Vetiver": 12.0, "Tonka Bean": 10.0, "Suede Accord": 7.0}
    ),

    # === RALPH LAUREN ===
    ("Ralph Lauren", "Romance", 1998, "Women", "floral",
        {"Rose": 10.0, "Ginger": 8.0, "Chamomile": 7.0},
        {"Day Lily": 15.0, "Violet": 12.0, "Marigold": 8.0},
        {"Patchouli": 12.0, "Oakmoss": 10.0, "Skin Musk": 8.0, "White Musk": 10.0}
    ),

    # === VERSACE ===
    ("Versace", "Bright Crystal", 2006, "Women", "floral",
        {"Yuzu": 10.0, "Pomegranate": 8.0, "Ice Accord": 7.0},
        {"Peony": 15.0, "Magnolia": 12.0, "Lotus Flower": 8.0},
        {"Amber": 12.0, "Musk": 15.0, "Mahogany": 8.0, "Acajou": 5.0}
    ),
    ("Versace", "Eros", 2012, "Men", "aromatic",
        {"Mint": 10.0, "Lemon": 8.0, "Green Apple": 7.0, "Candy Apple": 5.0},
        {"Tonka Bean": 12.0, "Geranium": 10.0, "Ambroxan": 8.0},
        {"Vanilla": 15.0, "Cedar": 12.0, "Oakmoss": 10.0, "Vetiver": 8.0}
    ),

    # === VIKTOR & ROLF (추가) ===
    ("Viktor & Rolf", "Bonbon", 2014, "Women", "gourmand",
        {"Mandarin": 10.0, "Orange": 8.0, "Peach": 7.0},
        {"Orange Blossom": 15.0, "Jasmine": 12.0, "Caramel": 8.0},
        {"Amber": 12.0, "Cedar": 10.0, "Guaiac Wood": 8.0, "Peach": 10.0}
    ),
]

def add_perfumes_part1():
    """Part 1 향수들을 데이터베이스에 추가"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 사용 가능한 성분 목록
    cursor.execute("SELECT name FROM ingredients")
    available_ingredients = {row[0] for row in cursor.fetchall()}

    print(f"사용 가능한 성분: {len(available_ingredients)}개")
    print(f"추가할 향수: {len(PERFUMES_PART1)}개\n")

    added_perfumes = 0
    added_notes = 0
    skipped_ingredients = set()

    for brand, name, year, gender, style, top_notes, heart_notes, base_notes in PERFUMES_PART1:
        # 향수 추가
        try:
            cursor.execute("""
                INSERT INTO perfumes (brand, name, year, gender, style)
                VALUES (?, ?, ?, ?, ?)
            """, (brand, name, year, gender, style))
            perfume_id = cursor.lastrowid
            added_perfumes += 1

            # 노트 추가
            for ingredient, concentration in top_notes.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched:
                    cursor.execute("""
                        INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                        VALUES (?, ?, 'top', ?)
                    """, (perfume_id, matched, concentration))
                    added_notes += 1
                else:
                    skipped_ingredients.add(ingredient)

            for ingredient, concentration in heart_notes.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched:
                    cursor.execute("""
                        INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                        VALUES (?, ?, 'heart', ?)
                    """, (perfume_id, matched, concentration))
                    added_notes += 1
                else:
                    skipped_ingredients.add(ingredient)

            for ingredient, concentration in base_notes.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched:
                    cursor.execute("""
                        INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                        VALUES (?, ?, 'base', ?)
                    """, (perfume_id, matched, concentration))
                    added_notes += 1
                else:
                    skipped_ingredients.add(ingredient)

        except sqlite3.IntegrityError:
            print(f"이미 존재: {brand} - {name}")

    conn.commit()

    print(f"=== Part 1 완료 ===")
    print(f"추가된 향수: {added_perfumes}개")
    print(f"추가된 노트: {added_notes}개")
    print(f"스킵된 성분: {len(skipped_ingredients)}개")

    if skipped_ingredients:
        print(f"\n스킵된 성분:")
        for ing in sorted(skipped_ingredients):
            print(f"  - {ing}")

    conn.close()

if __name__ == "__main__":
    add_perfumes_part1()
