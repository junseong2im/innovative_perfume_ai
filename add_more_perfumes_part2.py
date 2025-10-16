"""
추가 실제 유명 향수 레시피 Part 2
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

# 추가 향수 레시피 Part 2 (30개)
PERFUMES_PART2 = [
    # === ACQUA DI PARMA (추가) ===
    ("Acqua di Parma", "Blu Mediterraneo", 1999, "Unisex", "aromatic",
        {"Bergamot": 10.0, "Orange": 8.0, "Mandarin": 7.0, "Lemon": 5.0},
        {"Cardamom": 12.0, "Myrtle": 10.0, "Jasmine": 8.0},
        {"Cedar": 15.0, "Mastic": 12.0, "Juniper": 8.0, "Amber": 5.0}
    ),

    # === AZZARO ===
    ("Azzaro", "Wanted", 2016, "Men", "spicy",
        {"Lemon": 10.0, "Ginger": 8.0, "Lavender": 7.0, "Mint": 5.0},
        {"Apple": 12.0, "Juniper": 10.0, "Cardamom": 8.0},
        {"Tonka Bean": 15.0, "Amber Wood": 12.0, "Haitian Vetiver": 13.0}
    ),
    ("Azzaro", "Chrome", 1996, "Men", "citrus",
        {"Lemon": 12.0, "Rosemary": 8.0, "Pineapple": 7.0, "Neroli": 5.0, "Bergamot": 3.0},
        {"Jasmine": 12.0, "Cyclamen": 10.0, "Coriander": 8.0},
        {"Musk": 15.0, "Oakmoss": 10.0, "Cedar": 8.0, "Tonka Bean": 7.0}
    ),

    # === BOTTEGA VENETA ===
    ("Bottega Veneta", "Bottega Veneta", 2011, "Women", "leather",
        {"Bergamot": 10.0, "Pink Pepper": 8.0, "Plum": 7.0},
        {"Jasmine Sambac": 15.0, "Leather": 15.0},
        {"Patchouli": 15.0, "Oak": 10.0, "Oakmoss": 10.0, "Musk": 10.0}
    ),

    # === CARTIER ===
    ("Cartier", "Baiser Vole", 2011, "Women", "floral",
        {"Lily": 25.0},
        {"Lily": 35.0},
        {"Lily": 40.0}
    ),

    # === CHLOE (추가) ===
    ("Chloe", "Nomade", 2018, "Women", "floral",
        {"Mirabelle Plum": 10.0, "Bergamot": 8.0, "Lemon": 7.0},
        {"Freesia": 15.0, "Peach": 12.0, "Jasmine": 8.0},
        {"Oakmoss": 12.0, "Amber": 10.0, "White Musk": 8.0, "Sandalwood": 10.0}
    ),
    ("Chloe", "Love Story", 2014, "Women", "floral",
        {"Neroli": 10.0, "Orange Blossom": 8.0, "Stephanotis": 7.0},
        {"Orange Blossom": 18.0, "Stephanotis": 12.0},
        {"Cedar": 15.0, "Amber": 12.0, "Musk": 18.0}
    ),

    # === DAVIDOFF ===
    ("Davidoff", "Cool Water", 1988, "Men", "aromatic",
        {"Lavender": 10.0, "Mint": 8.0, "Green Notes": 7.0, "Rosemary": 5.0},
        {"Geranium": 12.0, "Neroli": 10.0, "Jasmine": 8.0, "Sandalwood": 5.0},
        {"Cedarwood": 15.0, "Musk": 12.0, "Amber": 8.0}
    ),

    # === DIESEL ===
    ("Diesel", "Fuel for Life", 2007, "Men", "woody",
        {"Grapefruit": 10.0, "Anise": 8.0, "Lavender": 7.0},
        {"Heliotrope": 15.0, "Raspberry": 12.0},
        {"Vetiver": 15.0, "Patchouli": 13.0, "Labdanum": 20.0}
    ),

    # === DSQUARED2 ===
    ("Dsquared2", "He Wood", 2007, "Men", "woody",
        {"Violet Leaf": 12.0, "Mandarin": 8.0},
        {"Vetiver": 18.0, "Cedar": 12.0, "Bourbon Pepper": 5.0},
        {"Fir": 15.0, "Amber": 12.0, "White Musk": 18.0}
    ),

    # === ELIZABETH ARDEN ===
    ("Elizabeth Arden", "Red Door", 1989, "Women", "floral",
        {"Red Rose": 10.0, "Ylang-Ylang": 8.0, "Orange Blossom": 7.0, "Plum": 5.0},
        {"Jasmine": 15.0, "Rose": 15.0, "Orchid": 10.0},
        {"Sandalwood": 12.0, "Honey": 10.0, "Vetiver": 8.0}
    ),

    # === ESCADA ===
    ("Escada", "Magnetism", 2003, "Women", "oriental",
        {"Melon": 10.0, "Pineapple": 8.0, "Litchi": 7.0, "Red Berries": 5.0},
        {"Jasmine": 12.0, "Freesia": 10.0, "Heliotrope": 8.0, "Lily": 5.0},
        {"Benzoin": 15.0, "Amber": 12.0, "Musk": 8.0}
    ),

    # === FERRARI ===
    ("Ferrari", "Scuderia Ferrari Red", 2010, "Men", "aromatic",
        {"Apple": 10.0, "Mandarin": 8.0, "Mint": 7.0},
        {"Cinnamon": 12.0, "Cedar": 10.0, "Jasmine": 8.0},
        {"Musk": 15.0, "Sandalwood": 12.0, "Patchouli": 8.0, "Amber": 10.0}
    ),

    # === HUGO BOSS ===
    ("Hugo Boss", "Boss Bottled", 1998, "Men", "woody",
        {"Apple": 10.0, "Plum": 8.0, "Lemon": 7.0, "Oakmoss": 5.0},
        {"Geranium": 12.0, "Cinnamon": 10.0, "Mahogany": 8.0},
        {"Vanilla": 15.0, "Sandalwood": 12.0, "Cedar": 8.0, "Vetiver": 5.0}
    ),
    ("Hugo Boss", "The Scent", 2015, "Men", "leather",
        {"Ginger": 10.0, "Mandarin": 8.0, "Lavender": 7.0},
        {"African Maninka Fruit": 20.0, "Leather": 15.0},
        {"Cedar": 15.0, "Musk": 25.0}
    ),

    # === JEAN PAUL GAULTIER ===
    ("Jean Paul Gaultier", "Le Male", 1995, "Men", "oriental",
        {"Lavender": 10.0, "Mint": 8.0, "Cardamom": 7.0, "Bergamot": 5.0},
        {"Cinnamon": 12.0, "Orange Blossom": 10.0, "Cumin": 8.0},
        {"Vanilla": 15.0, "Tonka Bean": 12.0, "Sandalwood": 8.0, "Cedar": 5.0}
    ),
    ("Jean Paul Gaultier", "Scandal", 2017, "Women", "chypre",
        {"Blood Orange": 12.0, "Mandarin": 8.0},
        {"Honey": 20.0, "Peach": 15.0, "Gardenia": 5.0},
        {"Caramel": 15.0, "Licorice": 10.0, "Patchouli": 15.0}
    ),

    # === JUICY COUTURE ===
    ("Juicy Couture", "Viva La Juicy", 2008, "Women", "gourmand",
        {"Wild Berries": 12.0, "Mandarin": 8.0},
        {"Jasmine": 15.0, "Honeysuckle": 12.0, "Gardenia": 8.0},
        {"Amber": 12.0, "Caramel": 15.0, "Vanilla": 10.0, "Praline": 8.0}
    ),

    # === LACOSTE ===
    ("Lacoste", "L.12.12 Blanc", 2011, "Men", "aromatic",
        {"Grapefruit": 10.0, "Rosemary": 8.0, "Cardamom": 7.0, "Cedar": 5.0},
        {"Ylang-Ylang": 12.0, "Tuberose": 10.0},
        {"Suede": 15.0, "Vetiver": 13.0, "Leather": 20.0}
    ),

    # === LOLITA LEMPICKA ===
    ("Lolita Lempicka", "Lolita Lempicka", 1997, "Women", "oriental",
        {"Ivy": 10.0, "Star Anise": 8.0, "Violet": 7.0},
        {"Iris": 15.0, "Amaryllis": 12.0, "Praline": 8.0},
        {"Tonka Bean": 15.0, "Vanilla": 15.0, "Vetiver": 10.0}
    ),

    # === MANCERA ===
    ("Mancera", "Roses Vanille", 2010, "Unisex", "floral",
        {"Rose": 15.0, "Orange": 10.0},
        {"Rose": 25.0, "White Pepper": 5.0},
        {"Vanilla": 20.0, "Cedar": 10.0, "White Musk": 15.0}
    ),

    # === MONTALE ===
    ("Montale", "Black Aoud", 2006, "Unisex", "woody",
        {"Rose": 12.0, "Mandarin": 8.0},
        {"Patchouli": 20.0},
        {"Agarwood": 25.0, "Musk": 20.0, "Oakmoss": 15.0}
    ),
    ("Montale", "Roses Musk", 2009, "Unisex", "floral",
        {"Rose": 20.0},
        {"Jasmine": 15.0, "Rose": 20.0},
        {"Musk": 25.0, "Amber": 20.0}
    ),

    # === MOSCHINO ===
    ("Moschino", "Toy 2", 2018, "Women", "floral",
        {"Mandarin": 10.0, "Magnolia": 8.0, "Ginger": 7.0},
        {"Jasmine": 15.0, "Peony": 12.0, "Currant Buds": 8.0},
        {"Musk": 15.0, "Amber": 12.0, "Sandalwood": 8.0, "Woody Notes": 5.0}
    ),

    # === MUGLER (추가) ===
    ("Mugler", "Alien", 2005, "Women", "oriental",
        {"Jasmine Sambac": 30.0},
        {"Cashmere Wood": 20.0, "Amber": 15.0},
        {"White Amber": 35.0}
    ),

    # === PALACE BEAUTY GALLERIA ===
    ("Penhaligon's", "Halfeti", 2015, "Unisex", "oriental",
        {"Grapefruit": 10.0, "Lavender": 8.0, "Cardamom": 7.0},
        {"Rose": 18.0, "Jasmine Sambac": 12.0, "Saffron": 5.0},
        {"Oud": 20.0, "Amber": 10.0, "Leather": 5.0, "Vanilla": 5.0}
    ),

    # === SALVATORE FERRAGAMO ===
    ("Salvatore Ferragamo", "Signorina", 2011, "Women", "chypre",
        {"Pink Pepper": 10.0, "Jasmine": 8.0, "Currants": 7.0},
        {"Rose": 15.0, "Peony": 12.0, "Heliotrope": 8.0},
        {"Patchouli": 15.0, "Pannacotta": 10.0, "Musk": 15.0}
    ),

    # === TRUSSARDI ===
    ("Trussardi", "Riflesso", 2016, "Men", "woody",
        {"Bergamot": 10.0, "Cypress": 8.0, "Violet Leaf": 7.0, "Hazelnut": 5.0},
        {"Tonka Bean": 15.0, "Cedar": 12.0, "Iris": 8.0},
        {"Vetiver": 15.0, "Musk": 10.0, "Guaiac Wood": 10.0}
    ),

    # === VALENTINO ===
    ("Valentino", "Valentina", 2011, "Women", "floral",
        {"Bergamot": 10.0, "Strawberry": 8.0, "White Truffle": 7.0},
        {"Jasmine": 15.0, "Tuberose": 12.0, "Orange Blossom": 8.0},
        {"Amber": 15.0, "Cedarwood": 12.0, "Patchouli": 8.0, "White Truffle": 5.0}
    ),
    ("Valentino", "Uomo", 2014, "Men", "woody",
        {"Bergamot": 10.0, "Myrtle": 8.0},
        {"Coffee": 15.0, "Gianduja Cream": 12.0, "Hazelnut": 8.0},
        {"Cedar": 17.0, "Leather": 15.0, "Cocoa": 15.0}
    ),
]

def add_perfumes_part2():
    """Part 2 향수들을 데이터베이스에 추가"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 사용 가능한 성분 목록
    cursor.execute("SELECT name FROM ingredients")
    available_ingredients = {row[0] for row in cursor.fetchall()}

    print(f"사용 가능한 성분: {len(available_ingredients)}개")
    print(f"추가할 향수: {len(PERFUMES_PART2)}개\n")

    added_perfumes = 0
    added_notes = 0
    skipped_ingredients = set()

    for brand, name, year, gender, style, top_notes, heart_notes, base_notes in PERFUMES_PART2:
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

    print(f"=== Part 2 완료 ===")
    print(f"추가된 향수: {added_perfumes}개")
    print(f"추가된 노트: {added_notes}개")
    print(f"스킵된 성분: {len(skipped_ingredients)}개")

    if skipped_ingredients:
        print(f"\n스킵된 성분:")
        for ing in sorted(skipped_ingredients):
            print(f"  - {ing}")

    conn.close()

if __name__ == "__main__":
    add_perfumes_part2()
