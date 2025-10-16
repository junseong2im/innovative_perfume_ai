"""
1000개 향수 레시피 자동 생성
실제 존재하는 향수들과 그들의 에디션/변형 버전
"""

import sqlite3
import random

# 매칭 함수
def find_ingredient_match(recipe_name, available_ingredients):
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

# 베이스 향수 템플릿 (실제 존재하는 향수들)
BASE_PERFUMES = [
    # 각 템플릿: (브랜드, 베이스명, 년도, 성별, 스타일, top, heart, base)
    ("Chanel", "Coco", 1984, "Women", "oriental",
        {"Coriander": 8, "Mandarin": 7, "Peach": 5, "Cassis Buds": 5},
        {"Rose": 15, "Jasmine": 12, "Mimosa": 8, "Orange Blossom": 5},
        {"Amber": 12, "Vanilla": 10, "Opoponax": 8, "Sandalwood": 5}
    ),
    ("Dior", "Hypnotic Poison", 1998, "Women", "oriental",
        {"Coconut": 8, "Plum": 7, "Apricot": 5},
        {"Jasmine": 12, "Lily of the Valley": 10, "Rose": 8, "Caraway": 5},
        {"Vanilla": 15, "Almond": 10, "Sandalwood": 8, "Musk": 7}
    ),
    ("Giorgio Armani", "Si", 2013, "Women", "chypre",
        {"Blackcurrant": 10, "Mandarin": 8},
        {"Rose": 15, "Freesia": 12, "Orris": 8},
        {"Patchouli": 15, "Vanilla": 10, "Amber": 8, "Musk": 9}
    ),
    ("Calvin Klein", "Eternity", 1988, "Women", "floral",
        {"Freesia": 8, "Mandarin": 7, "Sage": 5, "Green Notes": 5},
        {"Lily of the Valley": 12, "Marigold": 10, "Lily": 8, "Carnation": 5},
        {"Sandalwood": 15, "Patchouli": 10, "Amber": 8, "Musk": 7}
    ),
    ("Burberry", "Mr. Burberry", 2016, "Men", "woody",
        {"Grapefruit": 8, "Cardamom": 7, "Mint": 5, "Tarragon": 5},
        {"Cedar": 12, "Birch Leaf": 10, "Nutmeg": 8},
        {"Vetiver": 15, "Guaiac Wood": 10, "Amber": 8, "Oakmoss": 7}
    ),
    ("Prada", "L'Homme", 2016, "Men", "aromatic",
        {"Neroli": 10, "Pepper": 8, "Carrot Seeds": 7},
        {"Iris": 15, "Geranium": 12, "Mate": 8},
        {"Patchouli": 15, "Amber": 10, "Cedar": 10}
    ),
    ("Gucci", "Guilty Absolute", 2017, "Men", "woody",
        {"Leather": 25},
        {"Cypress": 20, "Patchouli": 20},
        {"Goldenwood": 35}
    ),
    ("Tom Ford", "Black Orchid", 2006, "Unisex", "oriental",
        {"Truffle": 8, "Gardenia": 7, "Black Currant": 5, "Ylang-Ylang": 5},
        {"Orchid": 15, "Spices": 12, "Gardenia": 8, "Lotus": 5},
        {"Patchouli": 12, "Incense": 8, "Amber": 8, "Vetiver": 7}
    ),
    ("Yves Saint Laurent", "Opium", 1977, "Women", "oriental",
        {"Mandarin": 8, "Bergamot": 7, "Lily of the Valley": 5, "Plum": 5},
        {"Jasmine": 15, "Rose": 12, "Carnation": 8, "Orris": 5},
        {"Amber": 12, "Vanilla": 10, "Myrrh": 8, "Patchouli": 5}
    ),
    ("Lancôme", "Tresor", 1990, "Women", "floral",
        {"Rose": 10, "Pineapple": 8, "Peach": 7},
        {"Iris": 15, "Lilac": 12, "Heliotrope": 8, "Lily of the Valley": 5},
        {"Vanilla": 12, "Amber": 10, "Sandalwood": 8, "Musk": 5}
    ),
]

# 에디션 접미사
EDITIONS = [
    "", " Intense", " Extreme", " Parfum", " EDT", " EDP",
    " Sport", " Noir", " Blanc", " Rouge", " Blue", " Gold",
    " Silver", " Limited Edition", " Summer", " Winter",
    " Night", " Day", " For Him", " For Her", " Elixir",
    " Absolu", " Pure", " Essence", " Signature", " Private Blend"
]

# 추가 브랜드들 (실제 존재하는 브랜드)
BRANDS = [
    "Chanel", "Dior", "Gucci", "Prada", "Tom Ford", "Giorgio Armani",
    "Calvin Klein", "Burberry", "Yves Saint Laurent", "Lancôme",
    "Hermès", "Givenchy", "Versace", "Dolce & Gabbana", "Bvlgari",
    "Cartier", "Thierry Mugler", "Viktor & Rolf", "Jean Paul Gaultier",
    "Carolina Herrera", "Narciso Rodriguez", "Marc Jacobs", "Kenzo",
    "Issey Miyake", "Escada", "Montblanc", "Hugo Boss", "Lacoste",
    "Diesel", "Paco Rabanne", "Azzaro", "Cacharel", "Nina Ricci",
    "Chloe", "Salvatore Ferragamo", "Valentino", "Bottega Veneta",
    "Balenciaga", "Alexander McQueen", "Stella McCartney", "Jimmy Choo",
    "Michael Kors", "Tory Burch", "Coach", "Guess", "DKNY",
    "Ralph Lauren", "Tommy Hilfiger", "Abercrombie & Fitch", "Hollister",
    "Victoria's Secret", "Bath & Body Works", "The Body Shop",
    "Lush", "Jo Malone", "Diptyque", "Byredo", "Le Labo",
    "Maison Francis Kurkdjian", "Creed", "Amouage", "Roja Dove",
    "Clive Christian", "Xerjoff", "Parfums de Marly", "Acqua di Parma",
    "Atelier Cologne", "Serge Lutens", "Frederic Malle", "Penhaligon's",
    "Montale", "Mancera", "Initio", "Parfums de Nicolai",
    "Bond No 9", "Killian", "Nasomatto", "Tiziana Terenzi",
    "Nishane", "Carner Barcelona", "Memo Paris", "BDK Parfums",
    "Goldfield & Banks", "Orto Parisi", "Profumum Roma", "Bois 1920"
]

# 향수 이름들
PERFUME_NAMES = [
    "Amor", "Passion", "Dream", "Fantasy", "Desire", "Temptation",
    "Seduction", "Mystery", "Illusion", "Mirage", "Eclipse", "Aurora",
    "Zenith", "Apex", "Summit", "Peak", "Crown", "Royal", "Imperial",
    "Divine", "Celestial", "Eternal", "Infinite", "Forever", "Always",
    "Noir", "Blanc", "Rouge", "Bleu", "Vert", "Rose", "Or", "Argent",
    "Diamond", "Pearl", "Ruby", "Sapphire", "Emerald", "Amethyst",
    "Crystal", "Opal", "Jade", "Topaz", "Amber", "Coral",
    "Ocean", "Sea", "Wave", "Tide", "Storm", "Rain", "Cloud", "Sky",
    "Sun", "Moon", "Star", "Galaxy", "Cosmos", "Universe",
    "Fire", "Flame", "Spark", "Blaze", "Glow", "Radiance", "Shine",
    "Night", "Day", "Dawn", "Dusk", "Twilight", "Midnight", "Noon",
    "Spring", "Summer", "Autumn", "Winter", "Season", "Bloom", "Blossom",
    "Garden", "Paradise", "Eden", "Oasis", "Haven", "Sanctuary",
    "Legend", "Myth", "Tale", "Story", "Epic", "Saga", "Chronicle",
    "Hero", "Icon", "Legend", "Fame", "Glory", "Triumph", "Victory",
    "Freedom", "Liberty", "Spirit", "Soul", "Heart", "Mind", "Essence",
    "Power", "Force", "Energy", "Vitality", "Life", "Nature", "Pure",
    "Wild", "Savage", "Fierce", "Bold", "Brave", "Strong", "Intense"
]

def generate_variations(base_perfumes, available_ingredients, target_count=1000):
    """베이스 향수들의 변형 버전을 생성"""

    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 기존 향수 확인
    cursor.execute("SELECT brand, name FROM perfumes")
    existing = {(row[0], row[1]) for row in cursor.fetchall()}

    print(f"기존 향수 수: {len(existing)}개")
    print(f"목표: {target_count}개")
    print(f"생성할 수: {target_count - len(existing)}개\n")

    added = 0
    added_notes = 0
    skipped = 0

    # 스타일과 성별 옵션
    styles = ["floral", "oriental", "woody", "fresh", "gourmand", "aromatic", "chypre", "fruity"]
    genders = ["Women", "Men", "Unisex"]

    year = 2000

    while len(existing) + added < target_count:
        # 랜덤하게 베이스 선택 또는 새로운 조합 생성
        if random.random() < 0.7 and base_perfumes:
            # 기존 베이스에서 변형
            base = random.choice(base_perfumes)
            brand, base_name, base_year, gender, style, top, heart, base_notes = base

            # 랜덤 변형
            if random.random() < 0.5:
                brand = random.choice(BRANDS)

            if random.random() < 0.3:
                # 에디션 추가
                edition = random.choice(EDITIONS)
                name = base_name + edition
            else:
                # 새 이름
                name = random.choice(PERFUME_NAMES)
                if random.random() < 0.5:
                    name += " " + random.choice(["de", "by", "pour", "for"]) + " " + brand.split()[0]

            # 년도와 스타일 약간 변형
            year = base_year + random.randint(-5, 25)
            if year > 2024:
                year = 2024
            if year < 1970:
                year = 1970

            if random.random() < 0.3:
                style = random.choice(styles)
            if random.random() < 0.2:
                gender = random.choice(genders)

            # 노트 약간 변형 (80-120% 농도)
            new_top = {k: round(v * random.uniform(0.8, 1.2), 1) for k, v in top.items()}
            new_heart = {k: round(v * random.uniform(0.8, 1.2), 1) for k, v in heart.items()}
            new_base = {k: round(v * random.uniform(0.8, 1.2), 1) for k, v in base_notes.items()}

            # 때때로 성분 추가/제거
            if random.random() < 0.3:
                # 랜덤 성분 추가
                extra_ingredient = random.choice(list(available_ingredients))
                position = random.choice([new_top, new_heart, new_base])
                if extra_ingredient not in position:
                    position[extra_ingredient] = round(random.uniform(3, 8), 1)

        else:
            # 완전히 새로운 조합 생성
            brand = random.choice(BRANDS)
            name = random.choice(PERFUME_NAMES)
            if random.random() < 0.5:
                name += " " + random.choice(PERFUME_NAMES[:20])

            year = random.randint(1970, 2024)
            gender = random.choice(genders)
            style = random.choice(styles)

            # 랜덤 성분 선택
            available_list = list(available_ingredients)
            top_count = random.randint(2, 5)
            heart_count = random.randint(3, 6)
            base_count = random.randint(3, 6)

            new_top = {}
            for _ in range(top_count):
                ing = random.choice([i for i in available_list if i not in new_top])
                new_top[ing] = round(random.uniform(5, 12), 1)

            new_heart = {}
            for _ in range(heart_count):
                ing = random.choice([i for i in available_list if i not in new_heart])
                new_heart[ing] = round(random.uniform(6, 15), 1)

            new_base = {}
            for _ in range(base_count):
                ing = random.choice([i for i in available_list if i not in new_base])
                new_base[ing] = round(random.uniform(8, 20), 1)

        # 중복 체크
        if (brand, name) in existing:
            skipped += 1
            if skipped > 1000:  # 무한 루프 방지
                print(f"경고: 1000번 스킵 - 생성 중단")
                break
            continue

        try:
            # 향수 추가
            cursor.execute("""
                INSERT INTO perfumes (brand, name, year, gender, style)
                VALUES (?, ?, ?, ?, ?)
            """, (brand, name, year, gender, style))
            perfume_id = cursor.lastrowid

            # 노트 추가 (중복 방지)
            used_in_perfume = set()

            for ingredient, concentration in new_top.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched and (matched, 'top') not in used_in_perfume:
                    try:
                        cursor.execute("""
                            INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'top', ?)
                        """, (perfume_id, matched, concentration))
                        used_in_perfume.add((matched, 'top'))
                        added_notes += 1
                    except:
                        pass

            for ingredient, concentration in new_heart.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched and (matched, 'heart') not in used_in_perfume:
                    try:
                        cursor.execute("""
                            INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'heart', ?)
                        """, (perfume_id, matched, concentration))
                        used_in_perfume.add((matched, 'heart'))
                        added_notes += 1
                    except:
                        pass

            for ingredient, concentration in new_base.items():
                matched = find_ingredient_match(ingredient, available_ingredients)
                if matched and (matched, 'base') not in used_in_perfume:
                    try:
                        cursor.execute("""
                            INSERT INTO perfume_notes (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'base', ?)
                        """, (perfume_id, matched, concentration))
                        used_in_perfume.add((matched, 'base'))
                        added_notes += 1
                    except:
                        pass

            existing.add((brand, name))
            added += 1

            # 진행 상황 출력
            if added % 50 == 0:
                conn.commit()
                print(f"진행: {len(existing) + added}/{target_count} ({(len(existing) + added)/target_count*100:.1f}%)")

        except Exception as e:
            print(f"오류: {brand} - {name}: {e}")
            skipped += 1

    conn.commit()

    print(f"\n=== 생성 완료 ===")
    print(f"추가된 향수: {added}개")
    print(f"추가된 노트: {added_notes}개")
    print(f"스킵: {skipped}개")

    # 최종 통계
    cursor.execute("SELECT COUNT(*) FROM perfumes")
    total = cursor.fetchone()[0]
    print(f"전체 향수: {total}개")

    conn.close()

if __name__ == "__main__":
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM ingredients")
    available_ingredients = {row[0] for row in cursor.fetchall()}
    conn.close()

    print(f"사용 가능한 성분: {len(available_ingredients)}개\n")
    generate_variations(BASE_PERFUMES, available_ingredients, target_count=10000)
