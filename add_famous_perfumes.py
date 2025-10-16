"""
Fragrantica 스타일 - 유명 향수들의 노트 구성 데이터 추가
실제 향수 브랜드들의 대표작들을 데이터베이스에 추가합니다.
"""

import sqlite3

# 유명 향수들의 노트 구성 (Fragrantica 참고)
# 형식: (브랜드, 향수명, 출시년도, 성별, [탑노트], [미들노트], [베이스노트])
FAMOUS_PERFUMES = [
    # 시트러스 & 프레시
    (
        "Chanel", "Chance Eau Fraiche", 2007, "Women",
        ["Lemon", "Lime", "Grapefruit"],
        ["Jasmine", "Violet", "White Musk"],
        ["Amber", "Patchouli", "Vetiver"]
    ),
    (
        "Dior", "Sauvage", 2015, "Men",
        ["Bergamot", "Black Pepper"],
        ["Lavender", "Pink Pepper", "Geranium"],
        ["Ambergris", "Patchouli", "Vetiver"]
    ),
    (
        "Acqua di Parma", "Colonia", 1916, "Unisex",
        ["Lemon", "Bergamot", "Orange"],
        ["Lavender", "Rosemary", "Rose"],
        ["Vetiver", "Sandalwood", "Patchouli"]
    ),
    (
        "Dolce & Gabbana", "Light Blue", 2001, "Women",
        ["Lemon", "Apple", "Grapefruit"],
        ["Jasmine", "Rose", "Bamboo"],
        ["Amber", "Musk", "Cedarwood"]
    ),

    # 플로럴
    (
        "Chanel", "No 5", 1921, "Women",
        ["Neroli", "Ylang-Ylang", "Bergamot"],
        ["Rose", "Jasmine", "Lily of the Valley"],
        ["Vanilla", "Sandalwood", "Vetiver"]
    ),
    (
        "Dior", "J'adore", 1999, "Women",
        ["Mandarin", "Champaca", "Ivy"],
        ["Rose", "Jasmine", "Orchid", "Violet"],
        ["Blackberry", "Musk", "Amaranth"]
    ),
    (
        "Gucci", "Bloom", 2017, "Women",
        ["Pomegranate", "Green Notes"],
        ["Jasmine", "Tuberose", "Rangoon Creeper"],
        ["Orris Root", "Sandalwood"]
    ),
    (
        "Yves Saint Laurent", "Black Opium", 2014, "Women",
        ["Pink Pepper", "Orange Blossom", "Pear"],
        ["Coffee", "Jasmine", "Orange Blossom"],
        ["Vanilla", "Patchouli", "Cedarwood"]
    ),
    (
        "Viktor & Rolf", "Flowerbomb", 2005, "Women",
        ["Bergamot", "Green Tea", "Osmanthus"],
        ["Rose", "Jasmine", "Orchid", "Freesia"],
        ["Patchouli", "Musk", "Vanilla"]
    ),

    # 우디 & 오리엔탈
    (
        "Tom Ford", "Oud Wood", 2007, "Unisex",
        ["Rosewood", "Cardamom", "Chinese Pepper"],
        ["Oud", "Sandalwood", "Vetiver"],
        ["Tonka Bean", "Vanilla", "Amber"]
    ),
    (
        "Chanel", "Bleu de Chanel", 2010, "Men",
        ["Lemon", "Pink Pepper", "Mint"],
        ["Ginger", "Nutmeg", "Jasmine"],
        ["Sandalwood", "Cedarwood", "Vetiver", "Patchouli"]
    ),
    (
        "Dior", "Homme Intense", 2011, "Men",
        ["Lavender", "Sage", "Bergamot"],
        ["Iris", "Ambrette", "Pear"],
        ["Vetiver", "Cedarwood", "Leather"]
    ),
    (
        "Creed", "Aventus", 2010, "Men",
        ["Pineapple", "Bergamot", "Apple", "Blackcurrant"],
        ["Pink Pepper", "Birch", "Patchouli", "Jasmine"],
        ["Musk", "Oakmoss", "Ambergris", "Vanilla"]
    ),
    (
        "Giorgio Armani", "Acqua di Gio", 1996, "Men",
        ["Lime", "Lemon", "Bergamot", "Jasmine", "Orange"],
        ["Calone", "Freesia", "Hyacinth", "Violet", "Rosemary"],
        ["White Musk", "Cedarwood", "Oakmoss", "Patchouli"]
    ),

    # 구르망 & 스위트
    (
        "Thierry Mugler", "Angel", 1992, "Women",
        ["Melon", "Coconut", "Mandarin", "Cassia"],
        ["Honey", "Apricot", "Blackberry", "Plum"],
        ["Patchouli", "Chocolate", "Vanilla", "Caramel", "Tonka Bean"]
    ),
    (
        "Prada", "Candy", 2011, "Women",
        ["Caramel"],
        ["Benzoin", "Musk", "Vanilla"],
        ["Benzoin", "Musk"]
    ),
    (
        "Lancome", "La Vie Est Belle", 2012, "Women",
        ["Blackcurrant", "Pear"],
        ["Iris", "Jasmine", "Orange Blossom"],
        ["Praline", "Vanilla", "Patchouli", "Tonka Bean"]
    ),

    # 레더 & 스파이시
    (
        "Tom Ford", "Tobacco Vanille", 2007, "Unisex",
        ["Tobacco Leaf", "Spicy Notes"],
        ["Tobacco Blossom", "Vanilla", "Cocoa"],
        ["Tonka Bean", "Dried Fruits", "Sweet Wood Sap"]
    ),
    (
        "Hermes", "Terre d'Hermes", 2006, "Men",
        ["Orange", "Grapefruit"],
        ["Pink Pepper", "Pelargonium"],
        ["Vetiver", "Cedarwood", "Patchouli", "Benzoin"]
    ),
    (
        "Guerlain", "Shalimar", 1925, "Women",
        ["Lemon", "Bergamot"],
        ["Iris", "Jasmine", "Rose"],
        ["Vanilla", "Incense", "Tonka Bean", "Opoponax"]
    ),

    # 아쿠아틱 & 오존
    (
        "Calvin Klein", "CK One", 1994, "Unisex",
        ["Lemon", "Bergamot", "Cardamom", "Pineapple"],
        ["Violet", "Rose", "Freesia", "Jasmine", "Nutmeg"],
        ["Musk", "Cedarwood", "Sandalwood", "Oakmoss", "Amber"]
    ),
    (
        "Issey Miyake", "L'Eau d'Issey", 1992, "Women",
        ["Lotus", "Freesia", "Cyclamen", "Rose Water", "Melon"],
        ["Peony", "Lily", "Carnation"],
        ["Musk", "Cedarwood", "Sandalwood", "Osmanthus"]
    ),

    # 그린 & 프레시
    (
        "Jo Malone", "Wood Sage & Sea Salt", 2014, "Unisex",
        ["Ambrette Seeds"],
        ["Sea Salt", "Sage"],
        ["Red Algae", "Grapefruit"]
    ),
    (
        "Hermes", "Un Jardin sur le Nil", 2005, "Unisex",
        ["Grapefruit", "Green Mango", "Tomato Leaf"],
        ["Lotus", "Orange Blossom", "Bulrush", "Peony"],
        ["Iris", "Incense", "Sycamore Wood", "Musk"]
    ),

    # 파우더리 & 알데히드
    (
        "Prada", "Infusion d'Iris", 2007, "Women",
        ["Mandarin", "Orange", "Galbanum", "Neroli"],
        ["Iris", "Tuberose", "Ylang-Ylang"],
        ["Frankincense", "Benzoin", "Vetiver", "Cedarwood"]
    ),
    (
        "Chloe", "Chloe Eau de Parfum", 2008, "Women",
        ["Peony", "Freesia", "Lychee"],
        ["Rose", "Magnolia", "Lily of the Valley"],
        ["Cedarwood", "Amber"]
    ),

    # 니치 향수
    (
        "Byredo", "Gypsy Water", 2008, "Unisex",
        ["Bergamot", "Lemon", "Pepper", "Juniper"],
        ["Incense", "Pine Needle", "Orris"],
        ["Vanilla", "Sandalwood", "Amber"]
    ),
    (
        "Le Labo", "Santal 33", 2011, "Unisex",
        ["Cardamom", "Iris", "Violet"],
        ["Papyrus", "Ambrox", "Cedarwood"],
        ["Leather", "Sandalwood", "Musk"]
    ),
    (
        "Maison Francis Kurkdjian", "Baccarat Rouge 540", 2015, "Unisex",
        ["Saffron", "Jasmine"],
        ["Amberwood", "Ambergris"],
        ["Fir Resin", "Cedarwood"]
    ),
]

def create_perfumes_table():
    """향수 테이블 생성"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 향수 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS perfumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            name TEXT NOT NULL,
            year INTEGER,
            gender TEXT,
            style TEXT,
            UNIQUE(brand, name)
        )
    """)

    # 향수-성분 관계 테이블 생성 (다대다)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS perfume_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            perfume_id INTEGER NOT NULL,
            ingredient_name TEXT NOT NULL,
            note_position TEXT NOT NULL,
            concentration REAL DEFAULT 10.0,
            FOREIGN KEY (perfume_id) REFERENCES perfumes(id),
            FOREIGN KEY (ingredient_name) REFERENCES ingredients(name),
            UNIQUE(perfume_id, ingredient_name, note_position)
        )
    """)

    conn.commit()
    conn.close()
    print("✅ 향수 테이블 생성 완료")

def determine_style(top_notes, middle_notes, base_notes):
    """노트 구성으로 향수 스타일 판단"""
    all_notes = top_notes + middle_notes + base_notes

    # 시트러스 체크
    citrus = ["Lemon", "Bergamot", "Orange", "Grapefruit", "Lime", "Mandarin"]
    citrus_count = sum(1 for note in all_notes if note in citrus)

    # 플로럴 체크
    floral = ["Rose", "Jasmine", "Violet", "Iris", "Tuberose", "Ylang-Ylang",
              "Lily of the Valley", "Magnolia", "Peony", "Freesia", "Orchid"]
    floral_count = sum(1 for note in all_notes if note in floral)

    # 우디 체크
    woody = ["Sandalwood", "Cedarwood", "Vetiver", "Patchouli", "Oud",
             "Oakmoss", "Pine", "Cypress"]
    woody_count = sum(1 for note in all_notes if note in woody)

    # 구르망 체크
    gourmand = ["Vanilla", "Chocolate", "Caramel", "Tonka Bean", "Coffee",
                "Honey", "Praline"]
    gourmand_count = sum(1 for note in all_notes if note in gourmand)

    # 스파이시 체크
    spicy = ["Pink Pepper", "Black Pepper", "Cardamom", "Ginger", "Cinnamon",
             "Nutmeg", "Clove"]
    spicy_count = sum(1 for note in all_notes if note in spicy)

    # 가장 많은 카테고리 선택
    max_count = max(citrus_count, floral_count, woody_count, gourmand_count, spicy_count)

    if max_count == citrus_count and citrus_count >= 2:
        return "fresh"
    elif max_count == floral_count and floral_count >= 2:
        return "floral"
    elif max_count == woody_count and woody_count >= 2:
        return "woody"
    elif max_count == gourmand_count and gourmand_count >= 2:
        return "gourmand"
    elif max_count == spicy_count and spicy_count >= 2:
        return "spicy"
    else:
        return "oriental"

def add_famous_perfumes():
    """유명 향수 데이터 추가"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 성분 이름 목록 가져오기
    cursor.execute("SELECT name FROM ingredients")
    available_ingredients = {row[0] for row in cursor.fetchall()}

    added_perfumes = 0
    added_notes = 0
    skipped_notes = 0

    for brand, name, year, gender, top_notes, middle_notes, base_notes in FAMOUS_PERFUMES:
        # 스타일 자동 판단
        style = determine_style(top_notes, middle_notes, base_notes)

        # 향수 추가 (이미 있으면 무시)
        try:
            cursor.execute("""
                INSERT INTO perfumes (brand, name, year, gender, style)
                VALUES (?, ?, ?, ?, ?)
            """, (brand, name, year, gender, style))
            perfume_id = cursor.lastrowid
            added_perfumes += 1

            # 노트 추가
            # Top notes (30% 분배)
            if top_notes:
                conc_per_note = 30.0 / len(top_notes)
                for note in top_notes:
                    if note in available_ingredients:
                        cursor.execute("""
                            INSERT OR IGNORE INTO perfume_notes
                            (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'top', ?)
                        """, (perfume_id, note, conc_per_note))
                        added_notes += 1
                    else:
                        skipped_notes += 1

            # Middle notes (40% 분배)
            if middle_notes:
                conc_per_note = 40.0 / len(middle_notes)
                for note in middle_notes:
                    if note in available_ingredients:
                        cursor.execute("""
                            INSERT OR IGNORE INTO perfume_notes
                            (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'heart', ?)
                        """, (perfume_id, note, conc_per_note))
                        added_notes += 1
                    else:
                        skipped_notes += 1

            # Base notes (30% 분배)
            if base_notes:
                conc_per_note = 30.0 / len(base_notes)
                for note in base_notes:
                    if note in available_ingredients:
                        cursor.execute("""
                            INSERT OR IGNORE INTO perfume_notes
                            (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, 'base', ?)
                        """, (perfume_id, note, conc_per_note))
                        added_notes += 1
                    else:
                        skipped_notes += 1

        except sqlite3.IntegrityError:
            # 이미 존재하는 향수
            pass

    conn.commit()

    # 통계 출력
    cursor.execute("SELECT COUNT(*) FROM perfumes")
    total_perfumes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM perfume_notes")
    total_notes = cursor.fetchone()[0]

    cursor.execute("SELECT style, COUNT(*) FROM perfumes GROUP BY style")
    style_counts = dict(cursor.fetchall())

    print(f"\n✅ 유명 향수 데이터 추가 완료!")
    print(f"   - 추가된 향수: {added_perfumes}개")
    print(f"   - 전체 향수: {total_perfumes}개")
    print(f"   - 추가된 노트: {added_notes}개")
    print(f"   - 전체 노트: {total_notes}개")
    print(f"   - 스킵된 노트: {skipped_notes}개 (성분 DB에 없음)")

    print(f"\n스타일별 향수:")
    for style, count in sorted(style_counts.items()):
        print(f"   - {style}: {count}개")

    # 샘플 향수 출력
    print(f"\n샘플 향수 (5개):")
    cursor.execute("""
        SELECT p.brand, p.name, p.year, p.gender, p.style,
               GROUP_CONCAT(DISTINCT CASE WHEN pn.note_position = 'top' THEN pn.ingredient_name END) as top,
               GROUP_CONCAT(DISTINCT CASE WHEN pn.note_position = 'heart' THEN pn.ingredient_name END) as heart,
               GROUP_CONCAT(DISTINCT CASE WHEN pn.note_position = 'base' THEN pn.ingredient_name END) as base
        FROM perfumes p
        LEFT JOIN perfume_notes pn ON p.id = pn.perfume_id
        GROUP BY p.id
        ORDER BY RANDOM()
        LIMIT 5
    """)

    for row in cursor.fetchall():
        brand, name, year, gender, style, top, heart, base = row
        print(f"\n{brand} - {name} ({year}, {gender}, {style})")
        if top:
            print(f"  Top: {top}")
        if heart:
            print(f"  Heart: {heart}")
        if base:
            print(f"  Base: {base}")

    conn.close()

if __name__ == "__main__":
    print("📦 향수 데이터베이스 확장 시작...\n")
    create_perfumes_table()
    add_famous_perfumes()
    print("\n✅ 모든 작업 완료!")
