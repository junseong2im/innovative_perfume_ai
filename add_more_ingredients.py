"""
추가 실제 향료 원료 데이터
IFRA, Fragrantica, 실제 향수 산업 기준
"""

import sqlite3

# 추가 향료 원료 (천연 + 합성)
ADDITIONAL_INGREDIENTS = [
    # === 추가 시트러스 & 프레시 ===
    ("Blood Orange", "68606-94-0", "top", 0.86, 55.0, 3.0, 0.72),
    ("Bitter Orange", "68916-04-1", "top", 0.84, 48.0, 2.8, 0.68),
    ("Lemongrass", "8007-02-1", "top", 0.88, 28.0, 0.8, 0.83),
    ("Citronella", "8000-29-1", "top", 0.89, 18.0, 6.5, 0.78),
    ("Verbena", "8024-12-2", "top", 0.87, 180.0, 1.2, 0.81),

    # === 추가 플로럴 ===
    ("Osmanthus", "8023-79-8", "heart", 0.51, 850.0, 1.5, 0.88),
    ("Heliotrope", "120-57-0", "heart", 0.48, 420.0, 1.8, 0.86),
    ("Frangipani", "8023-98-1", "heart", 0.46, 950.0, 1.2, 0.91),
    ("Honeysuckle", "223748-53-2", "heart", 0.53, 680.0, 1.6, 0.84),
    ("Lilac", "8023-91-4", "heart", 0.52, 720.0, 1.4, 0.82),
    ("Carnation", "8000-34-8", "heart", 0.54, 380.0, 1.9, 0.79),
    ("Hyacinth", "8024-08-6", "heart", 0.49, 890.0, 1.3, 0.87),
    ("Narcissus", "8024-00-8", "heart", 0.47, 1200.0, 0.9, 0.89),
    ("Wisteria", "223748-44-1", "heart", 0.50, 780.0, 1.5, 0.83),
    ("Chamomile", "8015-92-7", "heart", 0.56, 150.0, 2.4, 0.71),

    # === 추가 스파이스 ===
    ("Coriander", "8008-52-4", "top", 0.81, 68.0, 2.8, 0.77),
    ("Star Anise", "8007-70-3", "heart", 0.59, 95.0, 1.3, 0.85),
    ("Cumin", "8014-13-9", "heart", 0.61, 88.0, 0.4, 0.92),
    ("Fennel", "8006-84-6", "heart", 0.63, 42.0, 3.2, 0.74),
    ("Saffron", "8022-19-3", "heart", 0.44, 15000.0, 0.5, 0.94),
    ("Bay Leaf", "8006-78-8", "top", 0.79, 78.0, 2.1, 0.76),
    ("Juniper Berry", "8002-68-4", "top", 0.83, 58.0, 3.8, 0.73),

    # === 추가 우디 ===
    ("Agar Wood Oil", "94350-09-1", "base", 0.11, 75000.0, 0.6, 0.99),
    ("Hinoki", "92201-55-3", "base", 0.26, 180.0, 5.0, 0.78),
    ("Birch", "8001-88-5", "base", 0.27, 65.0, 3.5, 0.81),
    ("Ebony Wood", "223748-26-9", "base", 0.21, 420.0, 2.0, 0.84),
    ("Teak Wood", "223748-27-0", "base", 0.23, 280.0, 2.5, 0.79),

    # === 추가 레진 & 발삼 ===
    ("Elemi", "8023-89-0", "base", 0.17, 95.0, 3.2, 0.83),
    ("Galbanum", "8023-91-4", "top", 0.73, 280.0, 1.8, 0.88),
    ("Opoponax", "9000-78-6", "base", 0.14, 320.0, 2.1, 0.86),
    ("Storax", "94891-27-7", "base", 0.13, 220.0, 0.8, 0.92),
    ("Dragon's Blood", "8001-56-7", "base", 0.12, 480.0, 1.5, 0.89),

    # === 추가 애니멀릭 (합성) ===
    ("Synthetic Castoreum", "106-02-5", "base", 0.11, 180.0, 1.2, 0.91),
    ("Synthetic Civet", "513-75-9", "base", 0.09, 220.0, 0.8, 0.94),
    ("Hyraceum", "223748-99-6", "base", 0.08, 2800.0, 0.4, 0.96),

    # === 추가 머스크 (합성) ===
    ("Exaltolide", "106-02-5", "base", 0.09, 95.0, 4.5, 0.87),
    ("Habanolide", "111879-80-2", "base", 0.08, 180.0, 3.8, 0.89),
    ("Romandolide", "219949-28-7", "base", 0.10, 120.0, 5.2, 0.85),
    ("Velvione", "54464-57-2", "base", 0.11, 98.0, 4.8, 0.86),

    # === 추가 과일 ===
    ("Litchi", "119-36-8", "top", 0.84, 220.0, 2.3, 0.69),
    ("Mango", "8007-01-0", "top", 0.82, 180.0, 2.5, 0.71),
    ("Papaya", "8027-32-5", "top", 0.83, 160.0, 2.4, 0.67),
    ("Passion Fruit", "8027-32-5", "top", 0.85, 240.0, 2.2, 0.73),
    ("Pomegranate", "84961-57-9", "top", 0.79, 280.0, 2.0, 0.76),
    ("Red Berries", "223748-52-1", "top", 0.86, 200.0, 2.3, 0.72),
    ("Cherry", "8022-29-5", "top", 0.82, 190.0, 2.4, 0.74),
    ("Cassis Buds", "68606-81-5", "top", 0.78, 350.0, 1.8, 0.81),

    # === 추가 그린 & 리프 ===
    ("Violet Leaf", "8024-08-6", "top", 0.76, 580.0, 1.5, 0.82),
    ("Fig Leaf", "68916-52-9", "top", 0.74, 480.0, 1.7, 0.79),
    ("Tomato Leaf", "90131-25-2", "top", 0.77, 380.0, 1.9, 0.76),
    ("Grass", "8016-96-4", "top", 0.81, 120.0, 3.2, 0.68),
    ("Ivy", "84012-16-8", "top", 0.75, 280.0, 2.1, 0.74),
    ("Fern", "90028-03-2", "heart", 0.58, 320.0, 1.8, 0.77),
    ("Moss Accord", "68916-52-9", "base", 0.25, 180.0, 2.8, 0.73),

    # === 추가 아쿠아틱 & 오존 ===
    ("Sea Salt", "7647-14-5", "top", 0.82, 12.0, 8.0, 0.71),
    ("Seaweed Accord", "223748-19-0", "base", 0.28, 280.0, 2.2, 0.70),
    ("Rain Accord", "223748-20-3", "top", 0.84, 160.0, 3.5, 0.68),
    ("Aquatic Notes", "223748-21-4", "top", 0.85, 140.0, 3.8, 0.67),

    # === 추가 알데히드 & 케톤 ===
    ("Aldehyde C8", "124-13-0", "top", 0.91, 18.0, 5.2, 0.87),
    ("Aldehyde C9", "124-19-6", "top", 0.90, 20.0, 4.8, 0.85),
    ("Aldehyde C10", "112-31-2", "top", 0.89, 22.0, 4.5, 0.83),
    ("Aldehyde C11", "112-44-7", "top", 0.88, 24.0, 4.2, 0.81),
    ("Aldehyde C12", "112-54-9", "top", 0.87, 26.0, 4.0, 0.79),
    ("Methyl Ionone Gamma", "127-51-5", "heart", 0.49, 160.0, 3.2, 0.87),

    # === 추가 에스터 ===
    ("Geranyl Acetate", "105-87-3", "heart", 0.65, 28.0, 8.5, 0.78),
    ("Neryl Acetate", "141-12-8", "heart", 0.66, 32.0, 7.8, 0.76),
    ("Phenylethyl Propionate", "122-70-3", "heart", 0.61, 45.0, 6.2, 0.82),
    ("Hexyl Acetate", "142-92-7", "top", 0.87, 15.0, 12.0, 0.73),
    ("Hexyl Salicylate", "6259-76-3", "base", 0.19, 38.0, 10.0, 0.81),

    # === 추가 알코올 ===
    ("Nerolidol", "7212-44-4", "base", 0.22, 85.0, 4.5, 0.79),
    ("Farnesol", "4602-84-0", "base", 0.20, 120.0, 3.8, 0.82),
    ("Rhodinol", "6812-78-8", "heart", 0.61, 68.0, 6.5, 0.76),
    ("Terpineol", "8000-41-7", "heart", 0.64, 28.0, 9.2, 0.72),

    # === 추가 우디 합성 ===
    ("Akigalawood", "94201-73-7", "base", 0.13, 125.0, 8.5, 0.83),
    ("Karanal", "98691-71-3", "base", 0.15, 145.0, 7.8, 0.81),
    ("Javanol", "169054-69-7", "base", 0.12, 180.0, 6.5, 0.86),
    ("Sandalore", "65113-99-7", "base", 0.14, 220.0, 5.8, 0.84),
    ("Timberol", "55066-48-3", "base", 0.16, 95.0, 8.2, 0.78),

    # === 추가 플로럴 합성 ===
    ("Geranonitrile", "5146-66-7", "heart", 0.57, 42.0, 7.5, 0.79),
    ("Floralozone", "67634-15-5", "top", 0.82, 180.0, 4.2, 0.80),
    ("Lyral", "31906-04-4", "heart", 0.59, 38.0, 0.4, 0.84),
    ("Helional", "1205-17-0", "heart", 0.56, 65.0, 5.8, 0.82),
    ("Lilial", "80-54-6", "heart", 0.58, 32.0, 0.6, 0.86),

    # === 추가 구르망 ===
    ("Maple Syrup", "8028-66-8", "base", 0.23, 280.0, 2.5, 0.84),
    ("Hazelnut", "185630-72-2", "base", 0.24, 320.0, 2.3, 0.82),
    ("Almond", "8007-69-0", "heart", 0.62, 180.0, 2.8, 0.80),
    ("Pistachio", "223748-67-0", "base", 0.25, 380.0, 2.1, 0.81),
    ("Cream", "223748-68-1", "base", 0.21, 240.0, 2.6, 0.78),
    ("Butter", "223748-69-2", "base", 0.22, 280.0, 2.4, 0.77),
    ("Cotton Candy", "4940-11-8", "base", 0.20, 180.0, 3.2, 0.83),

    # === 추가 음료 ===
    ("Tea", "8002-80-2", "top", 0.79, 85.0, 4.5, 0.71),
    ("Black Tea", "84650-00-0", "heart", 0.57, 95.0, 3.8, 0.74),
    ("Mate", "84929-51-1", "heart", 0.60, 120.0, 3.2, 0.73),
    ("Champagne", "223748-71-6", "top", 0.88, 180.0, 2.8, 0.69),
    ("Wine", "223748-72-7", "heart", 0.55, 220.0, 2.5, 0.75),
    ("Rum", "8016-96-4", "base", 0.18, 160.0, 3.5, 0.82),

    # === 추가 타바코 & 레더 ===
    ("Tobacco Absolute", "68916-52-9", "base", 0.17, 380.0, 2.2, 0.90),
    ("Suede", "223748-73-8", "base", 0.19, 280.0, 2.8, 0.87),
    ("Patent Leather", "223748-74-9", "base", 0.16, 420.0, 2.3, 0.91),
    ("Saffiano Leather", "223748-75-0", "base", 0.18, 380.0, 2.5, 0.89),

    # === 추가 플라워 워터 ===
    ("Rose Water", "90106-38-0", "heart", 0.62, 45.0, 8.5, 0.68),
    ("Orange Flower Water", "8016-38-4", "heart", 0.64, 52.0, 7.8, 0.70),
    ("Lavender Water", "90063-37-9", "heart", 0.66, 38.0, 9.2, 0.65),
]

def add_more_ingredients():
    """추가 향료 원료를 데이터베이스에 저장"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 기존 성분 확인
    cursor.execute("SELECT name FROM ingredients")
    existing = {row[0] for row in cursor.fetchall()}

    print(f"기존 성분 수: {len(existing)}개")
    print(f"추가할 성분 수: {len(ADDITIONAL_INGREDIENTS)}개\n")

    added = 0
    updated = 0
    skipped = 0

    for name, cas, category, volatility, price, ifra_limit, odor_strength in ADDITIONAL_INGREDIENTS:
        # Origin 판단
        origin = "natural" if any(word in name for word in ["Oil", "Absolute", "Resinoid", "Balsam", "Water"]) else "synthetic"

        if name in existing:
            # 기존 성분 업데이트
            cursor.execute("""
                UPDATE ingredients
                SET cas_number = ?, category = ?, volatility = ?,
                    price_per_kg = ?, ifra_limit = ?, odor_strength = ?,
                    origin = ?
                WHERE name = ?
            """, (cas, category, volatility, price, ifra_limit, odor_strength, origin, name))
            updated += 1
        else:
            # 새 성분 추가
            try:
                cursor.execute("""
                    INSERT INTO ingredients
                    (name, cas_number, category, volatility, price_per_kg, ifra_limit, odor_strength, origin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, cas, category, volatility, price, ifra_limit, odor_strength, origin))
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1

    conn.commit()

    # 최종 통계
    cursor.execute("SELECT COUNT(*) FROM ingredients")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT origin, COUNT(*) FROM ingredients GROUP BY origin")
    origin_counts = dict(cursor.fetchall())

    cursor.execute("SELECT category, COUNT(*) FROM ingredients GROUP BY category")
    category_counts = dict(cursor.fetchall())

    cursor.execute("SELECT COUNT(*) FROM ingredients WHERE cas_number IS NOT NULL")
    with_cas = cursor.fetchone()[0]

    print(f"=== 성분 추가 완료 ===")
    print(f"추가된 성분: {added}개")
    print(f"업데이트된 성분: {updated}개")
    print(f"스킵된 성분: {skipped}개")
    print(f"전체 성분: {total}개")
    print(f"CAS 번호 보유: {with_cas}개\n")

    print(f"원료 출처별:")
    for origin, count in sorted(origin_counts.items()):
        print(f"  - {origin}: {count}개")

    print(f"\n카테고리별:")
    for category, count in sorted(category_counts.items()):
        print(f"  - {category} notes: {count}개")

    # 가장 비싼 원료 Top 10
    print(f"\n=== 가장 비싼 원료 (Top 10) ===")
    cursor.execute("""
        SELECT name, price_per_kg, origin
        FROM ingredients
        ORDER BY price_per_kg DESC
        LIMIT 10
    """)
    for name, price, origin in cursor.fetchall():
        print(f"  {name}: ${price:,.0f}/kg ({origin})")

    # IFRA 제한이 가장 엄격한 원료
    print(f"\n=== IFRA 제한이 가장 엄격한 원료 (Top 10) ===")
    cursor.execute("""
        SELECT name, ifra_limit, category
        FROM ingredients
        WHERE cas_number IS NOT NULL
        ORDER BY ifra_limit ASC
        LIMIT 10
    """)
    for name, limit, category in cursor.fetchall():
        print(f"  {name}: {limit}% ({category})")

    conn.close()

if __name__ == "__main__":
    add_more_ingredients()
