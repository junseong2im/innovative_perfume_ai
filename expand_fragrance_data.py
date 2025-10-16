"""
Fragrantica 스타일의 향수 성분 데이터베이스 확장 스크립트
실제 향수 산업에서 사용되는 성분들을 추가합니다.
"""

import sqlite3

# Fragrantica에서 자주 사용되는 향수 성분 데이터
# 카테고리: top, heart, base
# 휘발성: 0.0(낮음) ~ 1.0(높음)
# 가격($/kg): 실제 향료 시장 가격 참고
# IFRA 제한: 0.0 ~ 1.0 (향수에 사용 가능한 최대 비율)
# 향 강도: 0.0(약함) ~ 1.0(강함)

FRAGRANCE_INGREDIENTS = [
    # ===== TOP NOTES (시트러스 & 프레시) =====
    ("Bergamot", "top", 0.90, 85.0, 2.0, 0.80),
    ("Lemon", "top", 0.95, 65.0, 3.0, 0.90),
    ("Orange", "top", 0.85, 50.0, 2.5, 0.70),
    ("Grapefruit", "top", 0.88, 70.0, 2.0, 0.75),
    ("Mandarin", "top", 0.87, 60.0, 2.5, 0.65),
    ("Lime", "top", 0.92, 55.0, 2.0, 0.85),
    ("Yuzu", "top", 0.91, 150.0, 2.0, 0.70),
    ("Neroli", "top", 0.75, 4500.0, 1.5, 0.85),
    ("Petitgrain", "top", 0.78, 45.0, 2.0, 0.60),

    # TOP NOTES (그린 & 허브)
    ("Mint", "top", 0.93, 30.0, 1.5, 0.95),
    ("Basil", "top", 0.89, 40.0, 1.8, 0.80),
    ("Rosemary", "top", 0.84, 35.0, 2.0, 0.75),
    ("Lavender", "top", 0.80, 90.0, 5.0, 0.70),
    ("Thyme", "top", 0.86, 50.0, 1.5, 0.85),
    ("Sage", "top", 0.82, 60.0, 1.8, 0.75),
    ("Green Tea", "top", 0.88, 120.0, 2.0, 0.65),

    # TOP NOTES (과일 & 베리)
    ("Apple", "top", 0.85, 80.0, 2.5, 0.60),
    ("Pear", "top", 0.83, 75.0, 2.5, 0.55),
    ("Peach", "top", 0.81, 100.0, 2.0, 0.65),
    ("Blackcurrant", "top", 0.84, 250.0, 2.0, 0.75),
    ("Raspberry", "top", 0.86, 200.0, 2.0, 0.70),
    ("Strawberry", "top", 0.87, 180.0, 2.0, 0.65),
    ("Pineapple", "top", 0.88, 90.0, 2.5, 0.70),
    ("Melon", "top", 0.89, 85.0, 2.5, 0.60),

    # TOP NOTES (스파이시)
    ("Pink Pepper", "top", 0.76, 180.0, 1.5, 0.85),
    ("Black Pepper", "top", 0.78, 120.0, 1.5, 0.90),
    ("Cardamom", "top", 0.80, 200.0, 1.8, 0.80),
    ("Ginger", "top", 0.79, 110.0, 2.0, 0.85),

    # ===== HEART NOTES (플로럴) =====
    ("Rose", "heart", 0.50, 5000.0, 0.5, 0.95),
    ("Jasmine", "heart", 0.45, 8000.0, 0.3, 1.00),
    ("Lavender", "heart", 0.55, 90.0, 5.0, 0.70),
    ("Geranium", "heart", 0.52, 75.0, 2.0, 0.75),
    ("Ylang-Ylang", "heart", 0.48, 350.0, 0.8, 0.95),
    ("Tuberose", "heart", 0.43, 1500.0, 0.4, 0.98),
    ("Iris", "heart", 0.40, 12000.0, 1.0, 0.85),
    ("Violet", "heart", 0.47, 800.0, 1.5, 0.80),
    ("Lily of the Valley", "heart", 0.53, 600.0, 1.0, 0.85),
    ("Magnolia", "heart", 0.49, 700.0, 1.2, 0.80),
    ("Peony", "heart", 0.51, 650.0, 1.5, 0.75),
    ("Freesia", "heart", 0.54, 450.0, 1.8, 0.70),
    ("Gardenia", "heart", 0.46, 900.0, 0.8, 0.90),
    ("Orange Blossom", "heart", 0.56, 550.0, 1.5, 0.85),
    ("Mimosa", "heart", 0.52, 1200.0, 1.2, 0.80),

    # HEART NOTES (스파이시 & 우디)
    ("Nutmeg", "heart", 0.58, 180.0, 1.5, 0.80),
    ("Cinnamon", "heart", 0.60, 90.0, 0.5, 0.95),
    ("Clove", "heart", 0.57, 110.0, 0.5, 0.98),
    ("Star Anise", "heart", 0.59, 95.0, 1.0, 0.85),

    # HEART NOTES (프루티 & 구르망)
    ("Apricot", "heart", 0.62, 150.0, 2.0, 0.65),
    ("Plum", "heart", 0.60, 130.0, 2.0, 0.70),
    ("Fig", "heart", 0.58, 200.0, 2.0, 0.75),
    ("Coconut", "heart", 0.55, 180.0, 2.5, 0.80),

    # ===== BASE NOTES (우디) =====
    ("Sandalwood", "base", 0.20, 2500.0, 1.0, 0.85),
    ("Cedarwood", "base", 0.25, 40.0, 3.0, 0.75),
    ("Vetiver", "base", 0.22, 250.0, 2.0, 0.90),
    ("Patchouli", "base", 0.18, 120.0, 2.5, 0.95),
    ("Oud", "base", 0.10, 80000.0, 0.5, 1.00),
    ("Agarwood", "base", 0.12, 70000.0, 0.5, 0.98),
    ("Guaiac Wood", "base", 0.23, 180.0, 2.0, 0.85),
    ("Cypress", "base", 0.28, 65.0, 2.5, 0.70),
    ("Pine", "base", 0.30, 45.0, 3.0, 0.75),

    # BASE NOTES (발삼 & 레진)
    ("Vanilla", "base", 0.15, 1200.0, 2.0, 0.90),
    ("Tonka Bean", "base", 0.17, 450.0, 1.5, 0.88),
    ("Benzoin", "base", 0.14, 180.0, 2.0, 0.85),
    ("Labdanum", "base", 0.16, 300.0, 1.8, 0.90),
    ("Amber", "base", 0.13, 800.0, 1.5, 0.92),
    ("Myrrh", "base", 0.15, 250.0, 1.5, 0.88),
    ("Frankincense", "base", 0.19, 200.0, 2.0, 0.82),
    ("Balsam", "base", 0.18, 150.0, 2.0, 0.80),

    # BASE NOTES (머스크 & 애니말릭)
    ("Musk", "base", 0.08, 1500.0, 1.0, 0.95),
    ("White Musk", "base", 0.10, 1200.0, 1.2, 0.85),
    ("Ambergris", "base", 0.05, 45000.0, 0.3, 1.00),
    ("Castoreum", "base", 0.12, 8000.0, 0.5, 0.95),
    ("Civet", "base", 0.09, 12000.0, 0.2, 0.98),

    # BASE NOTES (구르망 & 스위트)
    ("Caramel", "base", 0.20, 300.0, 2.0, 0.85),
    ("Chocolate", "base", 0.22, 400.0, 2.0, 0.90),
    ("Coffee", "base", 0.24, 180.0, 2.5, 0.88),
    ("Honey", "base", 0.21, 350.0, 2.0, 0.83),
    ("Praline", "base", 0.23, 320.0, 2.0, 0.87),

    # BASE NOTES (레더 & 스모키)
    ("Leather", "base", 0.16, 600.0, 1.5, 0.92),
    ("Birch Tar", "base", 0.19, 450.0, 1.8, 0.95),
    ("Tobacco", "base", 0.17, 380.0, 2.0, 0.90),
    ("Incense", "base", 0.14, 280.0, 2.0, 0.88),
    ("Smoke", "base", 0.15, 220.0, 2.0, 0.85),

    # BASE NOTES (어시 & 아쿠아틱)
    ("Moss", "base", 0.26, 190.0, 2.5, 0.75),
    ("Oakmoss", "base", 0.24, 350.0, 0.1, 0.90),
    ("Seaweed", "base", 0.28, 280.0, 2.0, 0.70),
    ("Driftwood", "base", 0.27, 200.0, 2.5, 0.75),
]

def expand_database():
    """데이터베이스에 새로운 향수 성분 추가"""

    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # 기존 성분 확인
    cursor.execute("SELECT name FROM ingredients")
    existing = {row[0] for row in cursor.fetchall()}

    print(f"기존 성분 수: {len(existing)}")
    print(f"추가할 성분 수: {len(FRAGRANCE_INGREDIENTS)}")

    # 새로운 성분만 추가
    added = 0
    updated = 0

    for name, category, volatility, price, ifra_limit, odor_strength in FRAGRANCE_INGREDIENTS:
        if name in existing:
            # 기존 성분 업데이트
            cursor.execute("""
                UPDATE ingredients
                SET category = ?, volatility = ?, price_per_kg = ?,
                    ifra_limit = ?, odor_strength = ?
                WHERE name = ?
            """, (category, volatility, price, ifra_limit, odor_strength, name))
            updated += 1
        else:
            # 새 성분 추가
            cursor.execute("""
                INSERT INTO ingredients (name, category, volatility, price_per_kg, ifra_limit, odor_strength)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, category, volatility, price, ifra_limit, odor_strength))
            added += 1

    conn.commit()

    # 최종 성분 수 확인
    cursor.execute("SELECT COUNT(*) FROM ingredients")
    total = cursor.fetchone()[0]

    # 카테고리별 개수
    cursor.execute("SELECT category, COUNT(*) FROM ingredients GROUP BY category")
    category_counts = dict(cursor.fetchall())

    print(f"\n✅ 데이터베이스 확장 완료!")
    print(f"   - 추가된 성분: {added}개")
    print(f"   - 업데이트된 성분: {updated}개")
    print(f"   - 전체 성분: {total}개")
    print(f"\n카테고리별 성분:")
    print(f"   - Top notes: {category_counts.get('top', 0)}개")
    print(f"   - Heart notes: {category_counts.get('heart', 0)}개")
    print(f"   - Base notes: {category_counts.get('base', 0)}개")

    # 샘플 데이터 출력
    print(f"\n샘플 성분 (각 카테고리별 5개):")
    for category in ['top', 'heart', 'base']:
        cursor.execute("""
            SELECT name, volatility, price_per_kg, odor_strength
            FROM ingredients
            WHERE category = ?
            ORDER BY RANDOM()
            LIMIT 5
        """, (category,))
        print(f"\n{category.upper()} NOTES:")
        for row in cursor.fetchall():
            print(f"  - {row[0]}: 휘발성={row[1]:.2f}, 가격=${row[2]:.2f}/kg, 강도={row[3]:.2f}")

    conn.close()

if __name__ == "__main__":
    expand_database()
