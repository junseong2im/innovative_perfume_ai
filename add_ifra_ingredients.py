"""
IFRA (International Fragrance Association) 공식 향료 성분 데이터 추가
IFRA Transparency List 및 Standards를 기반으로 한 향료 성분 정보
"""

import sqlite3

# IFRA Standards 기반 향료 성분 데이터
# 형식: (이름, CAS번호, 카테고리, 휘발성, 가격$/kg, IFRA제한%, 향강도)
IFRA_INGREDIENTS = [
    # === NATURAL ESSENTIAL OILS (천연 에센셜 오일) ===

    # Citrus Oils (시트러스 오일)
    ("Bergamot Oil", "8007-75-8", "top", 0.90, 85.0, 0.4, 0.80),
    ("Lemon Oil", "8008-56-8", "top", 0.95, 65.0, 2.0, 0.90),
    ("Sweet Orange Oil", "8008-57-9", "top", 0.85, 50.0, 10.0, 0.70),
    ("Grapefruit Oil", "8016-20-4", "top", 0.88, 70.0, 4.0, 0.75),
    ("Mandarin Oil", "8008-31-9", "top", 0.87, 60.0, 3.0, 0.65),
    ("Lime Oil", "8008-26-2", "top", 0.92, 55.0, 0.7, 0.85),
    ("Petitgrain Oil", "8014-17-3", "top", 0.78, 45.0, 5.0, 0.60),

    # Floral Absolutes (플로럴 앱솔루트)
    ("Rose Otto", "8007-01-0", "heart", 0.50, 5000.0, 2.0, 0.95),
    ("Jasmine Absolute", "8022-96-6", "heart", 0.45, 8000.0, 0.7, 1.00),
    ("Ylang-Ylang Oil", "8006-81-3", "heart", 0.48, 350.0, 2.6, 0.95),
    ("Neroli Oil", "8016-38-4", "top", 0.75, 4500.0, 1.8, 0.85),
    ("Tuberose Absolute", "8024-05-3", "heart", 0.43, 1500.0, 0.5, 0.98),
    ("Lavender Oil", "8000-28-0", "heart", 0.55, 90.0, 13.3, 0.70),
    ("Geranium Oil", "8000-46-2", "heart", 0.52, 75.0, 4.3, 0.75),
    ("Orange Blossom Absolute", "8016-38-4", "heart", 0.56, 550.0, 1.8, 0.85),

    # Herbal & Green (허브 & 그린)
    ("Peppermint Oil", "8006-90-4", "top", 0.93, 30.0, 4.5, 0.95),
    ("Spearmint Oil", "8008-79-5", "top", 0.91, 35.0, 1.6, 0.85),
    ("Rosemary Oil", "8000-25-7", "top", 0.84, 35.0, 11.2, 0.75),
    ("Basil Oil", "8015-73-4", "top", 0.89, 40.0, 0.03, 0.80),
    ("Sage Oil", "8022-56-8", "top", 0.82, 60.0, 0.05, 0.75),
    ("Eucalyptus Oil", "8000-48-4", "top", 0.90, 20.0, 16.2, 0.85),
    ("Tea Tree Oil", "68647-73-4", "top", 0.88, 25.0, 7.0, 0.80),

    # Woody & Resinous (우디 & 레진)
    ("Sandalwood Oil", "8006-87-9", "base", 0.20, 2500.0, 5.0, 0.85),
    ("Cedarwood Oil Virginia", "8000-27-9", "base", 0.25, 40.0, 14.0, 0.75),
    ("Cedarwood Oil Atlas", "92201-55-3", "base", 0.24, 45.0, 12.0, 0.73),
    ("Vetiver Oil", "8016-96-4", "base", 0.22, 250.0, 15.0, 0.90),
    ("Patchouli Oil", "8014-09-3", "base", 0.18, 120.0, 8.0, 0.95),
    ("Frankincense Oil", "8016-36-2", "base", 0.19, 200.0, 8.7, 0.82),
    ("Myrrh Oil", "8016-37-3", "base", 0.15, 250.0, 7.8, 0.88),
    ("Pine Oil", "8002-09-3", "base", 0.30, 45.0, 5.6, 0.75),
    ("Cypress Oil", "8013-86-3", "base", 0.28, 65.0, 12.5, 0.70),

    # Spice Oils (스파이스 오일)
    ("Black Pepper Oil", "8006-82-4", "top", 0.78, 120.0, 7.0, 0.90),
    ("Pink Pepper Oil", "68917-52-2", "top", 0.76, 180.0, 1.6, 0.85),
    ("Cardamom Oil", "8000-66-6", "top", 0.80, 200.0, 4.9, 0.80),
    ("Ginger Oil", "8007-08-7", "top", 0.79, 110.0, 3.1, 0.85),
    ("Cinnamon Bark Oil", "8015-91-6", "heart", 0.60, 90.0, 0.01, 0.95),
    ("Clove Bud Oil", "8000-34-8", "heart", 0.57, 110.0, 0.05, 0.98),
    ("Nutmeg Oil", "8008-45-5", "heart", 0.58, 180.0, 0.2, 0.80),

    # === SYNTHETIC AROMA CHEMICALS (합성 향료) ===

    # Aldehydes (알데히드류)
    ("Citral", "5392-40-5", "top", 0.85, 35.0, 0.6, 0.90),
    ("Citronellal", "106-23-0", "top", 0.87, 28.0, 3.7, 0.85),
    ("Benzaldehyde", "100-52-7", "heart", 0.75, 15.0, 1.0, 0.80),
    ("Anisaldehyde", "123-11-5", "heart", 0.68, 45.0, 0.9, 0.85),
    ("Vanillin", "121-33-5", "base", 0.15, 25.0, 14.0, 0.90),
    ("Ethyl Vanillin", "121-32-4", "base", 0.14, 35.0, 15.0, 0.92),

    # Alcohols (알코올류)
    ("Linalool", "78-70-6", "heart", 0.65, 18.0, 10.9, 0.75),
    ("Citronellol", "106-22-9", "heart", 0.60, 22.0, 8.4, 0.80),
    ("Geraniol", "106-24-1", "heart", 0.62, 25.0, 2.1, 0.85),
    ("Phenylethyl Alcohol", "60-12-8", "heart", 0.55, 35.0, 13.6, 0.90),
    ("Benzyl Alcohol", "100-51-6", "heart", 0.58, 12.0, 50.0, 0.60),
    ("Alpha-Terpineol", "98-55-5", "heart", 0.63, 16.0, 14.3, 0.70),
    ("Menthol", "2216-51-5", "top", 0.70, 40.0, 19.0, 0.95),

    # Esters (에스터류)
    ("Linalyl Acetate", "115-95-7", "heart", 0.68, 20.0, 13.9, 0.75),
    ("Benzyl Acetate", "140-11-4", "heart", 0.64, 14.0, 17.0, 0.80),
    ("Ethyl Acetate", "141-78-6", "top", 0.95, 5.0, 100.0, 0.50),
    ("Methyl Anthranilate", "134-20-3", "heart", 0.60, 80.0, 0.09, 0.85),
    ("Isoamyl Acetate", "123-92-2", "top", 0.88, 8.0, 4.2, 0.70),

    # Ketones (케톤류)
    ("Alpha-Ionone", "127-41-3", "heart", 0.52, 180.0, 2.3, 0.90),
    ("Beta-Ionone", "14901-07-6", "heart", 0.50, 200.0, 2.6, 0.92),
    ("Methyl Ionone", "127-51-5", "heart", 0.48, 150.0, 3.5, 0.88),
    ("Calone", "28940-11-6", "top", 0.75, 320.0, 2.0, 0.85),
    ("Raspberry Ketone", "5471-51-2", "heart", 0.55, 400.0, 1.5, 0.80),

    # Musks (머스크류)
    ("Galaxolide", "1222-05-5", "base", 0.08, 45.0, 17.8, 0.90),
    ("Tonalide", "21145-77-7", "base", 0.09, 50.0, 16.0, 0.88),
    ("Muscone", "541-91-3", "base", 0.06, 1200.0, 2.0, 0.95),
    ("Ambrettolide", "123-69-3", "base", 0.10, 800.0, 3.0, 0.85),

    # Woody Synthetics (우디 합성)
    ("Iso E Super", "54464-57-2", "base", 0.12, 85.0, 21.4, 0.80),
    ("Cedramber", "67874-81-1", "base", 0.14, 95.0, 15.0, 0.82),
    ("Cashmeran", "33704-61-9", "base", 0.16, 120.0, 11.0, 0.85),
    ("Amber Xtreme", "65113-99-7", "base", 0.11, 110.0, 13.0, 0.88),

    # Lactones & Coumarins (락톤 & 쿠마린)
    ("Coumarin", "91-64-5", "base", 0.20, 25.0, 1.7, 0.85),
    ("Gamma-Decalactone", "706-14-9", "heart", 0.45, 180.0, 0.8, 0.90),
    ("Gamma-Nonalactone", "104-61-0", "heart", 0.48, 160.0, 0.9, 0.88),
    ("Tonka Bean Absolute", "8024-04-2", "base", 0.17, 450.0, 5.0, 0.88),

    # Balsamic (발삼)
    ("Benzoin Resinoid", "9000-05-9", "base", 0.14, 180.0, 5.0, 0.85),
    ("Peru Balsam", "8007-00-9", "base", 0.16, 300.0, 0.4, 0.90),
    ("Tolu Balsam", "9000-64-0", "base", 0.18, 250.0, 1.0, 0.88),
    ("Styrax Absolute", "8046-19-3", "base", 0.13, 220.0, 0.6, 0.92),

    # Animalic (애니멀릭)
    ("Civet Absolute", "68916-26-7", "base", 0.09, 12000.0, 0.05, 0.98),
    ("Castoreum Absolute", "8023-83-4", "base", 0.12, 8000.0, 0.1, 0.95),
    ("Ambergris", "8038-65-1", "base", 0.05, 45000.0, 0.5, 1.00),

    # Fresh & Aquatic (프레시 & 아쿠아틱)
    ("Dihydromyrcenol", "18479-58-8", "top", 0.80, 18.0, 43.0, 0.70),
    ("Hedione", "24851-98-7", "heart", 0.58, 45.0, 50.0, 0.75),
    ("Floralozone", "67634-15-5", "top", 0.82, 180.0, 4.0, 0.80),
    ("Ozonic Aldehyde", "68039-49-6", "top", 0.85, 160.0, 3.0, 0.75),

    # Gourmand (구르망)
    ("Maltol", "118-71-8", "base", 0.22, 40.0, 5.0, 0.85),
    ("Ethyl Maltol", "4940-11-8", "base", 0.21, 50.0, 6.0, 0.90),
    ("Furaneol", "3658-77-3", "heart", 0.50, 250.0, 1.0, 0.88),
    ("Caramel Furanone", "28664-35-9", "base", 0.24, 300.0, 1.5, 0.85),
]

def update_with_cas_numbers():
    """기존 성분에 CAS 번호 추가 및 새로운 IFRA 성분 추가"""
    conn = sqlite3.connect('data/fragrance_stable.db')
    cursor = conn.cursor()

    # CAS 번호 컬럼이 없으면 추가
    try:
        cursor.execute("ALTER TABLE ingredients ADD COLUMN cas_number TEXT")
        print("CAS 번호 컬럼 추가")
    except sqlite3.OperationalError:
        print("CAS 번호 컬럼이 이미 존재합니다")

    # 성분 origin 컬럼 추가 (natural, synthetic)
    try:
        cursor.execute("ALTER TABLE ingredients ADD COLUMN origin TEXT DEFAULT 'natural'")
        print("Origin 컬럼 추가")
    except sqlite3.OperationalError:
        print("Origin 컬럼이 이미 존재합니다")

    conn.commit()

    # 기존 성분 확인
    cursor.execute("SELECT name FROM ingredients")
    existing = {row[0] for row in cursor.fetchall()}

    print(f"\n기존 성분 수: {len(existing)}")
    print(f"추가할 IFRA 성분 수: {len(IFRA_INGREDIENTS)}")

    added = 0
    updated = 0

    for name, cas, category, volatility, price, ifra_limit, odor_strength in IFRA_INGREDIENTS:
        # Origin 판단 (천연 vs 합성)
        origin = "natural" if any(word in name for word in ["Oil", "Absolute", "Otto", "Resinoid", "Balsam"]) else "synthetic"

        # 단순화된 이름 (Oil, Absolute 등 제거)
        simple_name = name.replace(" Oil", "").replace(" Absolute", "").replace(" Otto", "")

        if name in existing or simple_name in existing:
            # 기존 성분 업데이트
            cursor.execute("""
                UPDATE ingredients
                SET cas_number = ?, category = ?, volatility = ?,
                    price_per_kg = ?, ifra_limit = ?, odor_strength = ?,
                    origin = ?
                WHERE name = ? OR name = ?
            """, (cas, category, volatility, price, ifra_limit, odor_strength, origin, name, simple_name))
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
                pass

    conn.commit()

    # 통계
    cursor.execute("SELECT COUNT(*) FROM ingredients")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT origin, COUNT(*) FROM ingredients GROUP BY origin")
    origin_counts = dict(cursor.fetchall())

    cursor.execute("SELECT category, COUNT(*) FROM ingredients GROUP BY category")
    category_counts = dict(cursor.fetchall())

    # CAS 번호가 있는 성분 개수
    cursor.execute("SELECT COUNT(*) FROM ingredients WHERE cas_number IS NOT NULL")
    with_cas = cursor.fetchone()[0]

    print(f"\n=== IFRA 데이터 추가 완료 ===")
    print(f"추가된 성분: {added}개")
    print(f"업데이트된 성분: {updated}개")
    print(f"전체 성분: {total}개")
    print(f"CAS 번호 보유: {with_cas}개")

    print(f"\n원료 출처별:")
    for origin, count in origin_counts.items():
        print(f"  - {origin}: {count}개")

    print(f"\n카테고리별:")
    for category, count in category_counts.items():
        print(f"  - {category} notes: {count}개")

    # 샘플 데이터 출력 (IFRA 제한 낮은 순)
    print(f"\n=== IFRA 제한이 엄격한 성분 (Top 10) ===")
    cursor.execute("""
        SELECT name, cas_number, ifra_limit, origin, category
        FROM ingredients
        WHERE cas_number IS NOT NULL
        ORDER BY ifra_limit ASC
        LIMIT 10
    """)
    for name, cas, limit, origin, category in cursor.fetchall():
        print(f"  {name} (CAS: {cas})")
        print(f"    IFRA 제한: {limit}%, {origin}, {category} note")

    # 가장 비싼 성분 Top 10
    print(f"\n=== 가장 비싼 성분 (Top 10) ===")
    cursor.execute("""
        SELECT name, price_per_kg, origin, category
        FROM ingredients
        ORDER BY price_per_kg DESC
        LIMIT 10
    """)
    for name, price, origin, category in cursor.fetchall():
        print(f"  {name}: ${price:.2f}/kg ({origin}, {category})")

    conn.close()

if __name__ == "__main__":
    update_with_cas_numbers()
