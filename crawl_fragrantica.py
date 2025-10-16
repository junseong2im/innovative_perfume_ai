import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# User-Agent 설정으로 403 우회
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def get_popular_perfumes(max_pages=10):
    """인기 향수 목록 가져오기"""
    perfumes = []

    for page in range(1, max_pages + 1):
        url = f"https://www.fragrantica.com/search/?page={page}"
        print(f"페이지 {page} 크롤링 중...")

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"  ✗ HTTP {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # 향수 카드 찾기
            perfume_cards = soup.find_all('div', class_='cell')

            for card in perfume_cards[:20]:  # 페이지당 20개
                try:
                    link = card.find('a', href=True)
                    if link and '/perfume/' in link['href']:
                        perfume_url = 'https://www.fragrantica.com' + link['href']
                        perfumes.append(perfume_url)
                except:
                    continue

            print(f"  ✓ {len(perfume_cards)} 개 발견")

            # Rate limiting - 서버에 부담 주지 않기
            time.sleep(random.uniform(2, 4))

        except Exception as e:
            print(f"  ✗ 에러: {e}")
            continue

    return perfumes

def parse_perfume_details(url):
    """개별 향수 상세 정보 파싱"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # 기본 정보
        brand = soup.find('span', itemprop='name')
        name = soup.find('h1', itemprop='name')

        if not brand or not name:
            return None

        brand_text = brand.text.strip()
        name_text = name.text.strip()

        # 년도
        year = None
        year_elem = soup.find('div', class_='flex-child-auto')
        if year_elem:
            year_text = year_elem.text
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
            if year_match:
                year = int(year_match.group())

        # 성별
        gender = 'Unisex'
        gender_imgs = soup.find_all('img', alt=True)
        for img in gender_imgs:
            alt = img['alt'].lower()
            if 'for women' in alt or 'women' in alt:
                gender = 'Women'
                break
            elif 'for men' in alt or 'men' in alt:
                gender = 'Men'
                break

        # 노트 정보
        notes = {'top': {}, 'heart': {}, 'base': {}}

        # 탑 노트
        top_notes = soup.find('div', class_='pyramid', string=lambda s: s and 'Top Notes' in s)
        if top_notes:
            parent = top_notes.find_parent('div')
            if parent:
                note_links = parent.find_all('a', href=lambda h: h and '/notes/' in h)
                for link in note_links:
                    note_name = link.text.strip()
                    notes['top'][note_name] = 10.0  # 기본값

        # 미들 노트
        middle_notes = soup.find('div', class_='pyramid', string=lambda s: s and 'Middle Notes' in s)
        if middle_notes:
            parent = middle_notes.find_parent('div')
            if parent:
                note_links = parent.find_all('a', href=lambda h: h and '/notes/' in h)
                for link in note_links:
                    note_name = link.text.strip()
                    notes['heart'][note_name] = 13.3  # 기본값

        # 베이스 노트
        base_notes = soup.find('div', class_='pyramid', string=lambda s: s and 'Base Notes' in s)
        if base_notes:
            parent = base_notes.find_parent('div')
            if parent:
                note_links = parent.find_all('a', href=lambda h: h and '/notes/' in h)
                for link in note_links:
                    note_name = link.text.strip()
                    notes['base'][note_name] = 10.0  # 기본값

        # 스타일/계열
        style = 'oriental'  # 기본값
        main_accords = soup.find_all('div', class_='accord-bar')
        if main_accords:
            # 가장 강한 어코드 찾기
            first_accord = main_accords[0].text.strip().lower()
            if 'woody' in first_accord:
                style = 'woody'
            elif 'floral' in first_accord:
                style = 'floral'
            elif 'fresh' in first_accord or 'citrus' in first_accord:
                style = 'fresh'
            elif 'aromatic' in first_accord:
                style = 'aromatic'

        # 노트 농도 정규화 (합계 100%)
        total_notes = len(notes['top']) + len(notes['heart']) + len(notes['base'])
        if total_notes > 0:
            top_pct = 30.0 / len(notes['top']) if notes['top'] else 0
            heart_pct = 40.0 / len(notes['heart']) if notes['heart'] else 0
            base_pct = 30.0 / len(notes['base']) if notes['base'] else 0

            notes['top'] = {k: top_pct for k in notes['top']}
            notes['heart'] = {k: heart_pct for k in notes['heart']}
            notes['base'] = {k: base_pct for k in notes['base']}

        return {
            'brand': brand_text,
            'name': name_text,
            'year': year or 2020,
            'gender': gender,
            'style': style,
            'notes': notes
        }

    except Exception as e:
        print(f"  ✗ 파싱 에러: {e}")
        return None

def save_to_database(perfumes_data):
    """데이터베이스에 저장"""
    conn = sqlite3.connect('fragrance_ai.db')
    cursor = conn.cursor()

    # 기존 향수 확인용
    cursor.execute("SELECT brand, name FROM recipes")
    existing = {(row[0], row[1]) for row in cursor.fetchall()}

    # 원료 목록 가져오기
    cursor.execute("SELECT name FROM fragrance_notes")
    available_ingredients = {row[0] for row in cursor.fetchall()}

    added = 0
    skipped = 0

    for data in perfumes_data:
        if (data['brand'], data['name']) in existing:
            skipped += 1
            continue

        try:
            # perfume 테이블에 추가
            cursor.execute("""
                INSERT INTO perfumes (brand, name, year, gender, style)
                VALUES (?, ?, ?, ?, ?)
            """, (data['brand'], data['name'], data['year'], data['gender'], data['style']))

            perfume_id = cursor.lastrowid

            # 노트 추가
            for position in ['top', 'heart', 'base']:
                for ingredient, concentration in data['notes'][position].items():
                    # 원료명 매칭
                    matched = find_ingredient_match(ingredient, available_ingredients)
                    if matched:
                        cursor.execute("""
                            INSERT OR IGNORE INTO perfume_notes
                            (perfume_id, ingredient_name, note_position, concentration)
                            VALUES (?, ?, ?, ?)
                        """, (perfume_id, matched, position, concentration))

            added += 1

        except Exception as e:
            print(f"  ✗ DB 저장 에러: {e}")
            continue

    conn.commit()
    conn.close()

    return added, skipped

def find_ingredient_match(recipe_name, available_ingredients):
    """원료명 스마트 매칭"""
    if recipe_name in available_ingredients:
        return recipe_name

    # 변형 시도
    variations = [
        recipe_name.replace(" Oil", ""),
        recipe_name.replace(" Absolute", ""),
        recipe_name.replace(" Extract", ""),
        recipe_name.replace(" Essence", ""),
        recipe_name + " Oil",
        recipe_name.lower().title(),
    ]

    for var in variations:
        if var in available_ingredients:
            return var

    return None

if __name__ == "__main__":
    print("=== Fragrantica 향수 데이터 크롤링 ===\n")

    # 1단계: 향수 URL 수집
    print("1단계: 향수 URL 수집 중...")
    perfume_urls = get_popular_perfumes(max_pages=5)  # 5페이지 = 약 100개
    print(f"✓ {len(perfume_urls)}개 URL 수집 완료\n")

    # 2단계: 상세 정보 파싱
    print("2단계: 상세 정보 파싱 중...")
    perfumes_data = []

    for i, url in enumerate(perfume_urls[:200], 1):  # 최대 200개
        print(f"[{i}/{min(len(perfume_urls), 200)}] {url}")

        data = parse_perfume_details(url)
        if data:
            perfumes_data.append(data)
            print(f"  ✓ {data['brand']} - {data['name']}")
        else:
            print(f"  ✗ 파싱 실패")

        # Rate limiting
        time.sleep(random.uniform(2, 5))

        # 중간 저장 (50개마다)
        if i % 50 == 0 and perfumes_data:
            print(f"\n중간 저장 중...")
            added, skipped = save_to_database(perfumes_data)
            print(f"✓ {added}개 저장, {skipped}개 중복\n")
            perfumes_data = []

    # 3단계: 최종 저장
    if perfumes_data:
        print("\n3단계: 최종 저장 중...")
        added, skipped = save_to_database(perfumes_data)
        print(f"✓ {added}개 저장, {skipped}개 중복")

    print("\n=== 크롤링 완료 ===")
