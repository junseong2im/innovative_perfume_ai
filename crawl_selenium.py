from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import sqlite3
import time
import random
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def setup_driver():
    """Chrome 드라이버 설정"""
    options = Options()
    # options.add_argument('--headless')  # 백그라운드 실행
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    driver = webdriver.Chrome(options=options)
    return driver

def crawl_perfume_page(driver, url):
    """개별 향수 페이지 크롤링"""
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 4))

        # 브랜드와 이름
        brand = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="name"]').text.strip()
        name = driver.find_element(By.CSS_SELECTOR, 'h1[itemprop="name"]').text.strip()

        # 년도 추출
        year = 2020
        try:
            year_text = driver.find_element(By.CSS_SELECTOR, 'div.flex-child-auto').text
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
            if year_match:
                year = int(year_match.group())
        except:
            pass

        # 성별
        gender = 'Unisex'
        try:
            images = driver.find_elements(By.TAG_NAME, 'img')
            for img in images:
                alt = img.get_attribute('alt')
                if alt and 'women' in alt.lower():
                    gender = 'Women'
                    break
                elif alt and 'men' in alt.lower():
                    gender = 'Men'
                    break
        except:
            pass

        # 노트 정보
        notes = {'top': [], 'heart': [], 'base': []}

        try:
            # 피라미드 구조에서 노트 추출
            pyramid_sections = driver.find_elements(By.CSS_SELECTOR, 'div.pyramid')

            for section in pyramid_sections:
                section_text = section.text.lower()
                note_links = section.find_elements(By.TAG_NAME, 'a')

                for link in note_links:
                    href = link.get_attribute('href')
                    if href and '/notes/' in href:
                        note_name = link.text.strip()

                        if 'top' in section_text:
                            notes['top'].append(note_name)
                        elif 'middle' in section_text or 'heart' in section_text:
                            notes['heart'].append(note_name)
                        elif 'base' in section_text:
                            notes['base'].append(note_name)
        except Exception as e:
            print(f"  노트 추출 에러: {e}")

        # 스타일 추출
        style = 'oriental'
        try:
            accords = driver.find_elements(By.CSS_SELECTOR, 'div.accord-bar')
            if accords:
                first_accord = accords[0].text.strip().lower()
                if 'woody' in first_accord:
                    style = 'woody'
                elif 'floral' in first_accord:
                    style = 'floral'
                elif 'fresh' in first_accord or 'citrus' in first_accord:
                    style = 'fresh'
                elif 'aromatic' in first_accord:
                    style = 'aromatic'
        except:
            pass

        # 농도 계산
        def calc_concentration(notes_dict):
            result = {}
            if notes_dict['top']:
                pct = 30.0 / len(notes_dict['top'])
                result['top'] = {n: pct for n in notes_dict['top']}
            else:
                result['top'] = {}

            if notes_dict['heart']:
                pct = 40.0 / len(notes_dict['heart'])
                result['heart'] = {n: pct for n in notes_dict['heart']}
            else:
                result['heart'] = {}

            if notes_dict['base']:
                pct = 30.0 / len(notes_dict['base'])
                result['base'] = {n: pct for n in notes_dict['base']}
            else:
                result['base'] = {}

            return result

        notes_with_pct = calc_concentration(notes)

        return {
            'brand': brand,
            'name': name,
            'year': year,
            'gender': gender,
            'style': style,
            'notes': notes_with_pct
        }

    except Exception as e:
        print(f"  ✗ 크롤링 에러: {e}")
        return None

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
    print("=== Selenium 향수 데이터 크롤링 ===\n")

    # 크롤링할 향수 URL 목록 (수동으로 준비)
    perfume_urls = [
        "https://www.fragrantica.com/perfume/Chanel/Chanel-No-5-615.html",
        "https://www.fragrantica.com/perfume/Dior/Sauvage-31861.html",
        "https://www.fragrantica.com/perfume/Tom-Ford/Black-Orchid-1018.html",
        "https://www.fragrantica.com/perfume/Yves-Saint-Laurent/Black-Opium-21729.html",
        "https://www.fragrantica.com/perfume/Giorgio-Armani/Acqua-di-Gio-410.html",
        "https://www.fragrantica.com/perfume/Gucci/Guilty-6777.html",
        "https://www.fragrantica.com/perfume/Paco-Rabanne/1-Million-3747.html",
        "https://www.fragrantica.com/perfume/Viktor-Rolf/Flowerbomb-2852.html",
        "https://www.fragrantica.com/perfume/Dolce-Gabbana/Light-Blue-222.html",
        "https://www.fragrantica.com/perfume/Lancome/La-Vie-Est-Belle-16492.html",
    ]

    try:
        driver = setup_driver()
        print("✓ Chrome 드라이버 시작\n")

        perfumes_data = []

        for i, url in enumerate(perfume_urls, 1):
            print(f"[{i}/{len(perfume_urls)}] {url}")

            data = crawl_perfume_page(driver, url)
            if data:
                perfumes_data.append(data)
                print(f"  ✓ {data['brand']} - {data['name']}")
                print(f"    Top: {len(data['notes']['top'])}개, Heart: {len(data['notes']['heart'])}개, Base: {len(data['notes']['base'])}개")
            else:
                print(f"  ✗ 크롤링 실패")

            time.sleep(random.uniform(3, 6))

        driver.quit()

        # DB 저장
        if perfumes_data:
            print(f"\n데이터베이스 저장 중...")

            conn = sqlite3.connect('fragrance_ai.db')
            cursor = conn.cursor()

            # 원료 목록
            cursor.execute("SELECT name FROM fragrance_notes")
            available_ingredients = {row[0] for row in cursor.fetchall()}

            added = 0
            for data in perfumes_data:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO perfumes (brand, name, year, gender, style)
                        VALUES (?, ?, ?, ?, ?)
                    """, (data['brand'], data['name'], data['year'], data['gender'], data['style']))

                    # ... 노트 저장 로직 ...

                    added += 1
                except Exception as e:
                    print(f"  DB 에러: {e}")

            conn.commit()
            conn.close()

            print(f"✓ {added}개 저장 완료")

    except Exception as e:
        print(f"✗ 에러: {e}")
        print("\nChrome 드라이버가 설치되어 있어야 합니다.")
        print("다운로드: https://chromedriver.chromium.org/")

    print("\n=== 크롤링 완료 ===")
