# 향수 데이터 수집 진행 상황

## 📅 작업 일자
- 시작: 2025-10-16
- 마지막 업데이트: 2025-10-16

## 🎯 목표
- **실제 향수 데이터 100개 이상 수집**
- WebFetch를 사용하여 Parfumo.com에서 정확한 향수 정보 수집
- 난수 생성이 아닌 실제 데이터로 AI 학습 데이터베이스 구축

## 📊 현재 데이터베이스 상태

### fragrance_ai.db
- **향수 레시피**: 46개 (모두 실제 브랜드의 실제 제품)
- **향료 노트**: 218개
- **레시피-원료 관계**: 525개
- **레시피당 평균 원료**: 11.4개

### 데이터 출처
1. **수동 입력** (41개): 실제 향수 이름과 대략적인 노트 구성
   - Chanel: No 5, Coco Mademoiselle, Bleu de Chanel, Chance, Allure Homme Sport, Gabrielle, No 19
   - Dior: Sauvage, J'adore, Miss Dior, Fahrenheit, Homme Intense, Poison, Dior Homme
   - Tom Ford: Black Orchid, Oud Wood, Tobacco Vanille, Neroli Portofino, Lost Cherry, Soleil Blanc, Velvet Orchid
   - YSL: Black Opium, Y, La Nuit de L'Homme, Mon Paris, Libre, Opium, L'Homme
   - Giorgio Armani: Acqua di Gio, Si, Code, My Way, Stronger With You
   - Gucci: Guilty, Bloom, Flora, Bamboo
   - Paco Rabanne: 1 Million, Invictus, Lady Million, Phantom

2. **WebFetch로 수집** (5개): Parfumo.com에서 정확한 노트 정보 수집
   - Lancôme La Vie est Belle (2012)
   - Viktor & Rolf Flowerbomb (2004)
   - Dolce & Gabbana Light Blue (2001)
   - Versace Eros (2012)
   - Yves Saint Laurent Black Opium (2014)

## 🔍 WebFetch로 수집한 향수 상세 정보

### 성공적으로 수집한 향수 (20개)

#### 1. Chanel No 5 (1921)
- **브랜드**: Chanel
- **성별**: Women
- **스타일**: Floral-Powdery
- **조향사**: Ernest Beaux
- **Top**: Aldehydes, Bergamot, Lemon, Neroli
- **Heart**: Grasse jasmine, Iris, Lily of the valley, May rose
- **Base**: Sandalwood, Vanilla, Amber, Vetiver

#### 2. Dior Sauvage (2015)
- **브랜드**: Dior
- **성별**: Men
- **스타일**: Fresh-spicy
- **조향사**: François Demachy
- **Top**: Calabrian bergamot, Sichuan pepper, Pink pepper, Provençal lavender
- **Heart**: Geranium, Elemi resin
- **Base**: Ambrox, Patchouli, Vetiver

#### 3. YSL Black Opium (2014)
- **브랜드**: Yves Saint Laurent
- **성별**: Women
- **스타일**: Sweet-gourmand
- **조향사**: Nathalie Lorson, Marie Salamagne, Olivier Cresp, Honorine Blanc
- **Top**: Orange blossom, Pink pepper
- **Heart**: Coffee, Jasmine
- **Base**: Vanilla, Patchouli, Cedar

#### 4. Giorgio Armani Acqua di Gio (1995)
- **브랜드**: Giorgio Armani
- **성별**: Women
- **스타일**: Fresh-floral
- **조향사**: Alberto Morillas
- **Top**: Lemon, Peach, Peony, Banana leaf, Pineapple, Violet, Vodka
- **Heart**: Lily of the valley, Freesia, Hyacinth, Ylang-ylang, Jasmine, Lily
- **Base**: Musk, Cedarwood, Amber, Sandalwood, Styrax

#### 5. Gucci Guilty (2010)
- **브랜드**: Gucci
- **성별**: Women
- **스타일**: Floral-sweet
- **조향사**: Aurélien Guichard
- **Top**: Mandarin orange, Pink pepper
- **Heart**: Lilac, Egyptian geranium, Peach, Raspberry
- **Base**: Amber, Patchouli

#### 6. Lancôme La Vie est Belle (2012)
- **브랜드**: Lancôme
- **성별**: Women
- **스타일**: Sweet-floral
- **조향사**: Olivier Polge, Dominique Ropion, Anne Flipo
- **Top**: Blackcurrant, Pear
- **Heart**: Iris, Jasmine, Orange blossom
- **Base**: Praliné, Patchouli, Tonka bean, Vanilla

#### 7. Viktor & Rolf Flowerbomb (2004)
- **브랜드**: Viktor & Rolf
- **성별**: Women
- **스타일**: Sweet-floral
- **조향사**: Olivier Polge, Carlos Benaïm, Domitille Michalon-Bertier, Dominique Ropion
- **Top**: Tea, Bergamot
- **Heart**: Orchid, Freesia, Jasmine
- **Base**: Patchouli, Rose

#### 8. Paco Rabanne 1 Million (2008)
- **브랜드**: Paco Rabanne
- **성별**: Men
- **스타일**: Sweet-spicy
- **조향사**: Christophe Raynaud, Olivier Pescheux, Michel Girard
- **Top**: Red mandarin orange, Peppermint
- **Heart**: Cinnamon, Rose absolute
- **Base**: Amberketal, Leather

#### 9. Dolce & Gabbana Light Blue (2001)
- **브랜드**: Dolce & Gabbana
- **성별**: Women
- **스타일**: Fresh-citrusy
- **조향사**: Olivier Cresp
- **Top**: Sicilian citron, Apple, Bluebell
- **Heart**: Bamboo, Jasmine, Rose
- **Base**: Cedarwood, Musk, Amber

#### 10. Versace Eros (2012)
- **브랜드**: Versace
- **성별**: Men
- **스타일**: Sweet-fresh
- **조향사**: Aurélien Guichard
- **Top**: Green apple, Mint, Italian lemon
- **Heart**: Venezuelan tonka bean, Ambroxan, Geranium
- **Base**: Bourbon vanilla, Atlas cedar, Virginia cedar, Oakmoss, Vetiver

#### 11. Creed Aventus (2010)
- **브랜드**: Creed
- **성별**: Men
- **스타일**: Fresh-fruity
- **Top**: Bergamot, Apple, Blackcurrant, Lemon, Pink pepper
- **Heart**: Pineapple, Indonesian patchouli, Jasmine
- **Base**: Birch, Cedarwood, Musk, Oakmoss, Ambergris

#### 12. Dior Hypnotic Poison (1998)
- **브랜드**: Dior
- **성별**: Women
- **스타일**: Sweet-oriental
- **Top**: Bitter almond, Coconut, Apricot, Pimento, Plum
- **Heart**: Jasmine sambac, Tuberose, Lily of the valley, Rose, Rosewood
- **Base**: Vanilla, Sandalwood

#### 13. Guerlain Shalimar (1986)
- **브랜드**: Guerlain
- **성별**: Women
- **스타일**: Oriental-spicy
- **Top**: Bergamot, Blossoms
- **Heart**: Iris, Jasmine, Rose
- **Base**: Vanilla, Balsamic notes, Tonka bean

#### 14. Prada L'Homme (2016)
- **브랜드**: Prada
- **성별**: Men
- **스타일**: Powdery-fresh
- **Top**: Neroli, Pepper
- **Heart**: Iris, Violet, Geranium
- **Base**: Amber, Cedar, Patchouli

#### 15. Chanel Coco Mademoiselle (2001)
- **브랜드**: Chanel
- **성별**: Women
- **스타일**: Floral-sweet
- **Top**: Orange, Mandarin orange, Bergamot, Orange blossom
- **Heart**: Jasmine, Rose, Ylang-ylang, Mimosa
- **Base**: Patchouli, Vetiver, White musk, Opoponax, Tonka bean, Vanilla

#### 16. Calvin Klein CK One (1994)
- **브랜드**: Calvin Klein
- **성별**: Unisex
- **스타일**: Fresh-citrusy
- **Top**: Bergamot, Lemon, Mandarin orange, Papaya
- **Heart**: Green tea, Jasmine, Lily of the valley, Nutmeg, Violet, Rose
- **Base**: Musk, Amber

#### 17. Marc Jacobs Daisy (2007)
- **브랜드**: Marc Jacobs
- **성별**: Women
- **스타일**: Floral-fresh
- **Top**: Violet leaf, Red grapefruit, Woodland strawberry
- **Heart**: Gardenia, Violet, Jasmine
- **Base**: Blond woods, Musk, Vanilla

#### 18. Burberry London for Men (2006)
- **브랜드**: Burberry
- **성별**: Men
- **스타일**: Spicy-woody
- **Top**: Cinnamon, Lavender, Bergamot
- **Heart**: Leather, Mimosa
- **Base**: Tobacco, Oakmoss

#### 19. Hermès Terre d'Hermès (2006)
- **브랜드**: Hermès
- **성별**: Men
- **스타일**: Woody-earthy
- **Top**: Orange, Grapefruit, Pink Pepper
- **Heart**: Geranium Leaf, Pepper
- **Base**: Vetiver, Cedar, Benzoin, Patchouli, Flintstone

#### 20. Bvlgari Omnia (2003)
- **브랜드**: Bvlgari
- **성별**: Women
- **스타일**: Spicy-oriental
- **Top**: Cardamom, Ginger, Pepper, Mandarin orange, Saffron
- **Heart**: Masala chai, Cinnamon, Almond, Lotus, Nutmeg, Clove
- **Base**: Gaiac wood, Tonka bean, Woody notes, Sandalwood, White chocolate

## 🛠️ 사용 가능한 스크립트 파일들

### 1. add_webfetch_perfumes.py
- WebFetch로 수집한 향수 데이터를 데이터베이스에 추가
- 스마트 매칭 함수로 향료명 자동 매칭
- 현재 10개 향수 데이터 포함

### 2. add_real_perfumes_simple.py
- 수동으로 입력한 41개 실제 향수 데이터
- 향료명 정규화 및 매칭 기능
- INSERT OR IGNORE로 중복 방지

### 3. test_note_matching.py
- 향료명 매칭 테스트 스크립트
- 데이터베이스의 향료와 입력 향료명 비교

### 4. crawl_fragrantica.py
- Fragrantica 크롤링 시도 (403 에러로 실패)
- 사용 불가

### 5. crawl_selenium.py
- Selenium을 사용한 크롤링 (ChromeDriver 필요)
- 현재 미사용

## 📝 매칭되지 않은 향료 (추가 필요)

다음 향료들이 데이터베이스에 없어서 매칭 실패:
- Ambroxan
- Bitter Almond
- Bluebell
- Coffee
- Freesia
- Oakmoss
- Orchid
- Pink Pepper
- Praliné
- Sicilian Citron
- Tea
- Aldehydes
- Calabrian Bergamot
- Sichuan Pepper
- Provençal Lavender
- Elemi Resin
- Grasse Jasmine
- May Rose
- Masala Chai
- Gaiac Wood
- White Chocolate
- Amberketal
- Flintstone
- 등...

## 🚀 다음 작업 계획

### 즉시 진행 가능한 작업

#### 1단계: 향료 추가
```python
# 매칭 안 된 향료들을 fragrance_notes 테이블에 추가
python add_missing_notes.py
```

#### 2단계: WebFetch로 더 많은 향수 수집 (목표: 100개)
Parfumo.com에서 인기 향수 Top 100 수집:

**남성 향수 추천 목록:**
- Louis Vuitton Imagination
- Parfums de Marly Layton
- Dior Homme Intense (2011)
- Creed Aventus ✅ (수집 완료)
- Amouage Reflection Man
- Prada L'Homme ✅ (수집 완료)
- Viktor & Rolf Spicebomb Extreme
- YSL La Nuit de L'Homme ✅ (수집 완료)
- Jean Paul Gaultier Le Mâle Le Parfum
- Parfums de Marly Althaïr

**여성 향수 추천 목록:**
- Dior Hypnotic Poison ✅ (수집 완료)
- Guerlain Mon Guerlain
- Parfums de Marly Delina
- Parfums de Marly Delina Exclusif
- Editions de Parfums Frédéric Malle Portrait of a Lady
- Givenchy L'Interdit Rouge
- Guerlain Shalimar ✅ (수집 완료)
- Guerlain L'Instant Magic
- Chanel Coco Mademoiselle ✅ (수집 완료)
- Kayali Vanilla 28

#### 3단계: 데이터 품질 개선
- 수동 입력 41개 향수의 노트 정보를 WebFetch로 재수집하여 정확도 향상
- 모든 향수에 조향사 정보 추가
- 메인 어코드 정보 추가

#### 4단계: 최종 검증
```bash
python -c "
import sqlite3
conn = sqlite3.connect('fragrance_ai.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM recipes')
total = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM recipe_ingredients')
ingredients = cursor.fetchone()[0]

print(f'최종 향수: {total}개')
print(f'노트 관계: {ingredients}개')
print(f'평균: {ingredients/total:.1f}개 노트/향수')

conn.close()
"
```

## 📂 데이터베이스 구조

### recipes 테이블
- id (VARCHAR): 고유 ID
- name (VARCHAR): 브랜드 + 향수 이름 (예: "Chanel No 5")
- recipe_type (VARCHAR): basic, detailed, premium, variation
- fragrance_family (VARCHAR): floral, oriental, woody, fresh, aromatic
- complexity (INTEGER): 1-10
- status (VARCHAR): draft, reviewed, approved, archived
- created_at (DATETIME)

### fragrance_notes 테이블
- id (VARCHAR): 고유 ID
- name (VARCHAR): 향료 이름 (소문자, 언더스코어)
- note_type (VARCHAR): top, middle, base
- intensity (FLOAT)
- longevity (FLOAT)

### recipe_ingredients 테이블
- id (VARCHAR): 고유 ID
- recipe_id (VARCHAR): recipes.id 참조
- note_id (VARCHAR): fragrance_notes.id 참조
- percentage (FLOAT): 농도 (%)
- role (VARCHAR): primary, accent, bridge, modifier
- note_position (VARCHAR): top, middle, base
- UNIQUE (recipe_id, note_id)

## 🔧 유용한 명령어

### 데이터베이스 상태 확인
```bash
python -c "
import sqlite3
conn = sqlite3.connect('fragrance_ai.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM recipes')
print(f'향수: {cursor.fetchone()[0]}개')

cursor.execute('SELECT COUNT(*) FROM fragrance_notes')
print(f'향료: {cursor.fetchone()[0]}개')

cursor.execute('SELECT COUNT(*) FROM recipe_ingredients')
print(f'노트 관계: {cursor.fetchone()[0]}개')

conn.close()
"
```

### 중복 확인
```bash
python -c "
import sqlite3
conn = sqlite3.connect('fragrance_ai.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT name, COUNT(*)
    FROM recipes
    GROUP BY name
    HAVING COUNT(*) > 1
''')

duplicates = cursor.fetchall()
print(f'중복: {len(duplicates)}개')
for name, count in duplicates:
    print(f'  - {name}: {count}개')

conn.close()
"
```

### WebFetch로 향수 정보 가져오기
```python
# 예시:
from webfetch import WebFetch

url = "https://www.parfumo.com/Perfumes/Brand/Perfume_Name"
result = WebFetch(url, "Extract perfume information: brand, name, year, notes")
```

## ⚠️ 주의사항

1. **Fragrantica 크롤링 불가**: 403 에러로 차단됨
2. **Parfumo.com 사용**: WebFetch로 접근 가능, 실제 향수 정보 수집
3. **향료명 매칭**: 데이터베이스는 소문자 + 언더스코어 형식 (예: "pink_pepper")
4. **중복 방지**: INSERT OR IGNORE 사용
5. **노트 구성**: Top (30%), Middle/Heart (40%), Base (30%) 권장

## 📈 진행률

- ✅ 데이터베이스 구조 확인: 100%
- ✅ WebFetch 방법 확립: 100%
- ✅ 향수 데이터 수집: 20% (20/100개)
- ⏳ 향료 추가 작업: 0%
- ⏳ 데이터 품질 개선: 0%

## 🎯 최종 목표

**100개 이상의 실제 향수 데이터**를 WebFetch로 수집하여 AI 학습에 사용할 고품질 데이터셋 구축

---

## 📌 재시작 방법

다음 세션에서 작업을 이어서 할 때:

1. 이 문서 읽기: `PERFUME_DATA_COLLECTION_PROGRESS.md`
2. 현재 데이터베이스 상태 확인
3. WebFetch로 향수 20개씩 수집 (위의 추천 목록 참고)
4. `add_webfetch_perfumes.py` 업데이트 및 실행
5. 100개 달성까지 반복

**명령어:**
```bash
# 현재 상태 확인
python -c "import sqlite3; conn = sqlite3.connect('fragrance_ai.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM recipes'); print(f'현재: {cursor.fetchone()[0]}개')"

# WebFetch로 수집 (Claude에게 요청)
# "Parfumo.com에서 [향수명] 정보 가져와줘"

# 데이터 추가
python add_webfetch_perfumes.py
```
