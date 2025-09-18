"""
향수 데이터 웹 스크래퍼
다양한 향수 웹사이트에서 체계적으로 데이터를 수집
"""

import asyncio
import aiohttp
import time
import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from pathlib import Path
import hashlib
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

@dataclass
class FragranceData:
    """향수 데이터 구조"""
    id: str
    name: str
    brand: str
    description: str
    top_notes: List[str]
    heart_notes: List[str]
    base_notes: List[str]
    price: Optional[float] = None
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    availability: Optional[str] = None
    image_urls: Optional[List[str]] = None
    category: Optional[str] = None
    launch_year: Optional[int] = None
    perfumer: Optional[str] = None
    longevity: Optional[str] = None
    sillage: Optional[str] = None
    gender: Optional[str] = None
    season: Optional[List[str]] = None
    occasion: Optional[List[str]] = None
    data_source: Optional[str] = None
    scraped_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FragranceDataScraper:
    """향수 데이터 전문 스크래퍼"""

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        enable_cache: bool = True,
        cache_dir: str = "./cache/scraper",
        user_agent: str = None
    ):
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

        # 웹드라이버 설정 (동적 콘텐츠용)
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument(f'--user-agent={self.user_agent}')

        logger.info(f"FragranceDataScraper initialized with rate_limit={rate_limit_delay}s")

    async def scrape_fragrance_data(
        self,
        url: str,
        max_items: int = 100,
        include_reviews: bool = False
    ) -> List[Dict[str, Any]]:
        """메인 스크래핑 함수"""

        try:
            # URL 파싱으로 소스 결정
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()

            if 'fragrantica' in domain:
                return self._scrape_fragrantica(url, max_items, include_reviews)
            elif 'basenotes' in domain:
                return self._scrape_basenotes(url, max_items, include_reviews)
            elif 'parfumo' in domain:
                return self._scrape_parfumo(url, max_items, include_reviews)
            elif 'amorepacific' in domain:
                return self._scrape_amorepacific(url, max_items, include_reviews)
            elif 'oliveyoung' in domain:
                return self._scrape_oliveyoung(url, max_items, include_reviews)
            else:
                return self._scrape_generic_site(url, max_items, include_reviews)

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return []

    async def _scrape_fragrantica(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """Fragrantica.com 스크래핑"""

        logger.info(f"Scraping Fragrantica: {url}")
        fragrances = []

        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)

            # 향수 리스트 페이지에서 개별 URL 수집
            perfume_links = []

            # 페이지네이션 처리
            page = 1
            while len(perfume_links) < max_items and page <= 10:  # 최대 10페이지
                try:
                    # 향수 링크 찾기
                    perfume_elements = driver.find_elements(
                        By.CSS_SELECTOR,
                        "div.perfume-cell a[href*='/perfumes/']"
                    )

                    for elem in perfume_elements:
                        href = elem.get_attribute('href')
                        if href and href not in perfume_links:
                            perfume_links.append(href)

                        if len(perfume_links) >= max_items:
                            break

                    # 다음 페이지로
                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, "a.next")
                        if next_button.is_enabled():
                            next_button.click()
                            await asyncio.sleep(2)
                            page += 1
                        else:
                            break
                    except:
                        break

                except Exception as e:
                    logger.warning(f"Failed to get page {page}: {e}")
                    break

            # 개별 향수 페이지 스크래핑
            for i, link in enumerate(perfume_links[:max_items]):
                try:
                    fragrance_data = self._scrape_fragrantica_detail(driver, link, include_reviews)
                    if fragrance_data:
                        fragrances.append(fragrance_data.to_dict())
                        logger.info(f"Scraped {i+1}/{min(len(perfume_links), max_items)}: {fragrance_data.name}")

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to scrape fragrance {link}: {e}")
                    continue

            driver.quit()

        except Exception as e:
            logger.error(f"Fragrantica scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} fragrances from Fragrantica")
        return fragrances

    async def _scrape_fragrantica_detail(
        self,
        driver: webdriver.Chrome,
        url: str,
        include_reviews: bool
    ) -> Optional[FragranceData]:
        """개별 Fragrantica 향수 페이지 스크래핑"""

        try:
            driver.get(url)
            await asyncio.sleep(1)

            # 기본 정보 추출
            name_elem = driver.find_element(By.CSS_SELECTOR, "h1[itemprop='name']")
            name = name_elem.text.strip()

            brand_elem = driver.find_element(By.CSS_SELECTOR, "span[itemprop='brand']")
            brand = brand_elem.text.strip()

            # 설명
            try:
                desc_elem = driver.find_element(By.CSS_SELECTOR, "div[itemprop='description']")
                description = desc_elem.text.strip()
            except:
                description = ""

            # 노트 정보
            top_notes = self._extract_fragrantica_notes(driver, "Top Notes")
            heart_notes = self._extract_fragrantica_notes(driver, "Middle Notes")
            base_notes = self._extract_fragrantica_notes(driver, "Base Notes")

            # 평점
            try:
                rating_elem = driver.find_element(By.CSS_SELECTOR, "span[itemprop='ratingValue']")
                rating = float(rating_elem.text.strip())
            except:
                rating = None

            # 퍼퓨머
            try:
                perfumer_elem = driver.find_element(By.CSS_SELECTOR, "span[itemprop='author']")
                perfumer = perfumer_elem.text.strip()
            except:
                perfumer = None

            # 출시년도
            try:
                year_elem = driver.find_element(By.XPATH, "//span[contains(text(), 'year')]")
                year_text = year_elem.text
                year_match = re.search(r'\d{4}', year_text)
                launch_year = int(year_match.group()) if year_match else None
            except:
                launch_year = None

            # 성별
            try:
                gender_elem = driver.find_element(By.XPATH, "//span[contains(text(), 'for')]")
                gender_text = gender_elem.text.lower()
                if 'women' in gender_text and 'men' in gender_text:
                    gender = 'unisex'
                elif 'women' in gender_text:
                    gender = 'women'
                elif 'men' in gender_text:
                    gender = 'men'
                else:
                    gender = None
            except:
                gender = None

            # 이미지 URL
            try:
                img_elem = driver.find_element(By.CSS_SELECTOR, "img[itemprop='image']")
                image_urls = [img_elem.get_attribute('src')]
            except:
                image_urls = []

            # ID 생성
            fragrance_id = hashlib.md5(f"{brand}_{name}".encode()).hexdigest()[:12]

            return FragranceData(
                id=fragrance_id,
                name=name,
                brand=brand,
                description=description,
                top_notes=top_notes,
                heart_notes=heart_notes,
                base_notes=base_notes,
                rating=rating,
                perfumer=perfumer,
                launch_year=launch_year,
                gender=gender,
                image_urls=image_urls,
                data_source='fragrantica',
                scraped_at=datetime.utcnow().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to scrape Fragrantica detail {url}: {e}")
            return None

    def _extract_fragrantica_notes(self, driver: webdriver.Chrome, note_type: str) -> List[str]:
        """Fragrantica 노트 추출"""
        try:
            notes_section = driver.find_element(
                By.XPATH,
                f"//div[contains(@class, 'pyramid')]//div[contains(text(), '{note_type}')]/following-sibling::div"
            )
            notes_elements = notes_section.find_elements(By.TAG_NAME, "a")
            return [elem.text.strip() for elem in notes_elements if elem.text.strip()]
        except:
            return []

    async def _scrape_basenotes(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """Basenotes.com 스크래핑"""

        logger.info(f"Scraping Basenotes: {url}")
        fragrances = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 향수 리스트 찾기
            perfume_elements = soup.find_all('div', class_='perfume-item')

            for i, elem in enumerate(perfume_elements[:max_items]):
                try:
                    # 기본 정보
                    name_elem = elem.find('h3') or elem.find('h4')
                    name = name_elem.text.strip() if name_elem else "Unknown"

                    brand_elem = elem.find('span', class_='brand')
                    brand = brand_elem.text.strip() if brand_elem else "Unknown"

                    # 설명
                    desc_elem = elem.find('p', class_='description')
                    description = desc_elem.text.strip() if desc_elem else ""

                    # 노트 (간단한 파싱)
                    notes_elem = elem.find('div', class_='notes')
                    notes_text = notes_elem.text if notes_elem else ""

                    # 노트를 대략적으로 분류 (실제 사이트 구조에 맞게 조정 필요)
                    all_notes = [note.strip() for note in notes_text.split(',') if note.strip()]
                    top_notes = all_notes[:3] if len(all_notes) > 3 else all_notes
                    heart_notes = all_notes[3:6] if len(all_notes) > 6 else []
                    base_notes = all_notes[6:] if len(all_notes) > 6 else []

                    # ID 생성
                    fragrance_id = hashlib.md5(f"{brand}_{name}".encode()).hexdigest()[:12]

                    fragrance = FragranceData(
                        id=fragrance_id,
                        name=name,
                        brand=brand,
                        description=description,
                        top_notes=top_notes,
                        heart_notes=heart_notes,
                        base_notes=base_notes,
                        data_source='basenotes',
                        scraped_at=datetime.utcnow().isoformat()
                    )

                    fragrances.append(fragrance.to_dict())

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to parse Basenotes item {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Basenotes scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} fragrances from Basenotes")
        return fragrances

    async def _scrape_oliveyoung(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """올리브영 향수 스크래핑"""

        logger.info(f"Scraping Olive Young: {url}")
        fragrances = []

        try:
            driver = webdriver.Chrome(options=self.chrome_options)
            driver.get(url)
            await asyncio.sleep(2)

            # 상품 리스트 찾기
            product_elements = driver.find_elements(
                By.CSS_SELECTOR,
                "li[data-ref-goodsno] .prod_info"
            )

            for i, elem in enumerate(product_elements[:max_items]):
                try:
                    # 상품명
                    name_elem = elem.find_element(By.CSS_SELECTOR, ".tx_name")
                    full_name = name_elem.text.strip()

                    # 브랜드와 제품명 분리
                    name_parts = full_name.split(']')
                    if len(name_parts) > 1:
                        brand = name_parts[0].replace('[', '').strip()
                        name = name_parts[1].strip()
                    else:
                        brand = "Unknown"
                        name = full_name

                    # 가격
                    try:
                        price_elem = elem.find_element(By.CSS_SELECTOR, ".tx_cur .tx_num")
                        price_text = price_elem.text.replace(',', '').replace('원', '')
                        price = float(price_text)
                    except:
                        price = None

                    # 평점
                    try:
                        rating_elem = elem.find_element(By.CSS_SELECTOR, ".ico_grade")
                        rating_text = rating_elem.get_attribute('class')
                        # 클래스명에서 별점 추출 (구체적인 구조에 맞게 조정)
                        rating_match = re.search(r'grade(\d)', rating_text)
                        rating = int(rating_match.group(1)) if rating_match else None
                    except:
                        rating = None

                    # ID 생성
                    fragrance_id = hashlib.md5(f"oliveyoung_{brand}_{name}".encode()).hexdigest()[:12]

                    fragrance = FragranceData(
                        id=fragrance_id,
                        name=name,
                        brand=brand,
                        description="",  # 상세 페이지에서 가져와야 함
                        top_notes=[],
                        heart_notes=[],
                        base_notes=[],
                        price=price,
                        rating=float(rating) if rating else None,
                        category="fragrance",
                        data_source='olive_young',
                        scraped_at=datetime.utcnow().isoformat()
                    )

                    fragrances.append(fragrance.to_dict())

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to parse Olive Young item {i}: {e}")
                    continue

            driver.quit()

        except Exception as e:
            logger.error(f"Olive Young scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} fragrances from Olive Young")
        return fragrances

    async def _scrape_amorepacific(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """아모레퍼시픽 향수 스크래핑"""

        logger.info(f"Scraping Amore Pacific: {url}")
        fragrances = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 상품 요소들 찾기 (실제 구조에 맞게 조정)
            product_elements = soup.find_all('div', class_='product-item')

            for i, elem in enumerate(product_elements[:max_items]):
                try:
                    # 상품명
                    name_elem = elem.find('h3') or elem.find('div', class_='product-name')
                    name = name_elem.text.strip() if name_elem else "Unknown"

                    # 브랜드 (아모레퍼시픽 계열사들)
                    brand_elem = elem.find('div', class_='brand')
                    brand = brand_elem.text.strip() if brand_elem else "아모레퍼시픽"

                    # 가격
                    try:
                        price_elem = elem.find('span', class_='price')
                        price_text = price_elem.text.replace(',', '').replace('원', '').replace('₩', '')
                        price = float(re.findall(r'\d+', price_text)[0])
                    except:
                        price = None

                    # 설명
                    desc_elem = elem.find('p', class_='description')
                    description = desc_elem.text.strip() if desc_elem else ""

                    # ID 생성
                    fragrance_id = hashlib.md5(f"amorepacific_{brand}_{name}".encode()).hexdigest()[:12]

                    fragrance = FragranceData(
                        id=fragrance_id,
                        name=name,
                        brand=brand,
                        description=description,
                        top_notes=[],  # 상세 페이지에서 가져와야 함
                        heart_notes=[],
                        base_notes=[],
                        price=price,
                        data_source='amorepacific',
                        scraped_at=datetime.utcnow().isoformat()
                    )

                    fragrances.append(fragrance.to_dict())

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to parse Amore Pacific item {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Amore Pacific scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} fragrances from Amore Pacific")
        return fragrances

    async def _scrape_parfumo(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """Parfumo 스크래핑"""

        logger.info(f"Scraping Parfumo: {url}")
        fragrances = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 향수 리스트 찾기
            perfume_elements = soup.find_all('div', class_='perfume-card')

            for i, elem in enumerate(perfume_elements[:max_items]):
                try:
                    # 기본 정보 추출 (실제 구조에 맞게 조정)
                    name_elem = elem.find('h4') or elem.find('a', class_='perfume-name')
                    name = name_elem.text.strip() if name_elem else "Unknown"

                    brand_elem = elem.find('span', class_='brand-name')
                    brand = brand_elem.text.strip() if brand_elem else "Unknown"

                    # 평점
                    try:
                        rating_elem = elem.find('span', class_='rating-value')
                        rating = float(rating_elem.text.strip())
                    except:
                        rating = None

                    # ID 생성
                    fragrance_id = hashlib.md5(f"parfumo_{brand}_{name}".encode()).hexdigest()[:12]

                    fragrance = FragranceData(
                        id=fragrance_id,
                        name=name,
                        brand=brand,
                        description="",
                        top_notes=[],
                        heart_notes=[],
                        base_notes=[],
                        rating=rating,
                        data_source='parfumo',
                        scraped_at=datetime.utcnow().isoformat()
                    )

                    fragrances.append(fragrance.to_dict())

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to parse Parfumo item {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Parfumo scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} fragrances from Parfumo")
        return fragrances

    async def _scrape_generic_site(
        self,
        url: str,
        max_items: int,
        include_reviews: bool
    ) -> List[Dict[str, Any]]:
        """일반 사이트 스크래핑 (폴백)"""

        logger.info(f"Scraping generic site: {url}")
        fragrances = []

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # 일반적인 상품 구조 패턴들 시도
            selectors = [
                'div[class*="product"]',
                'div[class*="item"]',
                'li[class*="product"]',
                'article[class*="product"]'
            ]

            product_elements = []
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    product_elements = elements[:max_items]
                    break

            for i, elem in enumerate(product_elements):
                try:
                    # 텍스트에서 향수 관련 키워드 찾기
                    text_content = elem.get_text().strip()

                    if not any(keyword in text_content.lower() for keyword in
                              ['perfume', '향수', 'fragrance', 'cologne', 'eau de']):
                        continue

                    # 기본 정보 추출 시도
                    name = "Unknown"
                    brand = "Unknown"

                    # 제목 또는 이름 요소 찾기
                    for tag in ['h1', 'h2', 'h3', 'h4', 'h5']:
                        title_elem = elem.find(tag)
                        if title_elem:
                            name = title_elem.text.strip()[:100]  # 길이 제한
                            break

                    # ID 생성
                    fragrance_id = hashlib.md5(f"generic_{brand}_{name}_{i}".encode()).hexdigest()[:12]

                    fragrance = FragranceData(
                        id=fragrance_id,
                        name=name,
                        brand=brand,
                        description=text_content[:500],  # 요약된 설명
                        top_notes=[],
                        heart_notes=[],
                        base_notes=[],
                        data_source='generic',
                        scraped_at=datetime.utcnow().isoformat()
                    )

                    fragrances.append(fragrance.to_dict())

                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Failed to parse generic item {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Generic site scraping failed: {e}")

        logger.info(f"Scraped {len(fragrances)} items from generic site")
        return fragrances

    def get_cached_data(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """캐시된 데이터 조회"""
        if not self.enable_cache:
            return None

        try:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                # 캐시 만료 확인 (24시간)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=24):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)

        except Exception as e:
            logger.warning(f"Failed to load cache for {url}: {e}")

        return None

    def save_to_cache(self, url: str, data: List[Dict[str, Any]]) -> None:
        """데이터를 캐시에 저장"""
        if not self.enable_cache:
            return

        try:
            cache_key = hashlib.md5(url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save cache for {url}: {e}")

    def cleanup_cache(self, max_age_days: int = 7) -> None:
        """오래된 캐시 파일 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

            for cache_file in self.cache_dir.glob("*.json"):
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                    cache_file.unlink()
                    logger.info(f"Removed old cache file: {cache_file.name}")

        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'session'):
            self.session.close()