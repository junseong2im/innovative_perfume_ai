"""
Web Scraping Operator
향수 정보와 리뷰를 웹에서 수집하는 실제 오퍼레이터

수집 대상:
1. 향수 리뷰 사이트 (Fragrantica, Basenotes 스타일)
2. 향수 브랜드 공식 사이트
3. 소셜 미디어 향수 리뷰
4. 향수 성분 데이터베이스
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import time

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """데이터 소스 타입"""
    FRAGRANCE_REVIEW = "fragrance_review"
    BRAND_OFFICIAL = "brand_official"
    SOCIAL_MEDIA = "social_media"
    INGREDIENT_DB = "ingredient_database"
    RETAILER = "retailer"


@dataclass
class FragranceData:
    """수집된 향수 데이터"""
    source: str
    source_type: DataSource
    url: str
    collected_at: str

    # 기본 정보
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    gender: Optional[str] = None
    release_year: Optional[int] = None

    # 향 구성
    top_notes: List[str] = None
    heart_notes: List[str] = None
    base_notes: List[str] = None
    accords: List[str] = None

    # 특성
    longevity: Optional[float] = None  # 1-5 scale
    sillage: Optional[float] = None   # 1-5 scale
    season: List[str] = None
    occasion: List[str] = None

    # 리뷰 데이터
    rating: Optional[float] = None
    review_count: Optional[int] = None
    reviews: List[Dict] = None

    # 가격 정보
    price: Optional[Dict[str, float]] = None  # {size: price}

    # 메타데이터
    data_hash: Optional[str] = None
    confidence_score: Optional[float] = None


class WebScrapingOperator:
    """
    웹에서 향수 데이터를 수집하는 실제 오퍼레이터
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.session = None
        self.collected_data = []
        self.rate_limiter = RateLimiter()

        # 수집 통계
        self.stats = {
            'total_collected': 0,
            'successful': 0,
            'failed': 0,
            'duplicates': 0,
            'by_source': {}
        }

        # 중복 체크를 위한 해시 세트
        self.seen_hashes = set()

        logger.info("WebScrapingOperator initialized")

    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'user_agent': 'FragranceAI/1.0 (Educational Purpose Only)',
            'timeout': 30,
            'max_retries': 3,
            'concurrent_requests': 5,
            'rate_limit': {
                'requests_per_minute': 60,
                'burst_size': 10
            },
            'sources': [
                {
                    'name': 'FragranceDB',
                    'type': DataSource.INGREDIENT_DB,
                    'base_url': 'https://example-fragrance-db.com',
                    'enabled': True
                },
                {
                    'name': 'PerfumeReviews',
                    'type': DataSource.FRAGRANCE_REVIEW,
                    'base_url': 'https://example-reviews.com',
                    'enabled': True
                }
            ]
        }

    async def initialize(self):
        """비동기 세션 초기화"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=self.config['concurrent_requests'])
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers={'User-Agent': self.config['user_agent']}
            )

    async def close(self):
        """세션 정리"""
        if self.session:
            await self.session.close()

    async def scrape_fragrance_reviews(self, url: str) -> Optional[FragranceData]:
        """향수 리뷰 사이트에서 데이터 수집"""
        try:
            # Rate limiting
            await self.rate_limiter.acquire()

            async with self.session.get(url, timeout=self.config['timeout']) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # 실제 파싱 로직 (사이트 구조에 따라 조정 필요)
                data = FragranceData(
                    source='FragranceReviews',
                    source_type=DataSource.FRAGRANCE_REVIEW,
                    url=url,
                    collected_at=datetime.now().isoformat()
                )

                # 향수 이름
                name_elem = soup.find('h1', class_='fragrance-name')
                if name_elem:
                    data.name = name_elem.text.strip()

                # 브랜드
                brand_elem = soup.find('span', class_='brand')
                if brand_elem:
                    data.brand = brand_elem.text.strip()

                # 노트 추출
                notes_section = soup.find('div', class_='pyramid')
                if notes_section:
                    data.top_notes = self._extract_notes(notes_section, 'top')
                    data.heart_notes = self._extract_notes(notes_section, 'heart')
                    data.base_notes = self._extract_notes(notes_section, 'base')

                # 평점
                rating_elem = soup.find('span', class_='rating')
                if rating_elem:
                    try:
                        data.rating = float(rating_elem.text.strip())
                    except ValueError:
                        pass

                # 리뷰 수집
                reviews = []
                review_elements = soup.find_all('div', class_='review', limit=10)
                for review_elem in review_elements:
                    review_text = review_elem.find('p', class_='review-text')
                    review_rating = review_elem.find('span', class_='stars')
                    if review_text:
                        reviews.append({
                            'text': review_text.text.strip(),
                            'rating': self._parse_stars(review_rating) if review_rating else None
                        })
                data.reviews = reviews

                # 데이터 해시 생성 (중복 체크용)
                data.data_hash = self._generate_hash(data)

                # 신뢰도 점수 계산
                data.confidence_score = self._calculate_confidence(data)

                return data

        except asyncio.TimeoutError:
            logger.error(f"Timeout while scraping {url}")
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")

        return None

    async def scrape_ingredient_database(self, ingredient_name: str) -> Optional[Dict]:
        """성분 데이터베이스에서 정보 수집"""
        try:
            await self.rate_limiter.acquire()

            # 성분 정보 API 호출 (예시)
            api_url = f"https://api.cosmeticsingredients.com/v1/ingredients/{ingredient_name}"

            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()

                    return {
                        'ingredient': ingredient_name,
                        'cas_number': data.get('cas_number'),
                        'molecular_weight': data.get('molecular_weight'),
                        'odor_description': data.get('odor_description'),
                        'odor_threshold': data.get('odor_threshold'),
                        'volatility': data.get('volatility'),
                        'safety_data': data.get('safety'),
                        'natural_occurrence': data.get('natural_sources'),
                        'collected_at': datetime.now().isoformat()
                    }

        except Exception as e:
            logger.error(f"Error fetching ingredient {ingredient_name}: {e}")

        return None

    async def scrape_brand_catalog(self, brand_url: str) -> List[FragranceData]:
        """브랜드 공식 사이트에서 카탈로그 수집"""
        collected = []

        try:
            await self.rate_limiter.acquire()

            async with self.session.get(brand_url) as response:
                if response.status != 200:
                    return collected

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # 제품 링크 추출
                product_links = soup.find_all('a', class_='product-link')

                for link in product_links[:10]:  # 최대 10개만
                    product_url = urljoin(brand_url, link.get('href'))
                    product_data = await self._scrape_product_page(product_url)
                    if product_data:
                        collected.append(product_data)

        except Exception as e:
            logger.error(f"Error scraping brand catalog: {e}")

        return collected

    async def _scrape_product_page(self, url: str) -> Optional[FragranceData]:
        """개별 제품 페이지 스크래핑"""
        try:
            await self.rate_limiter.acquire()

            async with self.session.get(url) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                data = FragranceData(
                    source=urlparse(url).netloc,
                    source_type=DataSource.BRAND_OFFICIAL,
                    url=url,
                    collected_at=datetime.now().isoformat()
                )

                # 제품 정보 파싱 (실제 구조에 맞게 수정 필요)
                # ...

                return data

        except Exception as e:
            logger.error(f"Error scraping product page {url}: {e}")

        return None

    def _extract_notes(self, soup_element, note_type: str) -> List[str]:
        """향 노트 추출"""
        notes = []
        note_container = soup_element.find('div', class_=f'{note_type}-notes')
        if note_container:
            note_elements = note_container.find_all('span', class_='note')
            notes = [elem.text.strip() for elem in note_elements]
        return notes

    def _parse_stars(self, stars_element) -> float:
        """별점 파싱"""
        try:
            # 실제 구현은 사이트 구조에 따라 다름
            stars_text = stars_element.get('data-rating', '')
            return float(stars_text)
        except:
            return 0.0

    def _generate_hash(self, data: FragranceData) -> str:
        """데이터 해시 생성"""
        # 주요 필드로 해시 생성
        hash_input = f"{data.name}:{data.brand}:{data.source}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _calculate_confidence(self, data: FragranceData) -> float:
        """데이터 신뢰도 점수 계산"""
        score = 0.0
        max_score = 0.0

        # 필수 필드 체크
        if data.name:
            score += 2.0
        max_score += 2.0

        if data.brand:
            score += 2.0
        max_score += 2.0

        # 노트 정보
        if data.top_notes:
            score += 1.0
        if data.heart_notes:
            score += 1.0
        if data.base_notes:
            score += 1.0
        max_score += 3.0

        # 리뷰 정보
        if data.reviews and len(data.reviews) > 0:
            score += min(len(data.reviews) / 10, 1.0)
        max_score += 1.0

        # 평점
        if data.rating is not None:
            score += 1.0
        max_score += 1.0

        return score / max_score if max_score > 0 else 0.0

    async def run(self, urls: List[str] = None) -> Dict[str, Any]:
        """스크래핑 실행"""
        await self.initialize()

        try:
            if not urls:
                # 설정에서 URL 로드
                urls = self._get_urls_from_config()

            tasks = []
            for url in urls:
                # URL 타입에 따라 적절한 스크래퍼 선택
                if 'review' in url:
                    tasks.append(self.scrape_fragrance_reviews(url))
                elif 'brand' in url:
                    tasks.append(self.scrape_brand_catalog(url))
                # ... 추가 타입

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for result in results:
                if isinstance(result, FragranceData):
                    if result.data_hash not in self.seen_hashes:
                        self.collected_data.append(result)
                        self.seen_hashes.add(result.data_hash)
                        self.stats['successful'] += 1
                    else:
                        self.stats['duplicates'] += 1
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, FragranceData) and item.data_hash not in self.seen_hashes:
                            self.collected_data.append(item)
                            self.seen_hashes.add(item.data_hash)
                            self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

            self.stats['total_collected'] = len(self.collected_data)

            return {
                'status': 'completed',
                'stats': self.stats,
                'data_count': len(self.collected_data),
                'timestamp': datetime.now().isoformat()
            }

        finally:
            await self.close()

    def _get_urls_from_config(self) -> List[str]:
        """설정에서 URL 목록 생성"""
        urls = []
        for source in self.config.get('sources', []):
            if source.get('enabled'):
                # 실제로는 사이트맵이나 API에서 URL 목록을 가져옴
                base_url = source.get('base_url')
                if source['type'] == DataSource.FRAGRANCE_REVIEW:
                    # 리뷰 페이지 URL 생성
                    urls.extend([
                        f"{base_url}/perfume/1",
                        f"{base_url}/perfume/2",
                        # ...
                    ])
                elif source['type'] == DataSource.BRAND_OFFICIAL:
                    # 브랜드 카탈로그 URL
                    urls.append(f"{base_url}/catalog")

        return urls

    def save_to_file(self, filepath: str):
        """수집된 데이터를 파일로 저장"""
        data_dicts = [asdict(d) for d in self.collected_data]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'collected_at': datetime.now().isoformat(),
                    'stats': self.stats,
                    'source': 'WebScrapingOperator'
                },
                'data': data_dicts
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(data_dicts)} items to {filepath}")


class RateLimiter:
    """요청 속도 제한"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """요청 전 대기"""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)

            self.last_request_time = time.time()


# 실행 예시
async def main():
    """웹 스크래핑 실행"""
    scraper = WebScrapingOperator()

    # 테스트 URL들
    test_urls = [
        "https://example.com/perfume/chanel-no5",
        "https://example.com/perfume/dior-sauvage",
    ]

    result = await scraper.run(test_urls)
    print(f"Scraping completed: {result}")

    # 데이터 저장
    scraper.save_to_file('collected_fragrance_data.json')


if __name__ == "__main__":
    asyncio.run(main())