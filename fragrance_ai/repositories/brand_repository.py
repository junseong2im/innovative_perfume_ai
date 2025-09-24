"""
브랜드 저장소

브랜드 데이터에 특화된 쿼리와 비즈니스 로직을 제공합니다.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_

from .base import BaseRepository
from ..database.models import Brand
from ..core.production_logging import get_logger

logger = get_logger(__name__)


class BrandRepository(BaseRepository[Brand]):
    """브랜드 저장소"""

    def __init__(self, session: Session):
        super().__init__(Brand, session)

    def find_by_name(self, name: str, exact: bool = False) -> List[Brand]:
        """이름으로 브랜드 검색"""
        try:
            query = self.session.query(Brand)

            if exact:
                query = query.filter(
                    or_(
                        Brand.name == name,
                        Brand.name_korean == name
                    )
                )
            else:
                search_pattern = f"%{name}%"
                query = query.filter(
                    or_(
                        Brand.name.ilike(search_pattern),
                        Brand.name_korean.ilike(search_pattern)
                    )
                )

            brands = query.order_by(Brand.name).all()
            logger.debug(f"Found {len(brands)} brands for name search: {name}")
            return brands

        except Exception as e:
            logger.error(f"Failed to search brands by name {name}: {str(e)}")
            raise

    def find_by_country(self, country: str) -> List[Brand]:
        """국가별 브랜드 조회"""
        return self.find_by(country=country)

    def find_by_brand_type(self, brand_type: str) -> List[Brand]:
        """브랜드 타입별 조회"""
        return self.find_by(brand_type=brand_type)

    def get_statistics(self) -> Dict[str, Any]:
        """브랜드 통계"""
        try:
            total_brands = self.count()

            # 국가별 분포
            country_distribution = {}
            results = self.session.query(
                Brand.country,
                func.count(Brand.id)
            ).group_by(Brand.country).all()

            for country, count in results:
                if country:
                    country_distribution[country] = count

            # 브랜드 타입별 분포
            type_distribution = {}
            results = self.session.query(
                Brand.brand_type,
                func.count(Brand.id)
            ).group_by(Brand.brand_type).all()

            for brand_type, count in results:
                if brand_type:
                    type_distribution[brand_type] = count

            return {
                'total_brands': total_brands,
                'country_distribution': country_distribution,
                'type_distribution': type_distribution
            }

        except Exception as e:
            logger.error(f"Failed to get brand statistics: {str(e)}")
            raise