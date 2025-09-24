"""
향료 노트 저장소

향료 노트 데이터에 특화된 쿼리와 비즈니스 로직을 제공합니다.
"""

from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from .base import BaseRepository
from ..database.models import FragranceNote, RecipeIngredient
from ..core.production_logging import get_logger

logger = get_logger(__name__)


class FragranceNoteRepository(BaseRepository[FragranceNote]):
    """향료 노트 저장소"""

    def __init__(self, session: Session):
        super().__init__(FragranceNote, session)

    # ==========================================
    # 향료 노트 특화 쿼리
    # ==========================================

    def find_by_name(self, name: str, exact: bool = False) -> List[FragranceNote]:
        """이름으로 향료 노트 검색"""
        try:
            query = self.session.query(FragranceNote)

            if exact:
                # 정확한 이름 매칭
                query = query.filter(
                    or_(
                        FragranceNote.name == name,
                        FragranceNote.name_korean == name,
                        FragranceNote.name_english == name
                    )
                )
            else:
                # 부분 매칭
                search_pattern = f"%{name}%"
                query = query.filter(
                    or_(
                        FragranceNote.name.ilike(search_pattern),
                        FragranceNote.name_korean.ilike(search_pattern),
                        FragranceNote.name_english.ilike(search_pattern),
                        FragranceNote.search_keywords.ilike(search_pattern)
                    )
                )

            notes = query.order_by(FragranceNote.name).all()
            logger.info(f"Found {len(notes)} notes for name search: {name}")
            return notes

        except Exception as e:
            logger.error(f"Failed to search notes by name {name}: {str(e)}")
            raise

    def find_by_fragrance_family(self, family: str) -> List[FragranceNote]:
        """향족으로 노트 검색"""
        return self.find_by(fragrance_family=family)

    def find_by_note_type(self, note_type: str) -> List[FragranceNote]:
        """노트 타입으로 검색"""
        return self.find_by(note_type=note_type)

    def find_by_characteristics(self,
                              intensity_min: Optional[float] = None,
                              intensity_max: Optional[float] = None,
                              longevity_min: Optional[float] = None,
                              longevity_max: Optional[float] = None,
                              sillage_min: Optional[float] = None,
                              sillage_max: Optional[float] = None) -> List[FragranceNote]:
        """향료 특성으로 검색"""
        try:
            query = self.session.query(FragranceNote)

            # 강도 필터
            if intensity_min is not None:
                query = query.filter(FragranceNote.intensity >= intensity_min)
            if intensity_max is not None:
                query = query.filter(FragranceNote.intensity <= intensity_max)

            # 지속성 필터
            if longevity_min is not None:
                query = query.filter(FragranceNote.longevity >= longevity_min)
            if longevity_max is not None:
                query = query.filter(FragranceNote.longevity <= longevity_max)

            # 확산성 필터
            if sillage_min is not None:
                query = query.filter(FragranceNote.sillage >= sillage_min)
            if sillage_max is not None:
                query = query.filter(FragranceNote.sillage <= sillage_max)

            notes = query.order_by(FragranceNote.intensity.desc()).all()
            logger.info(f"Found {len(notes)} notes by characteristics")
            return notes

        except Exception as e:
            logger.error(f"Failed to search notes by characteristics: {str(e)}")
            raise

    def find_by_tags(self,
                     mood_tags: Optional[List[str]] = None,
                     season_tags: Optional[List[str]] = None,
                     gender_tags: Optional[List[str]] = None,
                     match_all: bool = False) -> List[FragranceNote]:
        """태그로 노트 검색"""
        try:
            query = self.session.query(FragranceNote)
            conditions = []

            # 무드 태그 필터
            if mood_tags:
                if match_all:
                    # 모든 태그가 포함되어야 함
                    for tag in mood_tags:
                        conditions.append(
                            FragranceNote.mood_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    # 하나 이상의 태그가 포함되면 됨
                    mood_conditions = [
                        FragranceNote.mood_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in mood_tags
                    ]
                    conditions.append(or_(*mood_conditions))

            # 시즌 태그 필터
            if season_tags:
                if match_all:
                    for tag in season_tags:
                        conditions.append(
                            FragranceNote.season_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    season_conditions = [
                        FragranceNote.season_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in season_tags
                    ]
                    conditions.append(or_(*season_conditions))

            # 성별 태그 필터
            if gender_tags:
                if match_all:
                    for tag in gender_tags:
                        conditions.append(
                            FragranceNote.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        )
                else:
                    gender_conditions = [
                        FragranceNote.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                        for tag in gender_tags
                    ]
                    conditions.append(or_(*gender_conditions))

            # 조건 적용
            if conditions:
                if match_all:
                    query = query.filter(and_(*conditions))
                else:
                    query = query.filter(or_(*conditions))

            notes = query.order_by(FragranceNote.name).all()
            logger.info(f"Found {len(notes)} notes by tags")
            return notes

        except Exception as e:
            logger.error(f"Failed to search notes by tags: {str(e)}")
            raise

    def find_popular_notes(self, limit: int = 20) -> List[Tuple[FragranceNote, int]]:
        """인기 노트 조회 (레시피에서 많이 사용되는 순)"""
        try:
            # 레시피에서 사용 빈도 계산
            query = self.session.query(
                FragranceNote,
                func.count(RecipeIngredient.id).label('usage_count')
            ).outerjoin(
                RecipeIngredient, FragranceNote.id == RecipeIngredient.note_id
            ).group_by(
                FragranceNote.id
            ).order_by(
                func.count(RecipeIngredient.id).desc()
            ).limit(limit)

            results = query.all()
            logger.info(f"Found {len(results)} popular notes")
            return [(note, count) for note, count in results]

        except Exception as e:
            logger.error(f"Failed to get popular notes: {str(e)}")
            raise

    def get_notes_by_price_range(self,
                                min_price: Optional[float] = None,
                                max_price: Optional[float] = None) -> List[FragranceNote]:
        """가격 범위로 노트 조회"""
        try:
            query = self.session.query(FragranceNote).filter(
                FragranceNote.price_per_ml.is_not(None)
            )

            if min_price is not None:
                query = query.filter(FragranceNote.price_per_ml >= min_price)
            if max_price is not None:
                query = query.filter(FragranceNote.price_per_ml <= max_price)

            notes = query.order_by(FragranceNote.price_per_ml.asc()).all()
            logger.info(f"Found {len(notes)} notes in price range")
            return notes

        except Exception as e:
            logger.error(f"Failed to search notes by price range: {str(e)}")
            raise

    def get_notes_by_supplier(self, supplier: str) -> List[FragranceNote]:
        """공급업체로 노트 조회"""
        return self.find_by(supplier=supplier)

    def get_notes_by_grade(self, grade: str) -> List[FragranceNote]:
        """등급으로 노트 조회"""
        return self.find_by(grade=grade)

    def find_similar_notes(self, note_id: str, limit: int = 10) -> List[FragranceNote]:
        """유사한 노트 찾기"""
        try:
            # 기준 노트 조회
            base_note = self.get_by_id(note_id)
            if not base_note:
                return []

            # 같은 향족이면서 비슷한 특성을 가진 노트 찾기
            intensity_range = 1.0
            longevity_range = 1.0
            sillage_range = 1.0

            query = self.session.query(FragranceNote).filter(
                and_(
                    FragranceNote.id != note_id,
                    FragranceNote.fragrance_family == base_note.fragrance_family,
                    FragranceNote.note_type == base_note.note_type,
                    FragranceNote.intensity.between(
                        base_note.intensity - intensity_range,
                        base_note.intensity + intensity_range
                    ),
                    FragranceNote.longevity.between(
                        base_note.longevity - longevity_range,
                        base_note.longevity + longevity_range
                    ),
                    FragranceNote.sillage.between(
                        base_note.sillage - sillage_range,
                        base_note.sillage + sillage_range
                    )
                )
            ).order_by(
                func.abs(FragranceNote.intensity - base_note.intensity) +
                func.abs(FragranceNote.longevity - base_note.longevity) +
                func.abs(FragranceNote.sillage - base_note.sillage)
            ).limit(limit)

            similar_notes = query.all()
            logger.info(f"Found {len(similar_notes)} similar notes for {note_id}")
            return similar_notes

        except Exception as e:
            logger.error(f"Failed to find similar notes for {note_id}: {str(e)}")
            raise

    # ==========================================
    # 통계 및 분석
    # ==========================================

    def get_fragrance_family_distribution(self) -> Dict[str, int]:
        """향족별 분포 통계"""
        try:
            results = self.session.query(
                FragranceNote.fragrance_family,
                func.count(FragranceNote.id)
            ).group_by(
                FragranceNote.fragrance_family
            ).all()

            distribution = {family: count for family, count in results if family}
            logger.info(f"Fragrance family distribution: {distribution}")
            return distribution

        except Exception as e:
            logger.error(f"Failed to get fragrance family distribution: {str(e)}")
            raise

    def get_note_type_distribution(self) -> Dict[str, int]:
        """노트 타입별 분포 통계"""
        try:
            results = self.session.query(
                FragranceNote.note_type,
                func.count(FragranceNote.id)
            ).group_by(
                FragranceNote.note_type
            ).all()

            distribution = {note_type: count for note_type, count in results}
            logger.info(f"Note type distribution: {distribution}")
            return distribution

        except Exception as e:
            logger.error(f"Failed to get note type distribution: {str(e)}")
            raise

    def get_characteristics_statistics(self) -> Dict[str, Dict[str, float]]:
        """특성별 통계 (평균, 최대, 최소)"""
        try:
            result = self.session.query(
                func.avg(FragranceNote.intensity).label('avg_intensity'),
                func.min(FragranceNote.intensity).label('min_intensity'),
                func.max(FragranceNote.intensity).label('max_intensity'),
                func.avg(FragranceNote.longevity).label('avg_longevity'),
                func.min(FragranceNote.longevity).label('min_longevity'),
                func.max(FragranceNote.longevity).label('max_longevity'),
                func.avg(FragranceNote.sillage).label('avg_sillage'),
                func.min(FragranceNote.sillage).label('min_sillage'),
                func.max(FragranceNote.sillage).label('max_sillage')
            ).first()

            stats = {
                'intensity': {
                    'average': float(result.avg_intensity or 0),
                    'min': float(result.min_intensity or 0),
                    'max': float(result.max_intensity or 0)
                },
                'longevity': {
                    'average': float(result.avg_longevity or 0),
                    'min': float(result.min_longevity or 0),
                    'max': float(result.max_longevity or 0)
                },
                'sillage': {
                    'average': float(result.avg_sillage or 0),
                    'min': float(result.min_sillage or 0),
                    'max': float(result.max_sillage or 0)
                }
            }

            logger.info("Generated characteristics statistics")
            return stats

        except Exception as e:
            logger.error(f"Failed to get characteristics statistics: {str(e)}")
            raise

    # ==========================================
    # 유틸리티 메서드
    # ==========================================

    def create_note_with_validation(self, **kwargs) -> FragranceNote:
        """검증이 포함된 노트 생성"""
        # 필수 필드 검증
        required_fields = ['name', 'note_type', 'fragrance_family']
        for field in required_fields:
            if not kwargs.get(field):
                raise ValueError(f"Required field '{field}' is missing")

        # 노트 타입 검증
        valid_note_types = ['top', 'middle', 'base']
        if kwargs['note_type'] not in valid_note_types:
            raise ValueError(f"Invalid note_type. Must be one of: {valid_note_types}")

        # 특성 값 범위 검증
        for characteristic in ['intensity', 'longevity', 'sillage']:
            value = kwargs.get(characteristic)
            if value is not None and (value < 1 or value > 10):
                raise ValueError(f"{characteristic} must be between 1 and 10")

        # 중복 이름 확인
        existing = self.find_by_name(kwargs['name'], exact=True)
        if existing:
            raise ValueError(f"Note with name '{kwargs['name']}' already exists")

        return self.create(**kwargs)

    def search_advanced(self,
                       query_text: Optional[str] = None,
                       families: Optional[List[str]] = None,
                       note_types: Optional[List[str]] = None,
                       intensity_range: Optional[Tuple[float, float]] = None,
                       longevity_range: Optional[Tuple[float, float]] = None,
                       sillage_range: Optional[Tuple[float, float]] = None,
                       tags: Optional[List[str]] = None,
                       price_range: Optional[Tuple[float, float]] = None,
                       limit: int = 50,
                       offset: int = 0) -> List[FragranceNote]:
        """고급 통합 검색"""
        try:
            query = self.session.query(FragranceNote)

            # 텍스트 검색
            if query_text:
                search_pattern = f"%{query_text}%"
                query = query.filter(
                    or_(
                        FragranceNote.name.ilike(search_pattern),
                        FragranceNote.name_korean.ilike(search_pattern),
                        FragranceNote.name_english.ilike(search_pattern),
                        FragranceNote.description.ilike(search_pattern),
                        FragranceNote.search_keywords.ilike(search_pattern)
                    )
                )

            # 향족 필터
            if families:
                query = query.filter(FragranceNote.fragrance_family.in_(families))

            # 노트 타입 필터
            if note_types:
                query = query.filter(FragranceNote.note_type.in_(note_types))

            # 특성 범위 필터
            if intensity_range:
                query = query.filter(
                    FragranceNote.intensity.between(intensity_range[0], intensity_range[1])
                )
            if longevity_range:
                query = query.filter(
                    FragranceNote.longevity.between(longevity_range[0], longevity_range[1])
                )
            if sillage_range:
                query = query.filter(
                    FragranceNote.sillage.between(sillage_range[0], sillage_range[1])
                )

            # 가격 범위 필터
            if price_range:
                query = query.filter(
                    and_(
                        FragranceNote.price_per_ml >= price_range[0],
                        FragranceNote.price_per_ml <= price_range[1]
                    )
                )

            # 태그 필터
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.extend([
                        FragranceNote.mood_tags.op('JSON_CONTAINS')(f'"{tag}"'),
                        FragranceNote.season_tags.op('JSON_CONTAINS')(f'"{tag}"'),
                        FragranceNote.gender_tags.op('JSON_CONTAINS')(f'"{tag}"')
                    ])
                query = query.filter(or_(*tag_conditions))

            # 정렬 및 페이징
            query = query.order_by(FragranceNote.name).offset(offset).limit(limit)

            notes = query.all()
            logger.info(f"Advanced search returned {len(notes)} notes")
            return notes

        except Exception as e:
            logger.error(f"Failed to perform advanced search: {str(e)}")
            raise