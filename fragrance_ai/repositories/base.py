"""
Repository 패턴 기본 클래스

데이터 액세스 계층을 추상화하여 비즈니스 로직과 데이터 계층을 분리합니다.
이를 통해 테스트 용이성과 유지보수성을 향상시킵니다.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import SQLAlchemyError
import logging
from datetime import datetime

from ..core.production_logging import get_logger
from ..core.exceptions import DatabaseException, ErrorCode
from ..database.models import Base

logger = get_logger(__name__)

# 제네릭 타입 정의
T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T], ABC):
    """저장소 패턴의 기본 클래스"""

    def __init__(self, model_class: type[T], session: Optional[Session] = None):
        self.model_class = model_class
        self.session = session
        self._table_name = model_class.__tablename__

    @property
    def table_name(self) -> str:
        """테이블 이름 반환"""
        return self._table_name

    # ==========================================
    # 기본 CRUD 작업
    # ==========================================

    def create(self, **kwargs) -> T:
        """새로운 엔티티 생성"""
        try:
            entity = self.model_class(**kwargs)
            self.session.add(entity)
            self.session.flush()  # ID 생성을 위해

            logger.info(f"Created {self._table_name} entity with ID: {entity.id}")
            return entity

        except SQLAlchemyError as e:
            logger.error(f"Failed to create {self._table_name}: {str(e)}")
            self.session.rollback()
            raise DatabaseException(
                message=f"Failed to create {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def get_by_id(self, entity_id: str) -> Optional[T]:
        """ID로 엔티티 조회"""
        try:
            entity = self.session.query(self.model_class).filter(
                self.model_class.id == entity_id
            ).first()

            if entity:
                logger.info(f"Found {self._table_name} entity with ID: {entity_id}")
            else:
                logger.info(f"No {self._table_name} entity found with ID: {entity_id}")

            return entity

        except SQLAlchemyError as e:
            logger.error(f"Failed to get {self._table_name} by ID {entity_id}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to retrieve {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """모든 엔티티 조회"""
        try:
            query = self.session.query(self.model_class)

            if offset > 0:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            entities = query.all()
            logger.info(f"Retrieved {len(entities)} {self._table_name} entities")
            return entities

        except SQLAlchemyError as e:
            logger.error(f"Failed to get all {self._table_name}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to retrieve {self._table_name} list",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def update(self, entity_id: str, **kwargs) -> Optional[T]:
        """엔티티 업데이트"""
        try:
            entity = self.get_by_id(entity_id)
            if not entity:
                return None

            # 업데이트 가능한 필드만 적용
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            # 업데이트 시간 자동 설정
            if hasattr(entity, 'updated_at'):
                entity.updated_at = datetime.utcnow()

            self.session.flush()
            logger.debug(f"Updated {self._table_name} entity with ID: {entity_id}")
            return entity

        except SQLAlchemyError as e:
            logger.error(f"Failed to update {self._table_name} {entity_id}: {str(e)}")
            self.session.rollback()
            raise DatabaseException(
                message=f"Failed to update {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def delete(self, entity_id: str) -> bool:
        """엔티티 삭제"""
        try:
            entity = self.get_by_id(entity_id)
            if not entity:
                return False

            self.session.delete(entity)
            self.session.flush()
            logger.debug(f"Deleted {self._table_name} entity with ID: {entity_id}")
            return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to delete {self._table_name} {entity_id}: {str(e)}")
            self.session.rollback()
            raise DatabaseException(
                message=f"Failed to delete {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def exists(self, entity_id: str) -> bool:
        """엔티티 존재 여부 확인"""
        try:
            count = self.session.query(self.model_class).filter(
                self.model_class.id == entity_id
            ).count()
            return count > 0

        except SQLAlchemyError as e:
            logger.error(f"Failed to check existence of {self._table_name} {entity_id}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to check {self._table_name} existence",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """엔티티 개수 조회"""
        try:
            query = self.session.query(func.count(self.model_class.id))

            if filters:
                query = self._apply_filters(query, filters)

            count = query.scalar()
            logger.info(f"Counted {count} {self._table_name} entities")
            return count

        except SQLAlchemyError as e:
            logger.error(f"Failed to count {self._table_name}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to count {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    # ==========================================
    # 고급 쿼리 작업
    # ==========================================

    def find_by(self, **kwargs) -> List[T]:
        """조건으로 엔티티 검색"""
        try:
            query = self.session.query(self.model_class)

            for key, value in kwargs.items():
                if hasattr(self.model_class, key):
                    column = getattr(self.model_class, key)
                    query = query.filter(column == value)

            entities = query.all()
            logger.debug(f"Found {len(entities)} {self._table_name} entities with filters: {kwargs}")
            return entities

        except SQLAlchemyError as e:
            logger.error(f"Failed to find {self._table_name} by filters {kwargs}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to find {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def find_one_by(self, **kwargs) -> Optional[T]:
        """조건으로 단일 엔티티 검색"""
        entities = self.find_by(**kwargs)
        return entities[0] if entities else None

    def search(self,
               filters: Optional[Dict[str, Any]] = None,
               sort_by: Optional[str] = None,
               sort_desc: bool = False,
               limit: Optional[int] = None,
               offset: int = 0) -> List[T]:
        """고급 검색"""
        try:
            query = self.session.query(self.model_class)

            # 필터 적용
            if filters:
                query = self._apply_filters(query, filters)

            # 정렬 적용
            if sort_by and hasattr(self.model_class, sort_by):
                column = getattr(self.model_class, sort_by)
                query = query.order_by(desc(column) if sort_desc else asc(column))

            # 페이징 적용
            if offset > 0:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            entities = query.all()
            logger.debug(f"Search returned {len(entities)} {self._table_name} entities")
            return entities

        except SQLAlchemyError as e:
            logger.error(f"Failed to search {self._table_name}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to search {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def bulk_create(self, entities_data: List[Dict[str, Any]]) -> List[T]:
        """대량 생성"""
        try:
            entities = []
            for data in entities_data:
                entity = self.model_class(**data)
                entities.append(entity)
                self.session.add(entity)

            self.session.flush()
            logger.info(f"Bulk created {len(entities)} {self._table_name} entities")
            return entities

        except SQLAlchemyError as e:
            logger.error(f"Failed to bulk create {self._table_name}: {str(e)}")
            self.session.rollback()
            raise DatabaseException(
                message=f"Failed to bulk create {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """대량 업데이트"""
        try:
            updated_count = 0

            for update_data in updates:
                entity_id = update_data.pop('id', None)
                if not entity_id:
                    continue

                if self.update(entity_id, **update_data):
                    updated_count += 1

            logger.info(f"Bulk updated {updated_count} {self._table_name} entities")
            return updated_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to bulk update {self._table_name}: {str(e)}")
            self.session.rollback()
            raise DatabaseException(
                message=f"Failed to bulk update {self._table_name}",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    # ==========================================
    # 헬퍼 메서드
    # ==========================================

    def _apply_filters(self, query, filters: Dict[str, Any]):
        """필터 적용"""
        for key, value in filters.items():
            if not hasattr(self.model_class, key):
                continue

            column = getattr(self.model_class, key)

            if isinstance(value, dict):
                # 고급 필터 연산자
                for operator, operand in value.items():
                    if operator == 'eq':
                        query = query.filter(column == operand)
                    elif operator == 'ne':
                        query = query.filter(column != operand)
                    elif operator == 'gt':
                        query = query.filter(column > operand)
                    elif operator == 'gte':
                        query = query.filter(column >= operand)
                    elif operator == 'lt':
                        query = query.filter(column < operand)
                    elif operator == 'lte':
                        query = query.filter(column <= operand)
                    elif operator == 'like':
                        query = query.filter(column.like(f"%{operand}%"))
                    elif operator == 'ilike':
                        query = query.filter(column.ilike(f"%{operand}%"))
                    elif operator == 'in':
                        query = query.filter(column.in_(operand))
                    elif operator == 'not_in':
                        query = query.filter(~column.in_(operand))
            elif isinstance(value, list):
                # 리스트인 경우 IN 연산
                query = query.filter(column.in_(value))
            else:
                # 단순 동등 비교
                query = query.filter(column == value)

        return query

    def get_statistics(self) -> Dict[str, Any]:
        """기본 통계 정보 반환"""
        try:
            stats = {
                'total_count': self.count(),
                'table_name': self._table_name
            }

            # 타임스탬프가 있는 경우 생성 날짜 통계 추가
            if hasattr(self.model_class, 'created_at'):
                recent_count = self.session.query(func.count(self.model_class.id)).filter(
                    self.model_class.created_at >= func.date_sub(func.now(), '7 day')
                ).scalar()
                stats['recent_count_7days'] = recent_count

            return stats

        except SQLAlchemyError as e:
            logger.error(f"Failed to get statistics for {self._table_name}: {str(e)}")
            raise DatabaseException(
                message=f"Failed to get {self._table_name} statistics",
                cause=e,
                error_code=ErrorCode.DATABASE_ERROR
            )

    # ==========================================
    # 비동기 지원 (AsyncSession 사용 시)
    # ==========================================

    async def async_create(self, **kwargs) -> T:
        """비동기 생성"""
        # AsyncSession을 사용하는 경우의 구현
        # 현재는 동기 버전으로 구현
        return self.create(**kwargs)

    async def async_get_by_id(self, entity_id: str) -> Optional[T]:
        """비동기 ID 조회"""
        return self.get_by_id(entity_id)

    async def async_search(self, **kwargs) -> List[T]:
        """비동기 검색"""
        return self.search(**kwargs)


class ReadOnlyRepository(BaseRepository[T]):
    """읽기 전용 저장소 - 실제 구현"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._read_only_mode = True
        logger.warning("ReadOnlyRepository initialized - write operations will be logged but not executed")

    def create(self, **kwargs) -> T:
        """읽기 전용 모드에서 create 시도 시 경고 로그"""
        logger.error(f"Attempted CREATE operation in read-only mode: {kwargs}")

        # 감사 로그 기록
        self._log_audit_event("CREATE_ATTEMPT", kwargs)

        # 읽기 전용 예외를 발생시키되, 디버그 정보 포함
        raise PermissionError(
            f"Create operation not allowed in read-only repository. "
            f"Attempted to create entity with data: {list(kwargs.keys())}"
        )

    def update(self, entity_id: str, **kwargs) -> Optional[T]:
        """읽기 전용 모드에서 update 시도 시 경고 로그"""
        logger.error(f"Attempted UPDATE operation in read-only mode for ID: {entity_id}")

        # 감사 로그 기록
        self._log_audit_event("UPDATE_ATTEMPT", {"entity_id": entity_id, **kwargs})

        # 현재 엔티티 상태 확인 (읽기는 가능)
        current = self.get_by_id(entity_id)
        if current:
            logger.info(f"Current entity state: {current}")

        raise PermissionError(
            f"Update operation not allowed in read-only repository. "
            f"Attempted to update entity ID: {entity_id}"
        )

    def delete(self, entity_id: str) -> bool:
        """읽기 전용 모드에서 delete 시도 시 경고 로그"""
        logger.error(f"Attempted DELETE operation in read-only mode for ID: {entity_id}")

        # 감사 로그 기록
        self._log_audit_event("DELETE_ATTEMPT", {"entity_id": entity_id})

        # 삭제하려던 엔티티 정보 확인
        entity = self.get_by_id(entity_id)
        if entity:
            logger.warning(f"Attempted to delete existing entity: {entity}")

        raise PermissionError(
            f"Delete operation not allowed in read-only repository. "
            f"Attempted to delete entity ID: {entity_id}"
        )

    def bulk_create(self, entities_data: List[Dict[str, Any]]) -> List[T]:
        """읽기 전용 모드에서 bulk create 시도 시 경고 로그"""
        logger.error(f"Attempted BULK CREATE operation in read-only mode: {len(entities_data)} entities")

        # 감사 로그 기록
        self._log_audit_event("BULK_CREATE_ATTEMPT", {"count": len(entities_data)})

        raise PermissionError(
            f"Bulk create operation not allowed in read-only repository. "
            f"Attempted to create {len(entities_data)} entities"
        )

    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """읽기 전용 모드에서 bulk update 시도 시 경고 로그"""
        logger.error(f"Attempted BULK UPDATE operation in read-only mode: {len(updates)} updates")

        # 감사 로그 기록
        self._log_audit_event("BULK_UPDATE_ATTEMPT", {"count": len(updates)})

        raise PermissionError(
            f"Bulk update operation not allowed in read-only repository. "
            f"Attempted to update {len(updates)} entities"
        )

    def _log_audit_event(self, event_type: str, data: Dict[str, Any]):
        """감사 로그 기록"""
        audit_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "repository": self.__class__.__name__,
            "data_summary": data,
            "user": self._get_current_user(),
            "ip_address": self._get_client_ip()
        }

        # 감사 로그를 파일이나 데이터베이스에 기록
        logger.info(f"AUDIT: {json.dumps(audit_log)}")

        # 파일에도 기록
        try:
            audit_file = "audit_logs/repository_access.log"
            os.makedirs(os.path.dirname(audit_file), exist_ok=True)
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_log) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def _get_current_user(self) -> str:
        """현재 사용자 정보 가져오기"""
        # 실제로는 인증 컨텍스트에서 가져옴
        return os.environ.get('USER', 'unknown')

    def _get_client_ip(self) -> str:
        """클라이언트 IP 주소 가져오기"""
        # 실제로는 요청 컨텍스트에서 가져옴
        return "127.0.0.1"