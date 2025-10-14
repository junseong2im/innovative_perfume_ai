"""
Redis Metadata Manager with TTL
모든 메타키에 TTL을 적용하여 자동 만료 관리
"""

import json
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import redis.asyncio as redis
from loguru import logger


@dataclass
class MetadataEntry:
    """메타데이터 엔트리"""
    cid: str  # IPFS CID
    size: int  # 바이트 단위
    data_type: str  # feedback, checkpoint, recipe, etc.
    timestamp: str
    tags: list
    checksum: Optional[str] = None


class RedisMetadataManager:
    """
    Redis 메타데이터 관리자 (TTL 자동 적용)
    """

    # TTL 설정 (초 단위)
    TTL_CONFIG = {
        "feedback": 86400 * 30,      # 30일
        "checkpoint": 86400 * 90,    # 90일
        "recipe": 86400 * 365,       # 1년
        "session": 86400,            # 1일
        "cache": 3600,               # 1시간
        "temp": 1800,                # 30분
        "default": 86400 * 7         # 7일 (기본값)
    }

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.client = None
        logger.info(f"RedisMetadataManager initialized (url={redis_url})")

    async def connect(self):
        """Redis 연결"""
        if not self.client:
            self.client = await redis.from_url(self.redis_url)
            logger.info("Connected to Redis")

    async def disconnect(self):
        """Redis 연결 종료"""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis")

    def get_ttl(self, data_type: str) -> int:
        """
        데이터 타입에 따른 TTL 반환

        Args:
            data_type: 데이터 타입

        Returns:
            TTL (초)
        """
        return self.TTL_CONFIG.get(data_type, self.TTL_CONFIG["default"])

    async def store_metadata(
        self,
        key: str,
        metadata: MetadataEntry,
        custom_ttl: Optional[int] = None
    ) -> bool:
        """
        메타데이터 저장 (TTL 자동 적용)

        Args:
            key: Redis 키
            metadata: 메타데이터 엔트리
            custom_ttl: 커스텀 TTL (초)

        Returns:
            성공 여부
        """
        await self.connect()

        try:
            # 메타데이터를 딕셔너리로 변환
            data = asdict(metadata)

            # Redis HSET으로 저장
            await self.client.hset(
                f"metadata:{key}",
                mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()}
            )

            # TTL 설정
            ttl = custom_ttl if custom_ttl else self.get_ttl(metadata.data_type)
            await self.client.expire(f"metadata:{key}", ttl)

            logger.info(
                f"Metadata stored: {key} (type={metadata.data_type}, ttl={ttl}s)"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to store metadata: {e}")
            return False

    async def get_metadata(self, key: str) -> Optional[MetadataEntry]:
        """
        메타데이터 조회

        Args:
            key: Redis 키

        Returns:
            MetadataEntry or None
        """
        await self.connect()

        try:
            data = await self.client.hgetall(f"metadata:{key}")

            if not data:
                logger.warning(f"Metadata not found: {key}")
                return None

            # 바이트를 문자열로 변환
            decoded = {
                k.decode(): v.decode() for k, v in data.items()
            }

            # tags를 JSON에서 파싱
            if 'tags' in decoded:
                decoded['tags'] = json.loads(decoded['tags'])

            # size를 int로 변환
            if 'size' in decoded:
                decoded['size'] = int(decoded['size'])

            return MetadataEntry(**decoded)

        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None

    async def delete_metadata(self, key: str) -> bool:
        """
        메타데이터 삭제

        Args:
            key: Redis 키

        Returns:
            성공 여부
        """
        await self.connect()

        try:
            result = await self.client.delete(f"metadata:{key}")
            logger.info(f"Metadata deleted: {key}")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete metadata: {e}")
            return False

    async def get_ttl_remaining(self, key: str) -> Optional[int]:
        """
        남은 TTL 조회

        Args:
            key: Redis 키

        Returns:
            남은 시간 (초) or None
        """
        await self.connect()

        try:
            ttl = await self.client.ttl(f"metadata:{key}")

            if ttl == -2:
                logger.warning(f"Key does not exist: {key}")
                return None
            elif ttl == -1:
                logger.warning(f"Key has no TTL: {key}")
                return None

            return ttl

        except Exception as e:
            logger.error(f"Failed to get TTL: {e}")
            return None

    async def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """
        TTL 연장

        Args:
            key: Redis 키
            additional_seconds: 추가할 시간 (초)

        Returns:
            성공 여부
        """
        await self.connect()

        try:
            current_ttl = await self.client.ttl(f"metadata:{key}")

            if current_ttl <= 0:
                logger.warning(f"Cannot extend TTL for key: {key}")
                return False

            new_ttl = current_ttl + additional_seconds
            result = await self.client.expire(f"metadata:{key}", new_ttl)

            logger.info(f"TTL extended: {key} (+{additional_seconds}s, new={new_ttl}s)")

            return result

        except Exception as e:
            logger.error(f"Failed to extend TTL: {e}")
            return False

    async def list_by_type(self, data_type: str, limit: int = 100) -> list:
        """
        타입별 메타데이터 목록 조회

        Args:
            data_type: 데이터 타입
            limit: 최대 개수

        Returns:
            메타데이터 키 목록
        """
        await self.connect()

        try:
            # SCAN으로 모든 metadata 키 검색
            keys = []
            cursor = 0

            while True:
                cursor, partial_keys = await self.client.scan(
                    cursor,
                    match="metadata:*",
                    count=100
                )

                for key in partial_keys:
                    key_str = key.decode()

                    # 타입 확인
                    metadata = await self.get_metadata(key_str.replace("metadata:", ""))
                    if metadata and metadata.data_type == data_type:
                        keys.append(key_str)

                        if len(keys) >= limit:
                            break

                if cursor == 0 or len(keys) >= limit:
                    break

            logger.info(f"Found {len(keys)} metadata entries of type '{data_type}'")

            return keys

        except Exception as e:
            logger.error(f"Failed to list metadata: {e}")
            return []

    async def cleanup_expired(self) -> int:
        """
        만료된 키 정리 (수동 정리)

        Returns:
            정리된 키 개수
        """
        await self.connect()

        try:
            deleted_count = 0
            cursor = 0

            while True:
                cursor, keys = await self.client.scan(
                    cursor,
                    match="metadata:*",
                    count=100
                )

                for key in keys:
                    ttl = await self.client.ttl(key)

                    # TTL이 0 이하면 삭제 (이미 만료됨)
                    if ttl <= 0:
                        await self.client.delete(key)
                        deleted_count += 1

                if cursor == 0:
                    break

            logger.info(f"Cleaned up {deleted_count} expired metadata entries")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired metadata: {e}")
            return 0


# CLI 진입점
async def main():
    """메인 함수"""
    manager = RedisMetadataManager()

    # 테스트 메타데이터 저장
    test_metadata = MetadataEntry(
        cid="QmTest123456",
        size=1024,
        data_type="feedback",
        timestamp=datetime.now().isoformat(),
        tags=["rlhf", "qwen"],
        checksum="abc123"
    )

    print("=== Storing metadata ===")
    success = await manager.store_metadata("test_001", test_metadata)
    print(f"Store success: {success}")

    print("\n=== Getting metadata ===")
    retrieved = await manager.get_metadata("test_001")
    print(f"Retrieved: {retrieved}")

    print("\n=== Getting TTL ===")
    ttl = await manager.get_ttl_remaining("test_001")
    print(f"TTL remaining: {ttl}s (~{ttl/86400:.1f} days)")

    print("\n=== Extending TTL ===")
    extended = await manager.extend_ttl("test_001", 3600)  # +1시간
    print(f"Extended: {extended}")

    new_ttl = await manager.get_ttl_remaining("test_001")
    print(f"New TTL: {new_ttl}s (~{new_ttl/86400:.1f} days)")

    await manager.disconnect()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
