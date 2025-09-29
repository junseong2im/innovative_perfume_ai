"""
캐시 서비스 모듈
"""

from typing import Any, Optional, Dict
import hashlib
import json
import time
from datetime import datetime, timedelta

class CacheService:
    """간단한 인메모리 캐시 서비스"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = 3600  # 1시간 기본 TTL

    def _generate_key(self, data: Any) -> str:
        """데이터로부터 캐시 키 생성"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """캐시에서 값 조회"""
        if key in self.cache:
            cached_item = self.cache[key]
            # TTL 체크
            if time.time() < cached_item['expires_at']:
                return cached_item['value']
            else:
                # 만료된 항목 제거
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """캐시에 값 저장"""
        expiry_ttl = ttl if ttl is not None else self.ttl
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + expiry_ttl,
            'created_at': time.time()
        }

    def invalidate(self, key: str) -> bool:
        """특정 키의 캐시 무효화"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """전체 캐시 클리어"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_items = len(self.cache)
        expired_items = 0
        current_time = time.time()

        for key, item in self.cache.items():
            if current_time >= item['expires_at']:
                expired_items += 1

        return {
            'total_items': total_items,
            'expired_items': expired_items,
            'active_items': total_items - expired_items,
            'cache_size_bytes': len(str(self.cache).encode())
        }

# 싱글톤 인스턴스
_cache_service = None

def get_cache_service() -> CacheService:
    """캐시 서비스 싱글톤 인스턴스 반환"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service