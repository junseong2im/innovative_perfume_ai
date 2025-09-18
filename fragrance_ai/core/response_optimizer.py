"""
API 응답 최적화 시스템
- 응답 압축
- 지연 로딩
- 배치 응답
- 스트리밍 응답
"""

import asyncio
import gzip
import brotli
import zlib
import json
import time
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.types import Scope, Receive, Send
import orjson
from datetime import datetime, timezone

from .logging_config import get_logger
from .monitoring import MetricsCollector

logger = get_logger(__name__)

class CompressionType(Enum):
    GZIP = "gzip"
    BROTLI = "br"
    DEFLATE = "deflate"
    NONE = "none"

class ResponseFormat(Enum):
    JSON = "json"
    STREAMING_JSON = "streaming_json"
    CHUNKED = "chunked"
    COMPRESSED = "compressed"

@dataclass
class OptimizationSettings:
    # 압축 설정
    enable_compression: bool = True
    compression_threshold: int = 1024  # 1KB
    compression_level: int = 6
    preferred_compression: CompressionType = CompressionType.GZIP

    # 지연 로딩 설정
    enable_lazy_loading: bool = True
    lazy_threshold: int = 100  # 100개 이상 항목시 지연 로딩

    # 배치 처리 설정
    enable_batch_responses: bool = True
    batch_size: int = 50
    batch_timeout: float = 0.1  # 100ms

    # 스트리밍 설정
    enable_streaming: bool = True
    streaming_chunk_size: int = 8192  # 8KB

    # 캐싱 설정
    enable_response_caching: bool = True
    cache_control_max_age: int = 300  # 5분

    # 메트릭
    enable_metrics: bool = True

@dataclass
class ResponseMetrics:
    total_responses: int = 0
    compressed_responses: int = 0
    streamed_responses: int = 0
    cached_responses: int = 0
    avg_response_size: float = 0.0
    avg_compression_ratio: float = 0.0
    avg_response_time: float = 0.0
    compression_savings: int = 0

class ResponseOptimizer:
    """API 응답 최적화 관리자"""

    def __init__(self, settings: Optional[OptimizationSettings] = None):
        self.settings = settings or OptimizationSettings()
        self.metrics = ResponseMetrics()
        self.metrics_collector = MetricsCollector() if self.settings.enable_metrics else None

        # 배치 처리를 위한 큐
        self.batch_queue: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        self.batch_task: Optional[asyncio.Task] = None

        # 응답 캐시 (메모리 기반)
        self.response_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self.cache_max_size = 1000

        # 압축 캐시
        self.compression_cache: Dict[str, bytes] = {}

    def get_compression_type(self, request: Request) -> CompressionType:
        """클라이언트가 지원하는 압축 방식 결정"""
        if not self.settings.enable_compression:
            return CompressionType.NONE

        accept_encoding = request.headers.get("accept-encoding", "").lower()

        if "br" in accept_encoding and self.settings.preferred_compression == CompressionType.BROTLI:
            return CompressionType.BROTLI
        elif "gzip" in accept_encoding:
            return CompressionType.GZIP
        elif "deflate" in accept_encoding:
            return CompressionType.DEFLATE
        else:
            return CompressionType.NONE

    def should_compress(self, content: bytes) -> bool:
        """압축 여부 결정"""
        return (
            self.settings.enable_compression and
            len(content) >= self.settings.compression_threshold
        )

    def compress_content(self, content: bytes, compression_type: CompressionType) -> bytes:
        """콘텐츠 압축"""
        if compression_type == CompressionType.NONE:
            return content

        # 압축 캐시 확인
        cache_key = f"{compression_type.value}:{hash(content)}"
        if cache_key in self.compression_cache:
            return self.compression_cache[cache_key]

        start_time = time.time()

        try:
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(content, compresslevel=self.settings.compression_level)
            elif compression_type == CompressionType.BROTLI:
                compressed = brotli.compress(content, quality=self.settings.compression_level)
            elif compression_type == CompressionType.DEFLATE:
                compressed = zlib.compress(content, level=self.settings.compression_level)
            else:
                compressed = content

            # 압축 캐시에 저장 (메모리 제한)
            if len(self.compression_cache) < 100:
                self.compression_cache[cache_key] = compressed

            # 메트릭 업데이트
            compression_time = time.time() - start_time
            compression_ratio = len(content) / len(compressed) if len(compressed) > 0 else 1.0

            self.metrics.compressed_responses += 1
            self.metrics.avg_compression_ratio = (
                (self.metrics.avg_compression_ratio * (self.metrics.compressed_responses - 1) + compression_ratio) /
                self.metrics.compressed_responses
            )
            self.metrics.compression_savings += len(content) - len(compressed)

            if self.metrics_collector:
                self.metrics_collector.record_compression(
                    compression_type=compression_type.value,
                    original_size=len(content),
                    compressed_size=len(compressed),
                    compression_time=compression_time
                )

            return compressed

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return content

    def optimize_json_response(
        self,
        data: Any,
        request: Request,
        include_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None
    ) -> Response:
        """JSON 응답 최적화"""
        start_time = time.time()

        try:
            # 필드 필터링
            if include_fields or exclude_fields:
                data = self._filter_response_fields(data, include_fields, exclude_fields)

            # 고성능 JSON 직렬화
            content = orjson.dumps(
                data,
                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
            )

            # 압축 적용
            compression_type = self.get_compression_type(request)
            if self.should_compress(content):
                compressed_content = self.compress_content(content, compression_type)

                headers = {
                    "content-encoding": compression_type.value,
                    "content-type": "application/json",
                    "content-length": str(len(compressed_content))
                }

                # 캐시 헤더 추가
                if self.settings.enable_response_caching:
                    headers["cache-control"] = f"public, max-age={self.settings.cache_control_max_age}"
                    headers["etag"] = f'"{hash(content)}"'

                response = Response(
                    content=compressed_content,
                    headers=headers,
                    media_type="application/json"
                )
            else:
                headers = {
                    "content-type": "application/json",
                    "content-length": str(len(content))
                }

                if self.settings.enable_response_caching:
                    headers["cache-control"] = f"public, max-age={self.settings.cache_control_max_age}"
                    headers["etag"] = f'"{hash(content)}"'

                response = Response(
                    content=content,
                    headers=headers,
                    media_type="application/json"
                )

            # 메트릭 업데이트
            response_time = time.time() - start_time
            self._update_response_metrics(len(content), response_time)

            return response

        except Exception as e:
            logger.error(f"JSON response optimization failed: {e}")
            # 폴백: 기본 JSON 응답
            return Response(
                content=json.dumps(data),
                media_type="application/json"
            )

    def create_streaming_response(
        self,
        data_generator: AsyncGenerator[Any, None],
        request: Request,
        chunk_size: Optional[int] = None
    ) -> StreamingResponse:
        """스트리밍 응답 생성"""
        chunk_size = chunk_size or self.settings.streaming_chunk_size
        compression_type = self.get_compression_type(request)

        async def generate_chunks():
            buffer = []
            buffer_size = 0

            try:
                async for item in data_generator:
                    # JSON 직렬화
                    json_item = orjson.dumps(item) + b'\n'

                    buffer.append(json_item)
                    buffer_size += len(json_item)

                    # 청크 크기에 도달하면 전송
                    if buffer_size >= chunk_size:
                        chunk_data = b''.join(buffer)

                        # 압축 적용
                        if compression_type != CompressionType.NONE:
                            chunk_data = self.compress_content(chunk_data, compression_type)

                        yield chunk_data

                        buffer.clear()
                        buffer_size = 0

                # 남은 데이터 전송
                if buffer:
                    chunk_data = b''.join(buffer)

                    if compression_type != CompressionType.NONE:
                        chunk_data = self.compress_content(chunk_data, compression_type)

                    yield chunk_data

            except Exception as e:
                logger.error(f"Streaming response generation failed: {e}")
                yield b'{"error": "streaming_failed"}\n'

        headers = {"content-type": "application/json"}

        if compression_type != CompressionType.NONE:
            headers["content-encoding"] = compression_type.value

        # 스트리밍 메트릭 업데이트
        self.metrics.streamed_responses += 1

        return StreamingResponse(
            generate_chunks(),
            headers=headers,
            media_type="application/json"
        )

    def create_lazy_response(
        self,
        data: List[Any],
        request: Request,
        page_size: int = 20,
        total_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """지연 로딩 응답 생성"""
        total_count = total_count or len(data)

        # 첫 페이지 데이터만 포함
        first_page = data[:page_size] if isinstance(data, list) else list(data)[:page_size]

        base_url = str(request.url).split('?')[0]

        response_data = {
            "data": first_page,
            "pagination": {
                "total_count": total_count,
                "page_size": page_size,
                "has_more": total_count > page_size,
                "next_page_url": f"{base_url}?page=2&page_size={page_size}" if total_count > page_size else None
            },
            "lazy_loading": True
        }

        return response_data

    async def add_to_batch(self, response_data: Dict[str, Any]) -> str:
        """배치 응답에 데이터 추가"""
        if not self.settings.enable_batch_responses:
            return None

        batch_id = f"batch_{int(time.time() * 1000)}"

        async with self.batch_lock:
            self.batch_queue.append({
                "id": batch_id,
                "data": response_data,
                "timestamp": time.time()
            })

            # 배치 처리 작업 시작 (아직 없다면)
            if self.batch_task is None or self.batch_task.done():
                self.batch_task = asyncio.create_task(self._process_batch_queue())

        return batch_id

    async def _process_batch_queue(self):
        """배치 큐 처리"""
        while True:
            try:
                await asyncio.sleep(self.settings.batch_timeout)

                async with self.batch_lock:
                    if not self.batch_queue:
                        break

                    # 배치 크기만큼 처리
                    batch_items = self.batch_queue[:self.settings.batch_size]
                    self.batch_queue = self.batch_queue[self.settings.batch_size:]

                    if batch_items:
                        await self._send_batch_response(batch_items)

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

    async def _send_batch_response(self, batch_items: List[Dict[str, Any]]):
        """배치 응답 전송"""
        batch_response = {
            "batch_id": f"batch_{int(time.time() * 1000)}",
            "items": batch_items,
            "count": len(batch_items),
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Processed batch with {len(batch_items)} items")
        # 실제 환경에서는 웹소켓이나 SSE로 클라이언트에 전송

    def _filter_response_fields(
        self,
        data: Any,
        include_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None
    ) -> Any:
        """응답 필드 필터링"""
        if not isinstance(data, dict):
            return data

        filtered_data = data.copy()

        # 제외 필드 제거
        if exclude_fields:
            for field in exclude_fields:
                filtered_data.pop(field, None)

        # 포함 필드만 유지
        if include_fields:
            filtered_data = {
                key: value for key, value in filtered_data.items()
                if key in include_fields
            }

        return filtered_data

    def _update_response_metrics(self, response_size: int, response_time: float):
        """응답 메트릭 업데이트"""
        self.metrics.total_responses += 1

        # 평균 응답 크기 업데이트
        self.metrics.avg_response_size = (
            (self.metrics.avg_response_size * (self.metrics.total_responses - 1) + response_size) /
            self.metrics.total_responses
        )

        # 평균 응답 시간 업데이트
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_responses - 1) + response_time) /
            self.metrics.total_responses
        )

        if self.metrics_collector:
            self.metrics_collector.record_response_metrics(
                response_size=response_size,
                response_time=response_time
            )

    def get_cache_key(self, request: Request, extra_params: Optional[Dict[str, Any]] = None) -> str:
        """캐시 키 생성"""
        base_params = {
            "url": str(request.url),
            "method": request.method,
            "headers": dict(request.headers)
        }

        if extra_params:
            base_params.update(extra_params)

        cache_content = json.dumps(base_params, sort_keys=True)
        return f"response:{hash(cache_content)}"

    async def get_cached_response(self, cache_key: str) -> Optional[Response]:
        """캐시된 응답 조회"""
        if not self.settings.enable_response_caching:
            return None

        if cache_key in self.response_cache:
            cached_content, cached_time = self.response_cache[cache_key]

            # 캐시 만료 확인
            if (datetime.now() - cached_time).total_seconds() < self.settings.cache_control_max_age:
                self.metrics.cached_responses += 1

                return Response(
                    content=cached_content,
                    headers={
                        "content-type": "application/json",
                        "x-cache": "HIT"
                    }
                )

        return None

    async def cache_response(self, cache_key: str, content: bytes):
        """응답 캐시"""
        if not self.settings.enable_response_caching:
            return

        # 캐시 크기 제한
        if len(self.response_cache) >= self.cache_max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.response_cache.keys(),
                           key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_key]

        self.response_cache[cache_key] = (content, datetime.now())

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        total_responses = self.metrics.total_responses
        compression_rate = (
            (self.metrics.compressed_responses / total_responses * 100)
            if total_responses > 0 else 0
        )

        cache_hit_rate = (
            (self.metrics.cached_responses / total_responses * 100)
            if total_responses > 0 else 0
        )

        return {
            "total_responses": total_responses,
            "compressed_responses": self.metrics.compressed_responses,
            "compression_rate_percent": round(compression_rate, 2),
            "avg_compression_ratio": round(self.metrics.avg_compression_ratio, 2),
            "compression_savings_bytes": self.metrics.compression_savings,
            "streamed_responses": self.metrics.streamed_responses,
            "cached_responses": self.metrics.cached_responses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "avg_response_size_bytes": round(self.metrics.avg_response_size, 2),
            "avg_response_time_ms": round(self.metrics.avg_response_time * 1000, 2),
            "response_cache_size": len(self.response_cache),
            "compression_cache_size": len(self.compression_cache)
        }

    async def cleanup_caches(self):
        """캐시 정리"""
        current_time = datetime.now()

        # 만료된 응답 캐시 제거
        expired_keys = [
            key for key, (content, cached_time) in self.response_cache.items()
            if (current_time - cached_time).total_seconds() > self.settings.cache_control_max_age
        ]

        for key in expired_keys:
            del self.response_cache[key]

        # 압축 캐시 크기 제한
        if len(self.compression_cache) > 200:
            # 절반 제거 (LRU 방식은 복잡하므로 단순하게)
            keys_to_remove = list(self.compression_cache.keys())[:100]
            for key in keys_to_remove:
                del self.compression_cache[key]

        logger.info(f"Cache cleanup completed. Removed {len(expired_keys)} expired responses")


# 전역 응답 최적화 인스턴스
response_optimizer: Optional[ResponseOptimizer] = None

def get_response_optimizer() -> ResponseOptimizer:
    """글로벌 응답 최적화 인스턴스 반환"""
    global response_optimizer
    if response_optimizer is None:
        response_optimizer = ResponseOptimizer()
    return response_optimizer

def initialize_response_optimizer(settings: Optional[OptimizationSettings] = None) -> ResponseOptimizer:
    """응답 최적화 시스템 초기화"""
    global response_optimizer
    response_optimizer = ResponseOptimizer(settings)
    return response_optimizer