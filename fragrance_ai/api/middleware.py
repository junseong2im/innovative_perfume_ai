from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import time
import logging
import json
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """요청/응답 로깅 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Get request body for logging (if needed)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Don't log sensitive data
                    if "password" not in str(body).lower() and "token" not in str(body).lower():
                        logger.debug(f"Request body: {body.decode()[:500]}")
            except Exception as e:
                # 요청 본문 파싱 실패 시 로깅
                logger.debug(f"Failed to parse request body for logging: {e}")
                logger.debug(f"Request content type: {request.headers.get('content-type', 'unknown')}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} | "
            f"Time: {process_time:.3f}s | "
            f"Path: {request.url.path}"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """속도 제한 미들웨어"""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.clients = defaultdict(lambda: deque())
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        now = datetime.now()
        
        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        while self.clients[client_ip] and self.clients[client_ip][0] < minute_ago:
            self.clients[client_ip].popleft()
        
        # Check rate limit
        if len(self.clients[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.calls_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        self.clients[client_ip].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.calls_per_minute - len(self.clients[client_ip]))
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int((now + timedelta(minutes=1)).timestamp()))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        # Check for proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"


class SecurityMiddleware(BaseHTTPMiddleware):
    """보안 미들웨어"""
    
    def __init__(self, app):
        super().__init__(app)
        self.blocked_ips = set()  # In production, use Redis or database
        self.suspicious_patterns = [
            "SELECT * FROM",
            "<script",
            "javascript:",
            "../",
            "eval(",
            "exec("
        ]
    
    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "detail": "IP address blocked"}
            )
        
        # Check for suspicious patterns
        if await self._is_suspicious_request(request):
            logger.warning(f"Suspicious request from {client_ip}: {request.url}")
            return JSONResponse(
                status_code=400,
                content={"error": "Bad request", "detail": "Request blocked by security filter"}
            )
        
        # Add security headers
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _is_suspicious_request(self, request: Request) -> bool:
        """의심스러운 요청 확인"""
        try:
            # Check URL
            url_str = str(request.url)
            for pattern in self.suspicious_patterns:
                if pattern.lower() in url_str.lower():
                    return True
            
            # Check request body for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    body_str = body.decode().lower()
                    for pattern in self.suspicious_patterns:
                        if pattern.lower() in body_str:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking suspicious request: {e}")
            return False


class CacheMiddleware(BaseHTTPMiddleware):
    """캐싱 미들웨어"""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cacheable_paths = [
            "/api/v1/search/collections/",
            "/health",
            "/metrics"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if path is cacheable
        path = request.url.path
        is_cacheable = any(path.startswith(cacheable) for cacheable in self.cacheable_paths)
        
        if not is_cacheable:
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.method}:{path}:{str(request.query_params)}"
        
        # Check cache
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for {cache_key}")
            return JSONResponse(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers={**cached_response["headers"], "X-Cache": "HIT"}
            )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            await self._cache_response(cache_key, response)
        
        response.headers["X-Cache"] = "MISS"
        return response
    
    def _get_cached_response(self, cache_key: str) -> Dict[str, Any] or None:
        """캐시된 응답 조회"""
        if cache_key not in self.cache:
            return None
        
        cached = self.cache[cache_key]
        
        # Check if expired
        if time.time() > cached["expires_at"]:
            del self.cache[cache_key]
            return None
        
        return cached
    
    async def _cache_response(self, cache_key: str, response: Response):
        """응답 캐싱"""
        try:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Parse JSON response
            content = json.loads(response_body.decode())
            
            # Cache the response
            self.cache[cache_key] = {
                "content": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "expires_at": time.time() + self.cache_ttl
            }
            
            # Recreate response body iterator
            response.body_iterator = self._create_body_iterator(response_body)
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
    
    def _create_body_iterator(self, body: bytes):
        """응답 바디 이터레이터 생성"""
        async def body_iterator():
            yield body
        return body_iterator()


class MetricsMiddleware(BaseHTTPMiddleware):
    """메트릭스 수집 미들웨어"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_times = deque(maxlen=1000)  # Keep last 1000 request times
        self.status_codes = defaultdict(int)
        self.endpoint_calls = defaultdict(int)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Increment request count
        self.request_count += 1
        
        # Count endpoint calls
        endpoint = f"{request.method}:{request.url.path}"
        self.endpoint_calls[endpoint] += 1
        
        # Process request
        response = await call_next(request)
        
        # Record metrics
        process_time = time.time() - start_time
        self.request_times.append(process_time)
        self.status_codes[response.status_code] += 1
        
        # Add metrics headers
        response.headers["X-Request-Count"] = str(self.request_count)
        response.headers["X-Response-Time"] = f"{process_time:.3f}"
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭스 데이터 반환"""
        return {
            "total_requests": self.request_count,
            "average_response_time": sum(self.request_times) / len(self.request_times) if self.request_times else 0,
            "status_codes": dict(self.status_codes),
            "endpoint_calls": dict(self.endpoint_calls),
            "recent_response_times": list(self.request_times)[-10:]  # Last 10 response times
        }