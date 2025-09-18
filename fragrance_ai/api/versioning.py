"""
API 버전 관리 시스템
FastAPI를 통한 체계적인 API 버전 관리
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from fastapi import Request, Response, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class APIVersion(str, Enum):
    """API 버전 열거형"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"

class DeprecationStatus(str, Enum):
    """API 지원 상태"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"

class VersionInfo(BaseModel):
    """버전 정보 모델"""
    version: APIVersion
    status: DeprecationStatus
    release_date: str
    sunset_date: Optional[str] = None
    breaking_changes: List[str] = []
    new_features: List[str] = []
    documentation_url: str
    migration_guide_url: Optional[str] = None

class APIVersionManager:
    """API 버전 관리자"""

    def __init__(self):
        self.versions: Dict[APIVersion, VersionInfo] = {
            APIVersion.V1: VersionInfo(
                version=APIVersion.V1,
                status=DeprecationStatus.DEPRECATED,
                release_date="2023-01-01",
                sunset_date="2024-12-31",
                breaking_changes=[
                    "Legacy authentication removed",
                    "Old response format deprecated"
                ],
                documentation_url="/docs/v1",
                migration_guide_url="/docs/migrate-v1-to-v2"
            ),
            APIVersion.V2: VersionInfo(
                version=APIVersion.V2,
                status=DeprecationStatus.ACTIVE,
                release_date="2023-06-01",
                new_features=[
                    "Advanced RAG system",
                    "Enhanced security",
                    "Improved performance"
                ],
                documentation_url="/docs/v2"
            ),
            APIVersion.V3: VersionInfo(
                version=APIVersion.V3,
                status=DeprecationStatus.ACTIVE,
                release_date="2024-01-01",
                new_features=[
                    "Real-time recommendations",
                    "Multi-language support",
                    "Advanced analytics"
                ],
                documentation_url="/docs/v3"
            )
        }

        self.default_version = APIVersion.V2
        self.supported_versions = list(self.versions.keys())

        logger.info(f"API 버전 관리자 초기화 완료. 지원 버전: {self.supported_versions}")

    def get_version_from_request(self, request: Request) -> APIVersion:
        """요청에서 API 버전 추출"""

        # 1. Accept 헤더에서 버전 확인
        accept_header = request.headers.get("Accept", "")
        if "application/vnd.fragrance-ai.v" in accept_header:
            try:
                version_str = accept_header.split("application/vnd.fragrance-ai.v")[1].split("+")[0]
                version = APIVersion(f"v{version_str}")
                if self.is_version_supported(version):
                    return version
            except (ValueError, IndexError):
                pass

        # 2. API-Version 헤더에서 버전 확인
        version_header = request.headers.get("API-Version")
        if version_header:
            try:
                version = APIVersion(version_header.lower())
                if self.is_version_supported(version):
                    return version
            except ValueError:
                pass

        # 3. URL 경로에서 버전 확인
        path = request.url.path
        if path.startswith("/api/"):
            path_parts = path.split("/")
            if len(path_parts) >= 3:
                potential_version = path_parts[2]
                try:
                    version = APIVersion(potential_version)
                    if self.is_version_supported(version):
                        return version
                except ValueError:
                    pass

        # 4. 쿼리 매개변수에서 버전 확인
        version_param = request.query_params.get("version")
        if version_param:
            try:
                version = APIVersion(version_param.lower())
                if self.is_version_supported(version):
                    return version
            except ValueError:
                pass

        # 기본 버전 반환
        return self.default_version

    def is_version_supported(self, version: APIVersion) -> bool:
        """버전이 지원되는지 확인"""
        return version in self.supported_versions

    def is_version_deprecated(self, version: APIVersion) -> bool:
        """버전이 지원 중단되었는지 확인"""
        if version not in self.versions:
            return True
        return self.versions[version].status == DeprecationStatus.DEPRECATED

    def is_version_sunset(self, version: APIVersion) -> bool:
        """버전이 서비스 종료되었는지 확인"""
        if version not in self.versions:
            return True
        return self.versions[version].status == DeprecationStatus.SUNSET

    def get_version_info(self, version: APIVersion) -> Optional[VersionInfo]:
        """버전 정보 조회"""
        return self.versions.get(version)

    def add_deprecation_headers(self, response: Response, version: APIVersion) -> None:
        """지원 중단 헤더 추가"""
        version_info = self.get_version_info(version)
        if not version_info:
            return

        response.headers["API-Version"] = version.value
        response.headers["API-Supported-Versions"] = ",".join([v.value for v in self.supported_versions])

        if version_info.status == DeprecationStatus.DEPRECATED:
            response.headers["API-Deprecation"] = "true"
            if version_info.sunset_date:
                response.headers["API-Sunset"] = version_info.sunset_date
            if version_info.migration_guide_url:
                response.headers["API-Migration-Guide"] = version_info.migration_guide_url

            logger.warning(f"Deprecated API version {version} accessed")

    def validate_version(self, version: APIVersion) -> None:
        """버전 유효성 검사"""
        if not self.is_version_supported(version):
            raise HTTPException(
                status_code=406,
                detail={
                    "error": "Unsupported API version",
                    "version_requested": version.value,
                    "supported_versions": [v.value for v in self.supported_versions],
                    "default_version": self.default_version.value
                }
            )

        if self.is_version_sunset(version):
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "API version no longer available",
                    "version_requested": version.value,
                    "status": "sunset",
                    "migration_guide": self.versions[version].migration_guide_url
                }
            )

class VersionedAPIRouter(APIRouter):
    """버전별 API 라우터"""

    def __init__(self, version: APIVersion, version_manager: APIVersionManager, **kwargs):
        self.version = version
        self.version_manager = version_manager

        # 버전 프리픽스 설정
        if "prefix" not in kwargs:
            kwargs["prefix"] = f"/api/{version.value}"

        super().__init__(**kwargs)

        # 버전 정보 엔드포인트 추가
        self.add_version_endpoint()

    def add_version_endpoint(self):
        """버전 정보 엔드포인트 추가"""
        @self.get("/version")
        async def get_version_info():
            """현재 API 버전 정보 조회"""
            version_info = self.version_manager.get_version_info(self.version)
            return {
                "api_version": self.version.value,
                "version_info": version_info.dict() if version_info else None,
                "server_time": "2024-01-01T00:00:00Z"  # 실제로는 현재 시간
            }

# 글로벌 버전 관리자 인스턴스
api_version_manager = APIVersionManager()

# 미들웨어 함수들
async def version_middleware(request: Request, call_next):
    """API 버전 미들웨어"""

    # 요청에서 API 버전 추출
    version = api_version_manager.get_version_from_request(request)

    # 버전 유효성 검사
    try:
        api_version_manager.validate_version(version)
    except HTTPException as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=e.status_code, content=e.detail)

    # 요청에 버전 정보 저장
    request.state.api_version = version

    # 요청 처리
    response = await call_next(request)

    # 응답에 버전 관련 헤더 추가
    api_version_manager.add_deprecation_headers(response, version)

    return response

def get_api_version(request: Request) -> APIVersion:
    """현재 요청의 API 버전 조회"""
    return getattr(request.state, 'api_version', api_version_manager.default_version)

# 버전별 응답 변환 데코레이터
def version_compatible_response(func):
    """버전 호환성 응답 데코레이터"""
    async def wrapper(*args, **kwargs):
        request = kwargs.get('request') or args[0]
        version = get_api_version(request)

        # 원본 함수 실행
        result = await func(*args, **kwargs)

        # 버전별 응답 형식 변환
        if version == APIVersion.V1:
            # V1 호환성 변환
            if isinstance(result, dict) and "data" not in result:
                result = {"data": result, "status": "success"}

        elif version == APIVersion.V2:
            # V2 형식 (현재 기본)
            pass

        elif version == APIVersion.V3:
            # V3 향상된 형식
            if isinstance(result, dict):
                result["meta"] = {
                    "api_version": version.value,
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": getattr(request.state, 'request_id', None)
                }

        return result

    return wrapper

# 호환성 검사 유틸리티
class CompatibilityChecker:
    """API 호환성 검사기"""

    @staticmethod
    def check_breaking_changes(from_version: APIVersion, to_version: APIVersion) -> List[str]:
        """버전 간 호환성 파괴 변경 사항 확인"""
        version_manager = api_version_manager

        breaking_changes = []

        # 버전 정보 가져오기
        from_info = version_manager.get_version_info(from_version)
        to_info = version_manager.get_version_info(to_version)

        if to_info and to_info.breaking_changes:
            breaking_changes.extend(to_info.breaking_changes)

        return breaking_changes

    @staticmethod
    def suggest_migration_path(current_version: APIVersion) -> Dict[str, Any]:
        """마이그레이션 경로 제안"""
        version_manager = api_version_manager
        latest_version = max(version_manager.supported_versions, key=lambda v: v.value)

        if current_version == latest_version:
            return {"message": "현재 최신 버전을 사용 중입니다."}

        migration_info = {
            "current_version": current_version.value,
            "recommended_version": latest_version.value,
            "migration_required": current_version != latest_version
        }

        current_info = version_manager.get_version_info(current_version)
        if current_info and current_info.migration_guide_url:
            migration_info["migration_guide"] = current_info.migration_guide_url

        # 호환성 파괴 변경 사항 확인
        breaking_changes = CompatibilityChecker.check_breaking_changes(
            current_version, latest_version
        )
        if breaking_changes:
            migration_info["breaking_changes"] = breaking_changes

        return migration_info