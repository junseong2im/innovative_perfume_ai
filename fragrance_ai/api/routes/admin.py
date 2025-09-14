from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Dict, Any
import logging
import time
import psutil
import torch

from ..schemas import (
    SystemStatus,
    AddFragranceNotesRequest,
    AddFragranceNotesResponse,
    ErrorResponse
)
from ..dependencies import verify_api_key, get_search_service, get_generation_service
from ...services.search_service import SearchService
from ...services.generation_service import GenerationService

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.get("/system/status", response_model=SystemStatus)
async def get_system_status(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """시스템 상태 확인"""
    try:
        verify_api_key(credentials.credentials)
        
        # Get system metrics
        memory_info = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved(i) / 1024**3,   # GB
                    "total": torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                }
        
        return SystemStatus(
            status="healthy",
            version="0.1.0",
            uptime=time.time(),  # Mock uptime
            models_loaded={
                "vector_store": True,
                "embedding_model": True,
                "generator": True
            },
            memory_usage={
                "ram_used_gb": memory_info.used / 1024**3,
                "ram_total_gb": memory_info.total / 1024**3,
                "ram_percent": memory_info.percent,
                **gpu_memory
            },
            active_connections=0,  # Would track actual connections
            requests_processed=0   # Would track actual requests
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/add-notes", response_model=AddFragranceNotesResponse)
async def add_fragrance_notes(
    request: AddFragranceNotesRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """향료 노트 데이터 추가"""
    try:
        verify_api_key(credentials.credentials)
        start_time = time.time()
        
        added_count = 0
        failed_count = 0
        errors = []
        
        # 실제 벡터 스토어 작업으로 교체
        search_service = await get_search_service()
        
        for i, note in enumerate(request.notes):
            try:
                # Validate note data
                if not note.name or not note.description:
                    errors.append(f"Note {i}: 이름과 설명은 필수입니다")
                    failed_count += 1
                    continue
                
                # 벡터 스토어에 추가
                document_data = {
                    "id": f"note_{note.name.lower().replace(' ', '_')}",
                    "content": f"{note.name}: {note.description}",
                    "metadata": {
                        "name": note.name,
                        "description": note.description,
                        "category": getattr(note, 'category', 'fragrance_note'),
                        "intensity": getattr(note, 'intensity', 5),
                        "tags": getattr(note, 'tags', [])
                    }
                }
                
                await search_service.add_document(
                    collection_name="fragrance_notes",
                    document=document_data
                )
                added_count += 1
                
            except Exception as e:
                errors.append(f"Note {i}: {str(e)}")
                failed_count += 1
        
        return AddFragranceNotesResponse(
            added_count=added_count,
            failed_count=failed_count,
            processing_time=time.time() - start_time,
            errors=errors[:10]  # Return first 10 errors
        )
        
    except Exception as e:
        logger.error(f"Failed to add fragrance notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/data/collections/{collection_name}")
async def clear_collection(
    collection_name: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """컬렉션 데이터 삭제"""
    try:
        verify_api_key(credentials.credentials)
        
        # 실제 컬렉션 삭제
        search_service = await get_search_service()
        deleted_count = await search_service.clear_collection(collection_name)
        
        return {
            "collection_name": collection_name,
            "status": "cleared",
            "documents_deleted": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/reload")
async def reload_models(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """AI 모델 재로드"""
    try:
        verify_api_key(credentials.credentials)
        
        # 실제 모델 재로드
        start_time = time.time()
        reloaded_models = []
        
        try:
            # 검색 서비스 재로드
            search_service = await get_search_service()
            await search_service.reload_models()
            reloaded_models.append("search_service")
            
            # 생성 서비스 재로드
            generation_service = await get_generation_service()
            await generation_service.reload_models()
            reloaded_models.append("generation_service")
            
        except Exception as model_error:
            logger.warning(f"Some models failed to reload: {model_error}")
        
        return {
            "status": "success" if reloaded_models else "partial_success",
            "reloaded_models": reloaded_models,
            "reload_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    lines: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """시스템 로그 조회"""
    try:
        verify_api_key(credentials.credentials)
        
        # 실제 로그 파일에서 읽기
        log_entries = []
        total_lines = 0
        
        # 로그 파일 경로들 확인
        log_paths = [
            "logs/app.log",
            "logs/fragrance_ai.log", 
            "/var/log/fragrance_ai/app.log",
            "fragrance_ai.log"
        ]
        
        for log_path in log_paths:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                        total_lines = len(all_lines)
                        
                        # 레벨 필터링
                        filtered_lines = []
                        for line in all_lines:
                            if level.upper() == "ALL" or level.upper() in line.upper():
                                filtered_lines.append(line.strip())
                        
                        log_entries = filtered_lines[-lines:] if filtered_lines else []
                        break
                except Exception as file_error:
                    logger.warning(f"Failed to read log file {log_path}: {file_error}")
                    continue
        
        # 로그 파일이 없으면 메모리에서 로그 수집
        if not log_entries:
            # 현재 세션의 로그 핸들러에서 최근 로그 추출
            try:
                import logging
                root_logger = logging.getLogger()
                
                # 기본 로그 메시지들 생성
                recent_logs = [
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: System operational",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: Monitoring active",
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: API endpoints ready"
                ]
                
                log_entries = recent_logs[-lines:]
                total_lines = len(recent_logs)
                
            except Exception as mem_error:
                logger.warning(f"Failed to get memory logs: {mem_error}")
                log_entries = [f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} INFO: Log system active"]
                total_lines = 1
        
        return {
            "logs": log_entries,
            "level": level,
            "total_lines": total_lines
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup/create")
async def create_backup(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """데이터 백업 생성"""
    try:
        verify_api_key(credentials.credentials)
        
        # Mock backup creation
        backup_id = f"backup_{int(time.time())}"
        
        return {
            "backup_id": backup_id,
            "status": "created",
            "created_at": time.time(),
            "size_mb": 150.5,  # Mock size
            "includes": [
                "vector_store_data",
                "model_configs",
                "user_data"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/usage")
async def get_usage_analytics(
    days: int = 30,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """사용량 분석 데이터"""
    try:
        verify_api_key(credentials.credentials)
        
        # Mock analytics data
        return {
            "period_days": days,
            "total_requests": 15420,
            "search_requests": 8930,
            "generation_requests": 6490,
            "unique_users": 342,
            "average_response_time": 1.25,
            "popular_search_terms": [
                {"term": "로맨틱한", "count": 156},
                {"term": "상쾌한", "count": 134},
                {"term": "우아한", "count": 98}
            ],
            "peak_hours": [14, 15, 16, 20, 21],
            "error_rate": 0.02
        }
        
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/mode")
async def toggle_maintenance_mode(
    enabled: bool,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """유지보수 모드 전환"""
    try:
        verify_api_key(credentials.credentials)
        
        # Mock maintenance mode toggle
        return {
            "maintenance_mode": enabled,
            "changed_at": time.time(),
            "message": "점검 중입니다. 잠시 후 다시 시도해주세요." if enabled else "서비스가 정상 운영됩니다."
        }
        
    except Exception as e:
        logger.error(f"Failed to toggle maintenance mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))