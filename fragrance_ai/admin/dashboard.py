"""
Fragrance AI 관리자 대시보드
실시간 모니터링, 사용자 관리, 시스템 운영 도구
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..core.exceptions_advanced import AuthorizationError, ErrorCode
from ..database.connection import get_db_session
from ..business.billing_system import SubscriptionManager, PricingEngine
from .auth import require_admin_access

router = APIRouter(prefix="/admin", tags=["admin"])

# ==============================================================================
# 응답 모델
# ==============================================================================

class DashboardStats(BaseModel):
    """대시보드 통계"""
    total_users: int
    active_users_today: int
    total_subscriptions: int
    total_revenue_monthly: float
    api_calls_today: int
    fragrance_generations_today: int
    system_health: Dict[str, str]
    recent_errors: List[Dict[str, Any]]

class UserOverview(BaseModel):
    """사용자 개요"""
    id: str
    email: str
    created_at: datetime
    last_login: Optional[datetime]
    subscription_tier: str
    total_api_calls: int
    total_spent: float
    status: str

class SystemMetrics(BaseModel):
    """시스템 메트릭"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    response_times: Dict[str, float]

class UsageAnalytics(BaseModel):
    """사용량 분석"""
    period: str
    api_calls: List[Dict[str, Any]]
    fragrance_generations: List[Dict[str, Any]]
    user_growth: List[Dict[str, Any]]
    revenue_trend: List[Dict[str, Any]]

# ==============================================================================
# 대시보드 서비스
# ==============================================================================

class DashboardService:
    """대시보드 서비스"""

    def __init__(self, db: Session):
        self.db = db
        self.pricing_engine = PricingEngine()

    async def get_dashboard_stats(self) -> DashboardStats:
        """대시보드 통계 조회"""
        try:
            # 사용자 통계
            total_users = await self._get_total_users()
            active_users_today = await self._get_active_users_today()

            # 구독 통계
            total_subscriptions = await self._get_total_subscriptions()
            total_revenue_monthly = await self._get_monthly_revenue()

            # API 사용량 통계
            api_calls_today = await self._get_api_calls_today()
            fragrance_generations_today = await self._get_fragrance_generations_today()

            # 시스템 상태
            system_health = await self._get_system_health()
            recent_errors = await self._get_recent_errors()

            return DashboardStats(
                total_users=total_users,
                active_users_today=active_users_today,
                total_subscriptions=total_subscriptions,
                total_revenue_monthly=total_revenue_monthly,
                api_calls_today=api_calls_today,
                fragrance_generations_today=fragrance_generations_today,
                system_health=system_health,
                recent_errors=recent_errors
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard stats: {str(e)}")

    async def get_user_overview(
        self,
        page: int = 1,
        page_size: int = 20,
        search: Optional[str] = None,
        tier_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """사용자 개요 조회"""
        try:
            offset = (page - 1) * page_size

            # 기본 쿼리 (실제 구현에서는 User 모델 사용)
            query = """
            SELECT u.id, u.email, u.created_at, u.last_login,
                   s.tier as subscription_tier, u.status,
                   COALESCE(usage_stats.total_api_calls, 0) as total_api_calls,
                   COALESCE(billing_stats.total_spent, 0) as total_spent
            FROM users u
            LEFT JOIN subscriptions s ON u.id = s.user_id AND s.status = 'active'
            LEFT JOIN (
                SELECT user_id, COUNT(*) as total_api_calls
                FROM usage_records
                WHERE usage_type = 'api_call'
                GROUP BY user_id
            ) usage_stats ON u.id = usage_stats.user_id
            LEFT JOIN (
                SELECT user_id, SUM(total_cost) as total_spent
                FROM usage_records
                GROUP BY user_id
            ) billing_stats ON u.id = billing_stats.user_id
            WHERE 1=1
            """

            params = []

            if search:
                query += " AND u.email ILIKE %s"
                params.append(f"%{search}%")

            if tier_filter:
                query += " AND s.tier = %s"
                params.append(tier_filter)

            query += f" ORDER BY u.created_at DESC LIMIT {page_size} OFFSET {offset}"

            # 실제로는 SQLAlchemy ORM 사용
            users = []  # self.db.execute(query, params).fetchall()

            # 총 사용자 수
            total_query = "SELECT COUNT(*) FROM users WHERE 1=1"
            if search:
                total_query += " AND email ILIKE %s"
            if tier_filter:
                total_query += " AND id IN (SELECT user_id FROM subscriptions WHERE tier = %s AND status = 'active')"

            # total_users = self.db.execute(total_query, params[:len([p for p in [search, tier_filter] if p])]).scalar()

            return {
                "users": users,
                "total_users": 0,  # total_users,
                "page": page,
                "page_size": page_size,
                "total_pages": 0  # math.ceil(total_users / page_size)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get user overview: {str(e)}")

    async def get_usage_analytics(self, period: str = "7d") -> UsageAnalytics:
        """사용량 분석 조회"""
        try:
            if period == "7d":
                start_date = datetime.now() - timedelta(days=7)
            elif period == "30d":
                start_date = datetime.now() - timedelta(days=30)
            elif period == "90d":
                start_date = datetime.now() - timedelta(days=90)
            else:
                start_date = datetime.now() - timedelta(days=7)

            # API 호출 통계
            api_calls = await self._get_api_calls_trend(start_date)

            # 향수 생성 통계
            fragrance_generations = await self._get_fragrance_generations_trend(start_date)

            # 사용자 증가 통계
            user_growth = await self._get_user_growth_trend(start_date)

            # 수익 통계
            revenue_trend = await self._get_revenue_trend(start_date)

            return UsageAnalytics(
                period=period,
                api_calls=api_calls,
                fragrance_generations=fragrance_generations,
                user_growth=user_growth,
                revenue_trend=revenue_trend
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get usage analytics: {str(e)}")

    async def get_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 조회"""
        try:
            import psutil
            import time

            # 시스템 메트릭 수집
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 응답 시간 (실제로는 모니터링 시스템에서 가져오기)
            response_times = {
                "api_avg": 150.5,
                "database_avg": 25.3,
                "ai_model_avg": 2340.7
            }

            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_connections=await self._get_active_connections(),
                response_times=response_times
            )

        except Exception as e:
            # psutil이 없는 환경에서는 모의 데이터 반환
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=15.5,
                memory_usage=45.2,
                disk_usage=60.8,
                active_connections=42,
                response_times={
                    "api_avg": 150.5,
                    "database_avg": 25.3,
                    "ai_model_avg": 2340.7
                }
            )

    # 헬퍼 메서드들
    async def _get_total_users(self) -> int:
        """총 사용자 수"""
        # return self.db.query(func.count(User.id)).scalar()
        return 1250  # 모의 데이터

    async def _get_active_users_today(self) -> int:
        """오늘 활성 사용자 수"""
        today = datetime.now().date()
        # return self.db.query(func.count(User.id)).filter(func.date(User.last_login) == today).scalar()
        return 89  # 모의 데이터

    async def _get_total_subscriptions(self) -> int:
        """총 구독 수"""
        # return self.db.query(func.count(Subscription.id)).filter(Subscription.status == 'active').scalar()
        return 456  # 모의 데이터

    async def _get_monthly_revenue(self) -> float:
        """월간 수익"""
        # 이번 달 수익 계산
        return 12450.75  # 모의 데이터

    async def _get_api_calls_today(self) -> int:
        """오늘 API 호출 수"""
        today = datetime.now().date()
        # return self.db.query(func.count(UsageRecord.id)).filter(
        #     UsageRecord.usage_type == 'api_call',
        #     func.date(UsageRecord.timestamp) == today
        # ).scalar()
        return 15420  # 모의 데이터

    async def _get_fragrance_generations_today(self) -> int:
        """오늘 향수 생성 수"""
        today = datetime.now().date()
        return 234  # 모의 데이터

    async def _get_system_health(self) -> Dict[str, str]:
        """시스템 상태"""
        # 실제로는 각 서비스의 헬스체크 결과를 수집
        return {
            "api": "healthy",
            "database": "healthy",
            "ai_models": "healthy",
            "cache": "healthy",
            "storage": "healthy"
        }

    async def _get_recent_errors(self) -> List[Dict[str, Any]]:
        """최근 에러 목록"""
        # 실제로는 로그 시스템에서 에러 조회
        return [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "level": "ERROR",
                "message": "AI model timeout",
                "count": 3
            },
            {
                "timestamp": "2024-01-15T09:15:00Z",
                "level": "WARNING",
                "message": "High memory usage",
                "count": 1
            }
        ]

    async def _get_api_calls_trend(self, start_date: datetime) -> List[Dict[str, Any]]:
        """API 호출 트렌드"""
        # 실제로는 시계열 데이터 조회
        days = []
        current = start_date
        while current <= datetime.now():
            days.append({
                "date": current.strftime("%Y-%m-%d"),
                "count": 1000 + (hash(current.strftime("%Y-%m-%d")) % 500)
            })
            current += timedelta(days=1)
        return days

    async def _get_fragrance_generations_trend(self, start_date: datetime) -> List[Dict[str, Any]]:
        """향수 생성 트렌드"""
        days = []
        current = start_date
        while current <= datetime.now():
            days.append({
                "date": current.strftime("%Y-%m-%d"),
                "count": 50 + (hash(current.strftime("%Y-%m-%d")) % 100)
            })
            current += timedelta(days=1)
        return days

    async def _get_user_growth_trend(self, start_date: datetime) -> List[Dict[str, Any]]:
        """사용자 증가 트렌드"""
        days = []
        current = start_date
        cumulative_users = 1000
        while current <= datetime.now():
            new_users = 5 + (hash(current.strftime("%Y-%m-%d")) % 20)
            cumulative_users += new_users
            days.append({
                "date": current.strftime("%Y-%m-%d"),
                "new_users": new_users,
                "total_users": cumulative_users
            })
            current += timedelta(days=1)
        return days

    async def _get_revenue_trend(self, start_date: datetime) -> List[Dict[str, Any]]:
        """수익 트렌드"""
        days = []
        current = start_date
        while current <= datetime.now():
            days.append({
                "date": current.strftime("%Y-%m-%d"),
                "revenue": 100 + (hash(current.strftime("%Y-%m-%d")) % 500)
            })
            current += timedelta(days=1)
        return days

    async def _get_active_connections(self) -> int:
        """활성 연결 수"""
        # 실제로는 데이터베이스 연결 풀에서 조회
        return 42

# ==============================================================================
# API 엔드포인트
# ==============================================================================

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    request: Request,
    db: Session = Depends(get_db_session),
    admin_user = Depends(require_admin_access)
):
    """대시보드 통계 조회"""
    dashboard_service = DashboardService(db)
    return await dashboard_service.get_dashboard_stats()

@router.get("/users")
async def get_user_overview(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    tier_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db_session),
    admin_user = Depends(require_admin_access)
):
    """사용자 개요 조회"""
    dashboard_service = DashboardService(db)
    return await dashboard_service.get_user_overview(page, page_size, search, tier_filter)

@router.get("/analytics", response_model=UsageAnalytics)
async def get_usage_analytics(
    request: Request,
    period: str = Query("7d", regex="^(7d|30d|90d)$"),
    db: Session = Depends(get_db_session),
    admin_user = Depends(require_admin_access)
):
    """사용량 분석 조회"""
    dashboard_service = DashboardService(db)
    return await dashboard_service.get_usage_analytics(period)

@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    request: Request,
    admin_user = Depends(require_admin_access)
):
    """시스템 메트릭 조회"""
    db_session = None  # 실제로는 의존성 주입
    dashboard_service = DashboardService(db_session)
    return await dashboard_service.get_system_metrics()

@router.post("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db_session),
    admin_user = Depends(require_admin_access)
):
    """사용자 계정 정지"""
    try:
        # 실제로는 User 모델 업데이트
        # user = db.query(User).filter(User.id == user_id).first()
        # if not user:
        #     raise HTTPException(status_code=404, detail="User not found")
        #
        # user.status = "suspended"
        # user.suspended_at = datetime.now()
        # user.suspended_by = admin_user.id
        # db.commit()

        return {"message": f"User {user_id} suspended successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to suspend user: {str(e)}")

@router.post("/users/{user_id}/reactivate")
async def reactivate_user(
    user_id: str,
    request: Request,
    db: Session = Depends(get_db_session),
    admin_user = Depends(require_admin_access)
):
    """사용자 계정 재활성화"""
    try:
        # 실제로는 User 모델 업데이트
        return {"message": f"User {user_id} reactivated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reactivate user: {str(e)}")

@router.get("/logs")
async def get_system_logs(
    request: Request,
    level: str = Query("ERROR", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=1000),
    admin_user = Depends(require_admin_access)
):
    """시스템 로그 조회"""
    try:
        # 실제로는 로깅 시스템에서 조회
        logs = [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "level": "ERROR",
                "logger": "fragrance_ai.api.main",
                "message": "AI model inference failed",
                "extra": {"user_id": "user123", "model": "embedding"}
            },
            {
                "timestamp": "2024-01-15T10:25:00Z",
                "level": "WARNING",
                "logger": "fragrance_ai.core.cache",
                "message": "Cache miss rate high",
                "extra": {"hit_rate": 0.65}
            }
        ]

        return {
            "logs": logs[:limit],
            "total_count": len(logs),
            "level_filter": level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

@router.post("/maintenance-mode")
async def toggle_maintenance_mode(
    request: Request,
    enable: bool = True,
    message: str = "시스템 점검 중입니다.",
    admin_user = Depends(require_admin_access)
):
    """유지보수 모드 설정"""
    try:
        # 실제로는 Redis나 설정 파일에 저장
        maintenance_config = {
            "enabled": enable,
            "message": message,
            "started_at": datetime.now().isoformat(),
            "started_by": admin_user.get("id", "admin")
        }

        return {
            "maintenance_mode": enable,
            "message": message,
            "config": maintenance_config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle maintenance mode: {str(e)}")

# ==============================================================================
# HTML 대시보드 (개발용)
# ==============================================================================

@router.get("/", response_class=HTMLResponse)
async def admin_dashboard_html(
    request: Request,
    admin_user = Depends(require_admin_access)
):
    """관리자 대시보드 HTML (개발용)"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fragrance AI - 관리자 대시보드</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .stat-value { font-size: 2em; font-weight: bold; color: #2563eb; }
            .stat-label { color: #6b7280; margin-top: 5px; }
            .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .actions { display: flex; gap: 10px; flex-wrap: wrap; }
            .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 500; }
            .btn-primary { background: #2563eb; color: white; }
            .btn-danger { background: #dc2626; color: white; }
            .btn-warning { background: #d97706; color: white; }
            .system-health { display: flex; gap: 10px; flex-wrap: wrap; }
            .health-indicator { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
            .health-healthy { background: #dcfce7; color: #166534; }
            .health-warning { background: #fef3c7; color: #92400e; }
            .health-error { background: #fecaca; color: #991b1b; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌸 Fragrance AI 관리자 대시보드</h1>
            <p>실시간 시스템 모니터링 및 관리</p>
        </div>

        <div class="stats-grid" id="stats-grid">
            <!-- 통계는 JavaScript로 동적 로딩 -->
        </div>

        <div class="chart-container">
            <h3>API 사용량 추이 (7일)</h3>
            <canvas id="usageChart" width="400" height="200"></canvas>
        </div>

        <div class="chart-container">
            <h3>시스템 리소스</h3>
            <canvas id="systemChart" width="400" height="200"></canvas>
        </div>

        <div class="chart-container">
            <h3>관리 작업</h3>
            <div class="actions">
                <button class="btn btn-primary" onclick="refreshData()">데이터 새로고침</button>
                <button class="btn btn-warning" onclick="toggleMaintenance()">유지보수 모드</button>
                <button class="btn btn-danger" onclick="viewLogs()">시스템 로그</button>
            </div>
        </div>

        <script>
            async function loadDashboardData() {
                try {
                    const response = await fetch('/admin/dashboard');
                    const data = await response.json();

                    // 통계 카드 업데이트
                    const statsGrid = document.getElementById('stats-grid');
                    statsGrid.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${data.total_users.toLocaleString()}</div>
                            <div class="stat-label">총 사용자</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.active_users_today}</div>
                            <div class="stat-label">오늘 활성 사용자</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.total_subscriptions}</div>
                            <div class="stat-label">활성 구독</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">$${data.total_revenue_monthly.toLocaleString()}</div>
                            <div class="stat-label">월간 수익</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.api_calls_today.toLocaleString()}</div>
                            <div class="stat-label">오늘 API 호출</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${data.fragrance_generations_today}</div>
                            <div class="stat-label">오늘 향수 생성</div>
                        </div>
                    `;

                    // 시스템 상태 표시
                    const healthHTML = Object.entries(data.system_health)
                        .map(([service, status]) =>
                            `<span class="health-indicator health-${status}">${service}: ${status}</span>`
                        ).join('');

                    statsGrid.innerHTML += `
                        <div class="stat-card">
                            <div class="stat-label">시스템 상태</div>
                            <div class="system-health">${healthHTML}</div>
                        </div>
                    `;

                } catch (error) {
                    console.error('Failed to load dashboard data:', error);
                }
            }

            async function loadCharts() {
                // API 사용량 차트
                const usageCtx = document.getElementById('usageChart').getContext('2d');
                new Chart(usageCtx, {
                    type: 'line',
                    data: {
                        labels: ['월', '화', '수', '목', '금', '토', '일'],
                        datasets: [{
                            label: 'API 호출',
                            data: [12000, 15000, 18000, 14000, 16000, 19000, 15420],
                            borderColor: '#2563eb',
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });

                // 시스템 리소스 차트
                const systemCtx = document.getElementById('systemChart').getContext('2d');
                new Chart(systemCtx, {
                    type: 'bar',
                    data: {
                        labels: ['CPU', 'Memory', 'Disk'],
                        datasets: [{
                            label: '사용률 (%)',
                            data: [15.5, 45.2, 60.8],
                            backgroundColor: ['#10b981', '#f59e0b', '#ef4444']
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            }

            function refreshData() {
                loadDashboardData();
                alert('데이터를 새로고침했습니다.');
            }

            function toggleMaintenance() {
                if (confirm('유지보수 모드를 변경하시겠습니까?')) {
                    alert('유지보수 모드 설정이 변경되었습니다.');
                }
            }

            function viewLogs() {
                window.open('/admin/logs', '_blank');
            }

            // 초기 로딩
            document.addEventListener('DOMContentLoaded', function() {
                loadDashboardData();
                loadCharts();

                // 30초마다 데이터 자동 새로고침
                setInterval(loadDashboardData, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)