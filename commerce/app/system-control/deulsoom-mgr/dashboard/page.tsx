'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import UserService from '../../../../lib/user-service';

interface AdminUser {
  id: string;
  username: string;
  role: 'super_admin' | 'admin' | 'marketing_admin' | 'dev_admin';
  loginTime: string;
  sessionToken: string;
}

interface SystemStats {
  totalUsers: number;
  totalOrders: number;
  revenue: number;
  activeFragrances: number;
  systemHealth: 'good' | 'warning' | 'critical';
}

export default function SecureAdminDashboard() {
  const router = useRouter();
  const userService = UserService.getInstance();
  const [adminUser, setAdminUser] = useState<AdminUser | null>(null);
  const [stats, setStats] = useState<SystemStats>({
    totalUsers: 0,
    totalOrders: 0,
    revenue: 0,
    activeFragrances: 156,
    systemHealth: 'good'
  });
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    // 서버 세션 확인
    const checkSession = async () => {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

      try {
        const response = await fetch(`${API_URL}/api/v1/auth/admin/session`, {
          method: 'GET',
          credentials: 'include', // 쿠키 포함
          headers: {
            'X-CSRF-Token': sessionStorage.getItem('csrf_token') || ''
          }
        });

        if (!response.ok) {
          router.push('/system-control/deulsoom-mgr/login');
          return;
        }

        const userData = await response.json();
        setAdminUser({
          id: userData.user_id,
          username: userData.username || userData.email,
          role: userData.role || 'admin',
          loginTime: userData.login_time || new Date().toISOString(),
          sessionToken: 'server-managed'
        });
      } catch (error) {
        console.error('Session check failed:', error);
        router.push('/system-control/deulsoom-mgr/login');
        return;
      }
    };

    checkSession();

    // 통합 서비스에서 실시간 통계 로드
    const loadStats = () => {
      const statistics = userService.getStatistics();
      setStats({
        totalUsers: statistics.totalUsers,
        totalOrders: statistics.totalOrders,
        revenue: statistics.totalRevenue,
        activeFragrances: 156, // 고정값 (향수 제품 수)
        systemHealth: 'good'
      });
    };

    loadStats();

    // 시간 업데이트
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      loadStats(); // 1초마다 통계도 업데이트
    }, 1000);

    // 사용자 데이터 변경 감지
    const handleUsersUpdate = () => {
      loadStats();
    };

    window.addEventListener('usersUpdated', handleUsersUpdate);

    return () => {
      clearInterval(timer);
      window.removeEventListener('usersUpdated', handleUsersUpdate);
    };
  }, [router]);

  const handleLogout = async () => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

    try {
      await fetch(`${API_URL}/api/v1/auth/admin/logout`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'X-CSRF-Token': sessionStorage.getItem('csrf_token') || ''
        }
      });
    } catch (error) {
      console.error('Logout failed:', error);
    } finally {
      sessionStorage.removeItem('csrf_token');
      router.push('/system-control/deulsoom-mgr/login');
    }
  };

  const getRoleDisplayName = (role: string) => {
    const roleNames = {
      'super_admin': '최고 관리자',
      'admin': '일반 관리자',
      'marketing_admin': '마케팅 관리자',
      'dev_admin': '개발자'
    };
    return roleNames[role as keyof typeof roleNames] || role;
  };

  const getRoleColor = (role: string) => {
    const colors = {
      'super_admin': 'bg-red-900 text-red-200',
      'admin': 'bg-blue-900 text-blue-200',
      'marketing_admin': 'bg-green-900 text-green-200',
      'dev_admin': 'bg-purple-900 text-purple-200'
    };
    return colors[role as keyof typeof colors] || 'bg-gray-900 text-gray-200';
  };

  const getAccessibleFeatures = (role: string) => {
    const features = {
      'super_admin': [
        { name: '시스템 설정', icon: '⚙️', path: '/system-settings', description: '전체 시스템 구성' },
        { name: '사용자 관리', icon: '👥', path: '/user-management', description: '고객 및 관리자 계정' },
        { name: '주문 관리', icon: '📦', path: '/order-management', description: '주문 및 배송 관리' },
        { name: '상품 관리', icon: '🧴', path: '/product-management', description: '향수 제품 관리' },
        { name: '마케팅 도구', icon: '📊', path: '/marketing-tools', description: '프로모션 및 캠페인' },
        { name: '개발 도구', icon: '💻', path: '/dev-tools', description: '개발 및 배포 관리' },
        { name: '보안 설정', icon: '🔒', path: '/security-settings', description: '보안 정책 관리' },
        { name: '감사 로그', icon: '📋', path: '/audit-logs', description: '시스템 활동 로그' }
      ],
      'admin': [
        { name: '사용자 관리', icon: '👥', path: '/user-management', description: '고객 계정 관리' },
        { name: '주문 관리', icon: '📦', path: '/order-management', description: '주문 및 배송 관리' },
        { name: '상품 관리', icon: '🧴', path: '/product-management', description: '향수 제품 관리' },
        { name: '고객 지원', icon: '💬', path: '/customer-support', description: '고객 문의 처리' },
        { name: '재고 관리', icon: '📊', path: '/inventory', description: '재고 현황 관리' }
      ],
      'marketing_admin': [
        { name: '마케팅 도구', icon: '📊', path: '/marketing-tools', description: '프로모션 및 캠페인' },
        { name: '콘텐츠 관리', icon: '📝', path: '/content-management', description: '웹사이트 콘텐츠' },
        { name: '고객 분석', icon: '📈', path: '/customer-analytics', description: '고객 행동 분석' },
        { name: '이벤트 관리', icon: '🎉', path: '/event-management', description: '이벤트 및 프로모션' }
      ],
      'dev_admin': [
        { name: '개발 도구', icon: '💻', path: '/dev-tools', description: '개발 및 배포 관리' },
        { name: 'API 관리', icon: '🔌', path: '/api-management', description: 'API 설정 및 모니터링' },
        { name: '데이터베이스', icon: '🗄️', path: '/database', description: '데이터베이스 관리' },
        { name: '시스템 모니터링', icon: '📡', path: '/monitoring', description: '성능 및 오류 모니터링' }
      ]
    };
    return features[role as keyof typeof features] || [];
  };

  if (!adminUser) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white">로딩 중...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* 헤더 */}
      <header className="bg-black border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-white">들숨 관리 시스템</h1>
              <div className="ml-4 px-3 py-1 text-xs rounded-full bg-gray-800 text-gray-300">
                {currentTime.toLocaleString('ko-KR')}
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${getRoleColor(adminUser.role)}`}>
                {getRoleDisplayName(adminUser.role)}
              </div>
              <span className="text-gray-300">{adminUser.username}</span>
              <button
                onClick={handleLogout}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm transition-colors"
              >
                로그아웃
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 시스템 상태 카드 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center">
              <div className="text-blue-400">
                <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">총 사용자</p>
                <p className="text-2xl font-semibold text-white">{stats.totalUsers.toLocaleString()}</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center">
              <div className="text-green-400">
                <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">총 주문</p>
                <p className="text-2xl font-semibold text-white">{stats.totalOrders.toLocaleString()}</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center">
              <div className="text-yellow-400">
                <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">총 매출</p>
                <p className="text-2xl font-semibold text-white">₩{(stats.revenue / 1000000).toFixed(1)}M</p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center">
              <div className="text-purple-400">
                <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">활성 향수</p>
                <p className="text-2xl font-semibold text-white">{stats.activeFragrances}</p>
              </div>
            </div>
          </div>
        </div>

        {/* 관리 기능 */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-6">
            {getRoleDisplayName(adminUser.role)} 권한 - 관리 기능
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {getAccessibleFeatures(adminUser.role).map((feature, index) => (
              <Link
                key={index}
                href={feature.path.startsWith('/') ? `/system-control/deulsoom-mgr${feature.path}` : feature.path}
                className="block bg-gray-700 hover:bg-gray-600 rounded-lg p-4 transition-colors border border-gray-600 hover:border-gray-500"
              >
                <div className="text-center">
                  <div className="text-2xl mb-2">{feature.icon}</div>
                  <h3 className="font-medium text-white mb-1">{feature.name}</h3>
                  <p className="text-xs text-gray-400">{feature.description}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>

        {/* 최근 활동 */}
        <div className="mt-8 bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-6">최근 시스템 활동</h2>

          <div className="space-y-4">
            {[
              { time: '5분 전', user: 'customer_123', action: '새로운 향수 주문', status: 'success' },
              { time: '12분 전', user: 'admin', action: '상품 정보 업데이트', status: 'info' },
              { time: '1시간 전', user: 'system', action: '정기 백업 완료', status: 'success' },
              { time: '2시간 전', user: 'marketing', action: '프로모션 캠페인 시작', status: 'info' },
              { time: '3시간 전', user: 'customer_456', action: '고객 문의 접수', status: 'warning' }
            ].map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${
                    activity.status === 'success' ? 'bg-green-400' :
                    activity.status === 'warning' ? 'bg-yellow-400' :
                    activity.status === 'error' ? 'bg-red-400' : 'bg-blue-400'
                  }`}></div>
                  <div>
                    <p className="text-white text-sm">{activity.action}</p>
                    <p className="text-gray-400 text-xs">사용자: {activity.user}</p>
                  </div>
                </div>
                <span className="text-gray-400 text-xs">{activity.time}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 시스템 상태 */}
        <div className="mt-8 bg-gray-800 rounded-lg border border-gray-700 p-6">
          <h2 className="text-xl font-semibold text-white mb-6">시스템 상태</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-green-400 mb-2">
                <svg className="h-8 w-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-white font-medium">서버 상태</p>
              <p className="text-green-400 text-sm">정상 운영</p>
            </div>

            <div className="text-center">
              <div className="text-green-400 mb-2">
                <svg className="h-8 w-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
              <p className="text-white font-medium">데이터베이스</p>
              <p className="text-green-400 text-sm">정상 연결</p>
            </div>

            <div className="text-center">
              <div className="text-green-400 mb-2">
                <svg className="h-8 w-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <p className="text-white font-medium">보안 시스템</p>
              <p className="text-green-400 text-sm">모든 정상</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}