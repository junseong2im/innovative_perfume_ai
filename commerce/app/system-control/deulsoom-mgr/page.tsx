'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import UserService from '../../../lib/user-service';

interface LoginFormData {
  username: string;
  password: string;
  accessCode: string;
  twoFactorCode?: string;
}

interface IPRestriction {
  enabled: boolean;
  allowedIPs: string[];
}

export default function SecureAdminLogin() {
  const router = useRouter();
  const userService = UserService.getInstance();
  const [formData, setFormData] = useState<LoginFormData>({
    username: '',
    password: '',
    accessCode: '',
    twoFactorCode: ''
  });
  const [showTwoFactor, setShowTwoFactor] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [ipRestricted, setIpRestricted] = useState(false);

  useEffect(() => {
    // IP 접근 제한 확인
    checkIPRestriction();

    // 이미 로그인된 관리자인지 확인
    const adminUser = localStorage.getItem('deulsoom_admin_secure');
    if (adminUser) {
      const userData = JSON.parse(adminUser);
      if (userData.role === 'super_admin' || userData.role === 'admin') {
        router.push('/system-control/deulsoom-mgr/dashboard');
      }
    }
  }, [router]);

  const checkIPRestriction = async () => {
    try {
      // 실제 구현에서는 서버에서 IP를 확인하지만,
      // 데모를 위해 클라이언트에서 시뮬레이션
      const response = await fetch('https://api.ipify.org?format=json');
      const data = await response.json();
      const userIP = data.ip;

      // 허용된 IP 목록 (실제로는 서버 환경변수나 데이터베이스에서 관리)
      const allowedIPs = ['127.0.0.1', 'localhost', '::1'];
      const isLocalhost = window.location.hostname === 'localhost' ||
                         window.location.hostname === '127.0.0.1' ||
                         window.location.hostname === '::1';

      if (!isLocalhost && !allowedIPs.includes(userIP)) {
        setIpRestricted(true);
      }
    } catch (error) {
      console.warn('IP check failed, allowing access for demo');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateAccessCode = (code: string): boolean => {
    // 실제 구현에서는 서버에서 검증
    const validCodes = ['DEULSOOM2025', 'ADMIN_SECURE_ACCESS', 'MGMT_PORTAL_2025'];
    return validCodes.includes(code);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      // 1. 접근 코드 검증
      if (!validateAccessCode(formData.accessCode)) {
        throw new Error('잘못된 접근 코드입니다.');
      }

      // 2. 통합 서비스를 통한 관리자 인증
      const validAdmin = userService.authenticateAdmin(formData.username, formData.password);

      if (!validAdmin) {
        throw new Error('잘못된 관리자 자격 증명입니다.');
      }

      // 3. 2단계 인증 시뮬레이션
      if (!showTwoFactor) {
        setShowTwoFactor(true);
        setLoading(false);
        return;
      }

      // 2단계 인증 코드 검증 (실제로는 TOTP나 SMS)
      if (formData.twoFactorCode !== '123456') {
        throw new Error('잘못된 2단계 인증 코드입니다.');
      }

      // 4. 로그인 성공
      const adminData = {
        id: validAdmin.username,
        username: validAdmin.username,
        role: validAdmin.role,
        loginTime: new Date().toISOString(),
        sessionToken: `admin_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      };

      localStorage.setItem('deulsoom_admin_secure', JSON.stringify(adminData));

      // 로그인 로그 기록 (실제로는 서버에서)
      console.log(`Admin Login: ${validAdmin.username} (${validAdmin.role}) at ${new Date().toISOString()}`);

      router.push('/system-control/deulsoom-mgr/dashboard');

    } catch (err) {
      setError(err instanceof Error ? err.message : '로그인 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  if (ipRestricted) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#1a1a1a' }}>
        <div className="max-w-md w-full mx-auto">
          <div className="bg-red-900 border border-red-700 rounded-lg p-8 text-center">
            <div className="text-red-300 mb-4">
              <svg className="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-red-200 mb-2">접근 제한</h2>
            <p className="text-red-300">
              이 IP 주소에서는 관리자 시스템에 접근할 수 없습니다.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#0f1419' }}>
      <div className="max-w-md w-full mx-auto p-6">

        {/* 보안 경고 */}
        <div className="bg-orange-900 border border-orange-700 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <div className="text-orange-300 mr-3">
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <div>
              <h3 className="text-sm font-medium text-orange-200">보안 시스템</h3>
              <p className="text-xs text-orange-300 mt-1">
                이 페이지는 인증된 관리자만 접근 가능합니다.
              </p>
            </div>
          </div>
        </div>

        {/* 로그인 폼 */}
        <div className="bg-gray-900 border border-gray-700 rounded-lg p-8">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-semibold text-white mb-2">들숨 관리 시스템</h1>
            <p className="text-gray-400">System Control Portal</p>
          </div>

          {error && (
            <div className="bg-red-900 border border-red-700 rounded-lg p-3 mb-6">
              <p className="text-red-200 text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="accessCode" className="block text-sm font-medium text-gray-300 mb-2">
                접근 코드
              </label>
              <input
                id="accessCode"
                name="accessCode"
                type="password"
                required
                value={formData.accessCode}
                onChange={handleInputChange}
                placeholder="시스템 접근 코드"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                관리자 ID
              </label>
              <input
                id="username"
                name="username"
                type="text"
                required
                value={formData.username}
                onChange={handleInputChange}
                placeholder="관리자 계정"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                비밀번호
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                value={formData.password}
                onChange={handleInputChange}
                placeholder="비밀번호"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {showTwoFactor && (
              <div>
                <label htmlFor="twoFactorCode" className="block text-sm font-medium text-gray-300 mb-2">
                  2단계 인증 코드
                </label>
                <input
                  id="twoFactorCode"
                  name="twoFactorCode"
                  type="text"
                  required
                  value={formData.twoFactorCode}
                  onChange={handleInputChange}
                  placeholder="123456 (데모용)"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <p className="text-xs text-gray-400 mt-1">
                  데모용: 123456을 입력하세요
                </p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white font-medium py-2 px-4 rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
            >
              {loading ? '인증 중...' : showTwoFactor ? '로그인 완료' : '다음 단계'}
            </button>
          </form>

          {/* 데모 정보 */}
          <div className="mt-8 p-4 bg-gray-800 rounded-lg">
            <h3 className="text-sm font-medium text-gray-300 mb-2">데모 계정 정보:</h3>
            <div className="text-xs text-gray-400 space-y-1">
              <p><strong>접근 코드:</strong> DEULSOOM2025</p>
              <p><strong>최고 관리자:</strong> super_admin / DeulsoomSuper2025!</p>
              <p><strong>일반 관리자:</strong> admin / DeulsoomAdmin2025!</p>
              <p><strong>마케팅 관리자:</strong> marketing / DeulsoomMkt2025!</p>
              <p><strong>개발자:</strong> dev / DeulsoomDev2025!</p>
              <p><strong>2단계 인증:</strong> 123456</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}