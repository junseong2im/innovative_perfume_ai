'use client';

import { useState } from 'react';
import { AuthService, type LoginCredentials } from 'lib/auth';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: any) => void;
}

export default function LoginModal({ isOpen, onClose, onLoginSuccess }: LoginModalProps) {
  const [credentials, setCredentials] = useState<LoginCredentials>({
    username: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await AuthService.login(credentials);

      if (response.success && response.access_token) {
        const user = {
          username: response.user_role || credentials.username,
          role: response.user_role || 'customer',
          token: response.access_token
        };

        AuthService.saveUser(user);
        onLoginSuccess(user);
        onClose();
      } else {
        setError(response.message);
      }
    } catch (err) {
      setError('로그인 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickLogin = async (role: 'customer' | 'admin') => {
    // Request credentials from environment or remove quick login feature
    const quickCredentials = {
      customer: {
        username: process.env.NEXT_PUBLIC_TEST_CUSTOMER_USERNAME || '',
        password: process.env.NEXT_PUBLIC_TEST_CUSTOMER_PASSWORD || ''
      },
      admin: {
        username: process.env.NEXT_PUBLIC_TEST_ADMIN_USERNAME || '',
        password: process.env.NEXT_PUBLIC_TEST_ADMIN_PASSWORD || ''
      }
    };

    if (!quickCredentials[role].username || !quickCredentials[role].password) {
      setError('Quick login not configured. Please use manual login.');
      return;
    }

    setCredentials(quickCredentials[role]);

    // 자동으로 로그인 처리
    setLoading(true);
    try {
      const response = await AuthService.login(quickCredentials[role]);
      if (response.success && response.access_token) {
        const user = {
          username: response.user_role || role,
          role: response.user_role || role,
          token: response.access_token
        };

        AuthService.saveUser(user);
        onLoginSuccess(user);
        onClose();
      }
    } catch (err) {
      setError('로그인 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 배경 오버레이 */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50"
        onClick={onClose}
      />

      {/* 모달 컨텐츠 */}
      <div className="relative bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-light text-neutral-900">로그인</h2>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-600"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 빠른 로그인 버튼들 */}
        <div className="mb-6 space-y-3">
          <p className="text-sm text-neutral-600 text-center">빠른 로그인</p>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => handleQuickLogin('customer')}
              disabled={loading}
              className="px-4 py-3 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors text-sm font-medium disabled:opacity-50"
            >
              👤 고객으로 로그인
            </button>
            <button
              onClick={() => handleQuickLogin('admin')}
              disabled={loading}
              className="px-4 py-3 bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors text-sm font-medium disabled:opacity-50"
            >
              🔑 관리자로 로그인
            </button>
          </div>
        </div>

        <div className="relative mb-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-neutral-200" />
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-neutral-500">또는</span>
          </div>
        </div>

        {/* 로그인 폼 */}
        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-neutral-700 mb-2">
              사용자명
            </label>
            <input
              type="text"
              value={credentials.username}
              onChange={(e) => setCredentials({...credentials, username: e.target.value})}
              className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-neutral-500 focus:border-transparent"
              placeholder="customer 또는 admin"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-700 mb-2">
              비밀번호
            </label>
            <input
              type="password"
              value={credentials.password}
              onChange={(e) => setCredentials({...credentials, password: e.target.value})}
              className="w-full px-3 py-2 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-neutral-500 focus:border-transparent"
              placeholder="비밀번호 입력"
              required
            />
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                로그인 중...
              </div>
            ) : (
              '로그인'
            )}
          </button>
        </form>

        {/* 테스트 계정 안내 */}
        <div className="mt-6 p-4 bg-neutral-50 rounded-lg">
          <p className="text-xs text-neutral-600 mb-2">테스트 계정:</p>
          <div className="space-y-1 text-xs text-neutral-500">
            <p>• 고객: customer / customer123</p>
            <p>• 관리자: admin / admin123!</p>
          </div>
        </div>
      </div>
    </div>
  );
}