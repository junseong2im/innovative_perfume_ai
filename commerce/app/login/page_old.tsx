'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { AuthService, type LoginCredentials } from 'lib/auth';
import { SocialAuthService, socialProviders } from 'lib/social-auth';
import TermsModal from 'components/auth/terms-modal';

export default function LoginPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [phone, setPhone] = useState('');
  const [birthDate, setBirthDate] = useState('');
  const [gender, setGender] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [socialLoading, setSocialLoading] = useState('');
  const [error, setError] = useState('');

  // 약관 동의 상태
  const [agreements, setAgreements] = useState({
    all: false,
    terms: false,
    privacy: false,
    marketing: false
  });

  // 모달 상태
  const [showTermsModal, setShowTermsModal] = useState(false);
  const [modalType, setModalType] = useState<'terms' | 'privacy'>('terms');

  // 이메일 유효성 검사
  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // 비밀번호 유효성 검사
  const validatePassword = (password: string) => {
    return password.length >= 8 && /^(?=.*[a-zA-Z])(?=.*\d)/.test(password);
  };

  // 휴대폰 번호 유효성 검사
  const validatePhone = (phone: string) => {
    const phoneRegex = /^01[0-9]-?[0-9]{4}-?[0-9]{4}$/;
    return phoneRegex.test(phone);
  };

  // 전체 동의 체크박스 핸들러
  const handleAllAgreement = (checked: boolean) => {
    setAgreements({
      all: checked,
      terms: checked,
      privacy: checked,
      marketing: checked
    });
  };

  // 개별 동의 체크박스 핸들러
  const handleAgreementChange = (type: string, checked: boolean) => {
    const newAgreements = { ...agreements, [type]: checked };
    newAgreements.all = newAgreements.terms && newAgreements.privacy && newAgreements.marketing;
    setAgreements(newAgreements);
  };

  // 소셜 로그인 핸들러
  const handleSocialLogin = async (provider: string) => {
    setSocialLoading(provider);
    setError('');

    try {
      const response = await SocialAuthService.socialLogin(provider);

      if (response.success && response.user) {
        const user = {
          username: response.user.name,
          role: 'customer' as const,
          token: `social_token_${provider}_${Date.now()}`,
          provider: response.user.provider,
          picture: response.user.picture
        };

        AuthService.saveUser(user);
        router.push('/');
      } else {
        setError(response.error || '소셜 로그인에 실패했습니다.');
      }
    } catch (err) {
      setError('소셜 로그인 중 오류가 발생했습니다.');
    } finally {
      setSocialLoading('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        // 로그인 처리
        if (!validateEmail(email)) {
          setError('올바른 이메일 주소를 입력해주세요.');
          return;
        }

        const credentials: LoginCredentials = {
          username: email,
          password: password
        };

        const response = await AuthService.login(credentials);

        if (response.success && response.access_token) {
          const user = {
            username: response.user_role || email.split('@')[0],
            role: response.user_role || 'customer',
            token: response.access_token
          };

          AuthService.saveUser(user);
          router.push('/');
        } else {
          setError(response.message || '로그인에 실패했습니다.');
        }
      } else {
        // 회원가입 유효성 검사
        if (!validateEmail(email)) {
          setError('올바른 이메일 주소를 입력해주세요.');
          return;
        }

        if (!validatePassword(password)) {
          setError('비밀번호는 8자 이상이며, 영문과 숫자를 포함해야 합니다.');
          return;
        }

        if (password !== confirmPassword) {
          setError('비밀번호가 일치하지 않습니다.');
          return;
        }

        if (!name.trim()) {
          setError('이름을 입력해주세요.');
          return;
        }

        if (!validatePhone(phone)) {
          setError('올바른 휴대폰 번호를 입력해주세요. (예: 010-1234-5678)');
          return;
        }

        // 필수 약관 동의 확인
        if (!agreements.terms || !agreements.privacy) {
          setError('필수 약관에 동의해주세요.');
          return;
        }

        // 회원가입 처리 (실제로는 백엔드 API 호출)
        const userData = {
          email,
          name,
          phone,
          birthDate,
          gender,
          agreements
        };

        // 시뮬레이션
        await new Promise(resolve => setTimeout(resolve, 1500));

        const user = {
          username: name,
          role: 'customer' as const,
          token: `new_user_token_${Date.now()}`,
          email: email
        };

        AuthService.saveUser(user);
        router.push('/');
      }
    } catch (err) {
      setError('처리 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--ivory-light)' }}>
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-xl p-8 border border-opacity-20" style={{ borderColor: 'var(--vintage-gold)' }}>
          <div className="text-center mb-8">
            <Link
              href="/"
              className="text-3xl font-light tracking-[0.15em]"
              style={{
                fontFamily: 'Playfair Display, serif',
                color: 'var(--vintage-navy)'
              }}
            >
              Deulsoom
            </Link>
            <p className="text-sm mt-3" style={{ color: 'var(--vintage-gray)' }}>
              {isLogin ? '당신만의 향기로운 여정을 시작하세요' : '들숨과 함께 새로운 향의 세계로'}
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 rounded-lg" style={{ backgroundColor: 'var(--vintage-rose)', color: 'var(--deep-brown)' }}>
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          {/* 빠른 로그인 (테스트용) */}
          {isLogin && (
            <div className="mb-6 p-4 rounded-lg" style={{ backgroundColor: 'var(--ivory-dark)', borderColor: 'var(--vintage-gray-light)' }}>
              <p className="text-sm font-medium mb-3 text-center" style={{ color: 'var(--vintage-navy)' }}>
                빠른 로그인 (테스트)
              </p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => {
                    setEmail('customer');
                    setPassword('customer123');
                  }}
                  className="px-3 py-2 text-xs rounded-lg transition-all"
                  style={{
                    backgroundColor: 'var(--vintage-sage)',
                    color: 'white'
                  }}
                >
                  👤 고객 계정
                </button>
                <button
                  onClick={() => {
                    setEmail('admin');
                    setPassword('admin123!');
                  }}
                  className="px-3 py-2 text-xs rounded-lg transition-all"
                  style={{
                    backgroundColor: 'var(--vintage-gold)',
                    color: 'white'
                  }}
                >
                  🔑 관리자 계정
                </button>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                  이름
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                  style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                  placeholder="홍길동"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                이메일
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                placeholder="customer 또는 admin"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                비밀번호
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                placeholder="비밀번호 입력"
              />
            </div>

            {!isLogin && (
              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                  비밀번호 확인
                </label>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                  style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                  placeholder="비밀번호 다시 입력"
                />
              </div>
            )}

            {isLogin && (
              <div className="flex items-center justify-between">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="h-4 w-4 text-neutral-900 border-neutral-300 rounded"
                  />
                  <span className="ml-2 text-sm" style={{ color: 'var(--vintage-gray)' }}>로그인 상태 유지</span>
                </label>
                <Link href="/forgot-password" className="text-sm transition-colors" style={{ color: 'var(--vintage-gray)' }} onMouseEnter={(e) => e.target.style.color = 'var(--vintage-navy)'} onMouseLeave={(e) => e.target.style.color = 'var(--vintage-gray)'}>
                  비밀번호 찾기
                </Link>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-4 px-6 text-white font-medium rounded-lg transition-all transform hover:scale-[1.02] disabled:opacity-70 disabled:cursor-not-allowed"
              style={{ backgroundColor: 'var(--vintage-gold)' }}
              onMouseEnter={(e) => !loading && (e.currentTarget.style.backgroundColor = 'var(--vintage-gold-dark)')}
              onMouseLeave={(e) => !loading && (e.currentTarget.style.backgroundColor = 'var(--vintage-gold)')}
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
                  처리 중...
                </div>
              ) : (
                isLogin ? '로그인' : '회원가입'
              )}
            </button>
          </form>

          <div className="mt-6">
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t" style={{ borderColor: 'var(--vintage-gray-light)' }}></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="bg-white px-4" style={{ color: 'var(--vintage-gray)' }}>또는</span>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <button className="w-full flex items-center justify-center px-4 py-3 border rounded-lg transition-all" style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'white' }} onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--ivory-light)'} onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}>
                <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                  <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                  <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                  <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                  <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                <span style={{ color: 'var(--vintage-navy)' }}>Google로 계속하기</span>
              </button>

              <button className="w-full flex items-center justify-center px-4 py-3 border rounded-lg transition-all" style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'white' }} onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--ivory-light)'} onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}>
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2C6.477 2 2 6.477 2 12c0 4.991 3.657 9.128 8.438 9.879V14.89h-2.54V12h2.54V9.797c0-2.506 1.492-3.89 3.777-3.89 1.094 0 2.238.195 2.238.195v2.46h-1.26c-1.243 0-1.63.771-1.63 1.562V12h2.773l-.443 2.89h-2.33v6.989C18.343 21.129 22 16.99 22 12c0-5.523-4.477-10-10-10z"/>
                </svg>
                <span style={{ color: 'var(--vintage-navy)' }}>Facebook으로 계속하기</span>
              </button>
            </div>
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm" style={{ color: 'var(--vintage-gray)' }}>
              {isLogin ? '계정이 없으신가요?' : '이미 계정이 있으신가요?'}{' '}
              <button
                onClick={() => setIsLogin(!isLogin)}
                className="font-medium hover:underline transition-colors"
                style={{ color: 'var(--vintage-gold)' }}
                onMouseEnter={(e) => e.target.style.color = 'var(--vintage-gold-dark)'}
                onMouseLeave={(e) => e.target.style.color = 'var(--vintage-gold)'}
              >
                {isLogin ? '회원가입' : '로그인'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}