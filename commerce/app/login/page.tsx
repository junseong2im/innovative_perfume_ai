'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { AuthService, type LoginCredentials } from 'lib/auth';
import { SocialAuthService, socialProviders } from 'lib/social-auth';
import TermsModal from 'components/auth/terms-modal';
import UserService from '../../lib/user-service';

export default function LoginPage() {
  const router = useRouter();
  const userService = UserService.getInstance();
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

  // 유효성 검사 함수들
  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePassword = (password: string) => {
    return password.length >= 8 && /^(?=.*[a-zA-Z])(?=.*\d)/.test(password);
  };

  const validatePhone = (phone: string) => {
    const phoneRegex = /^01[0-9]-?[0-9]{4}-?[0-9]{4}$/;
    return phoneRegex.test(phone);
  };

  // 약관 동의 핸들러
  const handleAllAgreement = (checked: boolean) => {
    setAgreements({
      all: checked,
      terms: checked,
      privacy: checked,
      marketing: checked
    });
  };

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

  // 폼 제출 핸들러
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        // 통합 서비스를 통한 로그인 처리
        if (!validateEmail(email)) {
          setError('올바른 이메일 주소를 입력해주세요.');
          return;
        }

        // 통합 사용자 서비스로 인증
        const user = await userService.authenticateCustomer(email, password);

        if (user) {
          // 로그인 성공
          router.push('/');
        } else {
          // 기존 AuthService도 백업으로 시도
          const credentials: LoginCredentials = {
            username: email,
            password: password
          };

          const response = await AuthService.login(credentials);

          if (response.success && response.access_token) {
            const userData = {
              username: response.user_role || email.split('@')[0],
              role: response.user_role || 'customer',
              token: response.access_token
            };

            AuthService.saveUser(userData);
            router.push('/');
          } else {
            setError('이메일 또는 비밀번호가 올바르지 않습니다.');
          }
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

        // 통합 서비스를 통한 회원가입 처리
        const existingUser = userService.getUserByEmail(email);
        if (existingUser) {
          setError('이미 가입된 이메일입니다.');
          return;
        }

        // 새 사용자 추가
        const newUser = await userService.addUser({
          username: name,
          email: email,
          role: 'customer',
          status: 'active',
          profile: {
            firstName: name.split(' ')[0],
            lastName: name.split(' ').slice(1).join(' ') || '',
            phone: phone,
            birthday: birthDate,
            preferences: {
              fragrance_families: [],
              preferred_intensity: 'moderate',
              budget_range: '100000-300000'
            }
          }
        });

        // 로그인 처리
        await userService.authenticateCustomer(email, password);
        router.push('/');
      }
    } catch (err) {
      setError('처리 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  // 소셜 로그인 아이콘 컴포넌트
  const SocialIcon = ({ provider }: { provider: string }) => {
    switch (provider) {
      case 'google':
        return (
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
        );
      case 'facebook':
        return (
          <svg className="w-5 h-5" fill="#1877F2" viewBox="0 0 24 24">
            <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
          </svg>
        );
      case 'kakao':
        return (
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#000" d="M12 3C6.5 3 2 6.6 2 11.1c0 2.9 1.9 5.4 4.7 6.9l-1.2 4.4c-.1.4.4.7.7.4L10.3 19c.6.1 1.1.1 1.7.1 5.5 0 10-3.6 10-8.1S17.5 3 12 3z"/>
          </svg>
        );
      case 'naver':
        return (
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#03C75A" d="M16.273 12.845 7.376 0H0v24h7.726V11.156L16.624 24H24V0h-7.727v12.845z"/>
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center py-8" style={{ backgroundColor: 'var(--ivory-light)' }}>
      <div className="w-full max-w-lg">
        <div className="bg-white rounded-xl shadow-xl p-8 border border-opacity-20" style={{ borderColor: 'var(--vintage-gold)' }}>
          {/* 헤더 */}
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

          {/* 오류 메시지 */}
          {error && (
            <div className="mb-6 p-4 rounded-lg" style={{ backgroundColor: 'var(--vintage-rose)', color: 'var(--deep-brown)' }}>
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}


          {/* 소셜 로그인 */}
          <div className="mb-6">
            <div className="grid grid-cols-2 gap-3">
              {socialProviders.slice(0, 4).map((provider) => (
                <button
                  key={provider.provider}
                  onClick={() => handleSocialLogin(provider.provider)}
                  disabled={socialLoading === provider.provider}
                  className="flex items-center justify-center px-4 py-3 border rounded-lg transition-all hover:scale-[1.02] disabled:opacity-70"
                  style={{
                    borderColor: 'var(--vintage-gray-light)',
                    backgroundColor: 'white'
                  }}
                >
                  {socialLoading === provider.provider ? (
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2" style={{ borderColor: provider.color }} />
                  ) : (
                    <>
                      <SocialIcon provider={provider.provider} />
                      <span className="ml-2 text-sm" style={{ color: 'var(--vintage-navy)' }}>
                        {provider.provider.charAt(0).toUpperCase() + provider.provider.slice(1)}
                      </span>
                    </>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* 구분선 */}
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t" style={{ borderColor: 'var(--vintage-gray-light)' }}></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="bg-white px-4" style={{ color: 'var(--vintage-gray)' }}>또는</span>
            </div>
          </div>

          {/* 로그인/회원가입 폼 */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* 이름 (회원가입시만) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                  이름 <span className="text-red-500">*</span>
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

            {/* 이메일 */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                이메일 <span className="text-red-500">*</span>
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                placeholder={isLogin ? "customer 또는 admin" : "example@email.com"}
              />
            </div>

            {/* 비밀번호 */}
            <div>
              <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                비밀번호 <span className="text-red-500">*</span>
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
              {!isLogin && (
                <p className="text-xs mt-1" style={{ color: 'var(--vintage-gray)' }}>
                  8자 이상, 영문과 숫자 포함
                </p>
              )}
            </div>

            {/* 회원가입 추가 필드들 */}
            {!isLogin && (
              <>
                {/* 비밀번호 확인 */}
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                    비밀번호 확인 <span className="text-red-500">*</span>
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

                {/* 휴대폰 번호 */}
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                    휴대폰 번호 <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="tel"
                    value={phone}
                    onChange={(e) => setPhone(e.target.value)}
                    required
                    className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                    style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                    placeholder="010-1234-5678"
                  />
                </div>

                {/* 생년월일 & 성별 */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                      생년월일
                    </label>
                    <input
                      type="date"
                      value={birthDate}
                      onChange={(e) => setBirthDate(e.target.value)}
                      className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                      style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                      성별
                    </label>
                    <select
                      value={gender}
                      onChange={(e) => setGender(e.target.value)}
                      className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                      style={{ borderColor: 'var(--vintage-gray-light)', backgroundColor: 'var(--ivory-light)', color: 'var(--vintage-navy)' }}
                    >
                      <option value="">선택 안함</option>
                      <option value="male">남성</option>
                      <option value="female">여성</option>
                    </select>
                  </div>
                </div>

                {/* 약관 동의 섹션 */}
                <div className="space-y-4 p-4 rounded-lg" style={{ backgroundColor: 'var(--ivory-dark)', border: '1px solid var(--vintage-gray-light)' }}>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="agreeAll"
                      checked={agreements.all}
                      onChange={(e) => handleAllAgreement(e.target.checked)}
                      className="h-5 w-5 rounded border-2"
                      style={{ accentColor: 'var(--vintage-gold)' }}
                    />
                    <label htmlFor="agreeAll" className="ml-3 text-sm font-medium" style={{ color: 'var(--vintage-navy)' }}>
                      전체 동의
                    </label>
                  </div>

                  <div className="border-t pt-3 space-y-2" style={{ borderColor: 'var(--vintage-gray-light)' }}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="agreeTerms"
                          checked={agreements.terms}
                          onChange={(e) => handleAgreementChange('terms', e.target.checked)}
                          className="h-4 w-4 rounded"
                          style={{ accentColor: 'var(--vintage-gold)' }}
                        />
                        <label htmlFor="agreeTerms" className="ml-2 text-sm" style={{ color: 'var(--vintage-navy)' }}>
                          이용약관 동의 <span className="text-red-500">*</span>
                        </label>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setModalType('terms');
                          setShowTermsModal(true);
                        }}
                        className="text-xs underline" style={{ color: 'var(--vintage-gray)' }}
                      >
                        전문보기
                      </button>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          id="agreePrivacy"
                          checked={agreements.privacy}
                          onChange={(e) => handleAgreementChange('privacy', e.target.checked)}
                          className="h-4 w-4 rounded"
                          style={{ accentColor: 'var(--vintage-gold)' }}
                        />
                        <label htmlFor="agreePrivacy" className="ml-2 text-sm" style={{ color: 'var(--vintage-navy)' }}>
                          개인정보 처리방침 동의 <span className="text-red-500">*</span>
                        </label>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setModalType('privacy');
                          setShowTermsModal(true);
                        }}
                        className="text-xs underline" style={{ color: 'var(--vintage-gray)' }}
                      >
                        전문보기
                      </button>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="agreeMarketing"
                        checked={agreements.marketing}
                        onChange={(e) => handleAgreementChange('marketing', e.target.checked)}
                        className="h-4 w-4 rounded"
                        style={{ accentColor: 'var(--vintage-gold)' }}
                      />
                      <label htmlFor="agreeMarketing" className="ml-2 text-sm" style={{ color: 'var(--vintage-gray)' }}>
                        마케팅 정보 수신 동의 (선택)
                      </label>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* 로그인 상태 유지 & 비밀번호 찾기 */}
            {isLogin && (
              <div className="flex items-center justify-between">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    className="h-4 w-4 rounded"
                    style={{ accentColor: 'var(--vintage-gold)' }}
                  />
                  <span className="ml-2 text-sm" style={{ color: 'var(--vintage-gray)' }}>로그인 상태 유지</span>
                </label>
                <Link href="/forgot-password" className="text-sm transition-colors" style={{ color: 'var(--vintage-gray)' }}>
                  비밀번호 찾기
                </Link>
              </div>
            )}

            {/* 제출 버튼 */}
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

          {/* 로그인/회원가입 토글 */}
          <div className="mt-8 text-center space-y-2">
            <p className="text-sm" style={{ color: 'var(--vintage-gray)' }}>
              {isLogin ? '계정이 없으신가요?' : '이미 계정이 있으신가요?'}{' '}
              <button
                onClick={() => {
                  setIsLogin(!isLogin);
                  setError('');
                  setAgreements({ all: false, terms: false, privacy: false, marketing: false });
                }}
                className="font-medium hover:underline transition-colors"
                style={{ color: 'var(--vintage-gold)' }}
              >
                {isLogin ? '회원가입' : '로그인'}
              </button>
            </p>

            {/* 관리자 로그인 링크 */}
            {isLogin && (
              <div className="pt-4 border-t" style={{ borderColor: 'var(--vintage-gray-light)' }}>
                <Link
                  href="/admin/login"
                  className="inline-flex items-center text-xs transition-colors"
                  style={{ color: 'var(--vintage-gray)' }}
                >
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                  관리자 로그인
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 약관 모달 */}
      <TermsModal
        isOpen={showTermsModal}
        onClose={() => setShowTermsModal(false)}
        type={modalType}
      />
    </div>
  );
}