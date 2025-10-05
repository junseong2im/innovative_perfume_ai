'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { authService, LoginCredentials } from '@/lib/auth';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial } from '@react-three/drei';
import { FiEye, FiEyeOff, FiLock, FiMail, FiUser, FiSmartphone, FiCheck, FiX, FiArrowRight } from 'react-icons/fi';
import { FaGoogle, FaFacebook, FaGithub, FaApple } from 'react-icons/fa';
import { RiKakaoTalkFill } from 'react-icons/ri';

// 3D 애니메이션 배경
function AnimatedSphere() {
  const meshRef = useRef<any>();

  useEffect(() => {
    const interval = setInterval(() => {
      if (meshRef.current) {
        meshRef.current.rotation.x += 0.001;
        meshRef.current.rotation.y += 0.002;
      }
    }, 10);
    return () => clearInterval(interval);
  }, []);

  return (
    <Sphere ref={meshRef} args={[1, 100, 200]} scale={2.5}>
      <MeshDistortMaterial
        color="#8B6F47"
        attach="material"
        distort={0.5}
        speed={2}
        roughness={0.2}
        metalness={0.8}
      />
    </Sphere>
  );
}

// 비밀번호 강도 표시기
function PasswordStrength({ password }: { password: string }) {
  const strength = authService.checkPasswordStrength(password);
  const colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e'];
  const labels = ['매우 약함', '약함', '보통', '강함', '매우 강함'];

  return (
    <div className="space-y-2">
      <div className="flex gap-1">
        {[0, 1, 2, 3, 4].map((level) => (
          <div
            key={level}
            className={`h-1 flex-1 rounded-full transition-all duration-300 ${
              level <= strength.score
                ? `bg-gradient-to-r from-${colors[strength.score]} to-${colors[strength.score]}`
                : 'bg-gray-200'
            }`}
            style={{
              backgroundColor: level <= strength.score ? colors[strength.score] : undefined
            }}
          />
        ))}
      </div>
      {password && (
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium" style={{ color: colors[strength.score] }}>
            {labels[strength.score]}
          </span>
          <span className="text-gray-500">{strength.crackTime}</span>
        </div>
      )}
      {strength.suggestions.length > 0 && (
        <ul className="text-xs text-gray-600 space-y-1">
          {strength.suggestions.map((suggestion, idx) => (
            <li key={idx} className="flex items-start gap-1">
              <span className="text-amber-500 mt-0.5">•</span>
              <span>{suggestion}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<'login' | 'register' | 'reset'>('login');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [acceptTerms, setAcceptTerms] = useState(false);
  const [marketingConsent, setMarketingConsent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [twoFARequired, setTwoFARequired] = useState(false);
  const [twoFAMethod, setTwoFAMethod] = useState<'sms' | 'email' | 'authenticator'>('email');
  const [twoFACode, setTwoFACode] = useState('');

  // 폼 데이터
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    username: '',
    firstName: '',
    lastName: ''
  });

  // 실시간 유효성 검사
  const [validation, setValidation] = useState({
    email: { valid: false, message: '' },
    password: { valid: false, message: '' },
    confirmPassword: { valid: false, message: '' },
    username: { valid: false, message: '' }
  });

  // 이메일 유효성 검사
  const validateEmail = (email: string) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email) {
      return { valid: false, message: '' };
    }
    if (!regex.test(email)) {
      return { valid: false, message: '올바른 이메일 형식이 아닙니다' };
    }
    return { valid: true, message: '' };
  };

  // 비밀번호 확인
  const validatePasswordMatch = (password: string, confirmPassword: string) => {
    if (!confirmPassword) {
      return { valid: false, message: '' };
    }
    if (password !== confirmPassword) {
      return { valid: false, message: '비밀번호가 일치하지 않습니다' };
    }
    return { valid: true, message: '' };
  };

  // 사용자명 유효성 검사
  const validateUsername = (username: string) => {
    if (!username) {
      return { valid: false, message: '' };
    }
    if (username.length < 3) {
      return { valid: false, message: '최소 3자 이상이어야 합니다' };
    }
    if (!/^[a-zA-Z0-9_]+$/.test(username)) {
      return { valid: false, message: '영문, 숫자, 언더스코어만 가능합니다' };
    }
    return { valid: true, message: '' };
  };

  // 입력 변경 핸들러
  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));

    // 실시간 유효성 검사
    switch (field) {
      case 'email':
        setValidation(prev => ({
          ...prev,
          email: validateEmail(value)
        }));
        break;
      case 'password':
        setValidation(prev => ({
          ...prev,
          password: authService.checkPasswordStrength(value).score >= 3
            ? { valid: true, message: '' }
            : { valid: false, message: '비밀번호 강도가 부족합니다' }
        }));
        if (formData.confirmPassword) {
          setValidation(prev => ({
            ...prev,
            confirmPassword: validatePasswordMatch(value, formData.confirmPassword)
          }));
        }
        break;
      case 'confirmPassword':
        setValidation(prev => ({
          ...prev,
          confirmPassword: validatePasswordMatch(formData.password, value)
        }));
        break;
      case 'username':
        setValidation(prev => ({
          ...prev,
          username: validateUsername(value)
        }));
        break;
    }
  };

  // 로그인 처리
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const result = await authService.login({
        email: formData.email,
        password: formData.password,
        rememberMe
      });

      if (result.requiresTwoFactor) {
        setTwoFARequired(true);
        setTwoFAMethod(result.twoFactorMethod || 'email');
        setSuccess('2단계 인증 코드를 입력해주세요');
      } else if (result.success) {
        setSuccess('로그인 성공! 리다이렉트 중...');
        setTimeout(() => {
          router.push('/dashboard');
        }, 1500);
      } else {
        setError(result.message || '로그인에 실패했습니다');
      }
    } catch (err: any) {
      setError(err.message || '로그인 중 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  // 회원가입 처리
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // 유효성 검사
    if (!validation.email.valid || !validation.password.valid ||
        !validation.confirmPassword.valid || !validation.username.valid) {
      setError('모든 필드를 올바르게 입력해주세요');
      setLoading(false);
      return;
    }

    if (!acceptTerms) {
      setError('이용약관에 동의해주세요');
      setLoading(false);
      return;
    }

    try {
      const result = await authService.register({
        email: formData.email,
        password: formData.password,
        username: formData.username,
        firstName: formData.firstName,
        lastName: formData.lastName,
        acceptTerms,
        marketingConsent
      });

      if (result.success) {
        setSuccess('회원가입 성공! 이메일을 확인해주세요');
        setTimeout(() => {
          setMode('login');
        }, 3000);
      } else {
        setError(result.message || '회원가입에 실패했습니다');
      }
    } catch (err: any) {
      setError(err.message || '회원가입 중 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  // 2FA 검증
  const handleVerify2FA = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await authService.verifyTwoFactor(twoFACode, twoFAMethod);

      if (result.success) {
        setSuccess('인증 성공! 리다이렉트 중...');
        setTimeout(() => {
          router.push('/dashboard');
        }, 1500);
      } else {
        setError(result.message || '인증에 실패했습니다');
      }
    } catch (err: any) {
      setError(err.message || '인증 중 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  // 비밀번호 재설정
  const handlePasswordReset = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const success = await authService.requestPasswordReset(formData.email);

      if (success) {
        setSuccess('비밀번호 재설정 링크를 이메일로 전송했습니다');
        setTimeout(() => {
          setMode('login');
        }, 3000);
      } else {
        setError('비밀번호 재설정 요청에 실패했습니다');
      }
    } catch (err: any) {
      setError(err.message || '오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  // OAuth 로그인
  const handleOAuthLogin = async (provider: 'google' | 'facebook' | 'github' | 'apple' | 'kakao') => {
    try {
      await authService.oauthLogin(provider);
    } catch (err: any) {
      setError(err.message || 'OAuth 로그인 중 오류가 발생했습니다');
    }
  };

  // 생체 인증
  const handleBiometricLogin = async () => {
    setLoading(true);
    try {
      const result = await authService.biometricLogin();

      if (result.success) {
        setSuccess('생체 인증 성공! 리다이렉트 중...');
        setTimeout(() => {
          router.push('/dashboard');
        }, 1500);
      } else {
        setError(result.message || '생체 인증에 실패했습니다');
      }
    } catch (err: any) {
      setError(err.message || '생체 인증 중 오류가 발생했습니다');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FAF7F2] via-white to-[#F5EDE4] relative overflow-hidden">
      {/* 3D 배경 */}
      <div className="absolute inset-0 opacity-10">
        <Canvas>
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} />
          <AnimatedSphere />
          <OrbitControls enableZoom={false} enablePan={false} />
        </Canvas>
      </div>

      {/* 네비게이션 */}
      <nav className="relative z-10 flex justify-between items-center p-8">
        <Link href="/" className="text-2xl font-serif tracking-wider text-[#8B6F47]">
          Deulsoom
        </Link>
        <Link href="/" className="text-sm text-gray-600 hover:text-[#8B6F47] transition-colors">
          홈으로 돌아가기
        </Link>
      </nav>

      {/* 메인 컨테이너 */}
      <div className="relative z-10 max-w-6xl mx-auto px-8 py-12">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* 왼쪽: 브랜딩 */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="hidden lg:block space-y-8"
          >
            <div>
              <h1 className="text-5xl font-serif mb-4 text-[#8B6F47]">
                당신만의<br />향기 여정을<br />시작하세요
              </h1>
              <p className="text-gray-600 leading-relaxed">
                AI가 만들어내는 개인 맞춤형 향수,<br />
                당신의 이야기를 향기로 담아냅니다
              </p>
            </div>

            {/* 특징 리스트 */}
            <div className="space-y-4">
              {[
                '개인 맞춤형 AI 향수 제작',
                '3D DNA 시각화 기술',
                '실시간 협업 및 공유',
                '프리미엄 향료 데이터베이스'
              ].map((feature, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 + idx * 0.1 }}
                  className="flex items-center gap-3"
                >
                  <div className="w-8 h-8 rounded-full bg-[#8B6F47]/10 flex items-center justify-center">
                    <FiCheck className="text-[#8B6F47] text-sm" />
                  </div>
                  <span className="text-gray-700">{feature}</span>
                </motion.div>
              ))}
            </div>

            {/* 소셜 증명 */}
            <div className="pt-8 border-t border-gray-200">
              <div className="flex items-center gap-4 mb-4">
                <div className="flex -space-x-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-10 h-10 rounded-full bg-gradient-to-br from-[#8B6F47] to-[#6B5637] border-2 border-white"
                    />
                  ))}
                </div>
                <span className="text-sm text-gray-600">
                  <strong>10,000+</strong> 명이 함께하고 있습니다
                </span>
              </div>
            </div>
          </motion.div>

          {/* 오른쪽: 폼 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="bg-white/80 backdrop-blur-lg rounded-3xl shadow-2xl p-8 lg:p-10"
          >
            {/* 2FA 입력 폼 */}
            {twoFARequired ? (
              <form onSubmit={handleVerify2FA} className="space-y-6">
                <div>
                  <h2 className="text-2xl font-semibold mb-2">2단계 인증</h2>
                  <p className="text-gray-600 text-sm">
                    {twoFAMethod === 'email' && '이메일로 전송된 '}
                    {twoFAMethod === 'sms' && 'SMS로 전송된 '}
                    {twoFAMethod === 'authenticator' && '인증 앱에 표시된 '}
                    6자리 코드를 입력하세요
                  </p>
                </div>

                <div className="space-y-4">
                  <input
                    type="text"
                    value={twoFACode}
                    onChange={(e) => setTwoFACode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    placeholder="000000"
                    className="w-full text-center text-2xl tracking-widest p-4 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none"
                    maxLength={6}
                    autoComplete="one-time-code"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading || twoFACode.length !== 6}
                  className="w-full py-4 bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white rounded-xl font-medium hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? '확인 중...' : '인증 확인'}
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setTwoFARequired(false);
                    setTwoFACode('');
                  }}
                  className="w-full text-gray-600 hover:text-[#8B6F47] transition-colors text-sm"
                >
                  돌아가기
                </button>
              </form>
            ) : (
              <>
                {/* 탭 선택 */}
                <div className="flex rounded-xl bg-gray-100 p-1 mb-8">
                  {['login', 'register'].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setMode(tab as 'login' | 'register')}
                      className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                        mode === tab
                          ? 'bg-white text-[#8B6F47] shadow-sm'
                          : 'text-gray-600 hover:text-gray-900'
                      }`}
                    >
                      {tab === 'login' ? '로그인' : '회원가입'}
                    </button>
                  ))}
                </div>

                {/* 에러/성공 메시지 */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-2 text-red-700"
                    >
                      <FiX />
                      <span className="text-sm">{error}</span>
                    </motion.div>
                  )}
                  {success && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl flex items-center gap-2 text-green-700"
                    >
                      <FiCheck />
                      <span className="text-sm">{success}</span>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* 로그인 폼 */}
                {mode === 'login' && (
                  <form onSubmit={handleLogin} className="space-y-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        이메일
                      </label>
                      <div className="relative">
                        <FiMail className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => handleInputChange('email', e.target.value)}
                          className="w-full pl-12 pr-4 py-3.5 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none transition-colors"
                          placeholder="your@email.com"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        비밀번호
                      </label>
                      <div className="relative">
                        <FiLock className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={formData.password}
                          onChange={(e) => handleInputChange('password', e.target.value)}
                          className="w-full pl-12 pr-12 py-3.5 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none transition-colors"
                          placeholder="••••••••"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-4 top-4 text-gray-400 hover:text-gray-600"
                        >
                          {showPassword ? <FiEyeOff /> : <FiEye />}
                        </button>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={rememberMe}
                          onChange={(e) => setRememberMe(e.target.checked)}
                          className="w-4 h-4 text-[#8B6F47] border-gray-300 rounded focus:ring-[#8B6F47]"
                        />
                        <span className="text-sm text-gray-600">로그인 상태 유지</span>
                      </label>
                      <button
                        type="button"
                        onClick={() => setMode('reset')}
                        className="text-sm text-[#8B6F47] hover:underline"
                      >
                        비밀번호 찾기
                      </button>
                    </div>

                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full py-4 bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white rounded-xl font-medium hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {loading ? '로그인 중...' : '로그인'}
                      {!loading && <FiArrowRight />}
                    </button>

                    {/* 생체 인증 버튼 */}
                    {typeof window !== 'undefined' && window.PublicKeyCredential && (
                      <button
                        type="button"
                        onClick={handleBiometricLogin}
                        className="w-full py-3 border-2 border-[#8B6F47] text-[#8B6F47] rounded-xl font-medium hover:bg-[#8B6F47] hover:text-white transition-all flex items-center justify-center gap-2"
                      >
                        <FiSmartphone />
                        생체 인증으로 로그인
                      </button>
                    )}

                    <div className="relative">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-200"></div>
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-4 bg-white text-gray-500">또는</span>
                      </div>
                    </div>

                    {/* OAuth 버튼들 */}
                    <div className="grid grid-cols-2 gap-3">
                      <button
                        type="button"
                        onClick={() => handleOAuthLogin('google')}
                        className="py-3 px-4 border-2 border-gray-200 rounded-xl hover:border-gray-300 transition-colors flex items-center justify-center gap-2"
                      >
                        <FaGoogle className="text-red-500" />
                        <span className="text-sm font-medium">Google</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOAuthLogin('facebook')}
                        className="py-3 px-4 border-2 border-gray-200 rounded-xl hover:border-gray-300 transition-colors flex items-center justify-center gap-2"
                      >
                        <FaFacebook className="text-blue-600" />
                        <span className="text-sm font-medium">Facebook</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOAuthLogin('github')}
                        className="py-3 px-4 border-2 border-gray-200 rounded-xl hover:border-gray-300 transition-colors flex items-center justify-center gap-2"
                      >
                        <FaGithub className="text-gray-800" />
                        <span className="text-sm font-medium">GitHub</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => handleOAuthLogin('kakao')}
                        className="py-3 px-4 border-2 border-gray-200 rounded-xl hover:border-gray-300 transition-colors flex items-center justify-center gap-2"
                      >
                        <RiKakaoTalkFill className="text-yellow-500" />
                        <span className="text-sm font-medium">카카오</span>
                      </button>
                    </div>
                  </form>
                )}

                {/* 회원가입 폼 */}
                {mode === 'register' && (
                  <form onSubmit={handleRegister} className="space-y-5">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          이름
                        </label>
                        <input
                          type="text"
                          value={formData.firstName}
                          onChange={(e) => handleInputChange('firstName', e.target.value)}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none"
                          placeholder="길동"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          성
                        </label>
                        <input
                          type="text"
                          value={formData.lastName}
                          onChange={(e) => handleInputChange('lastName', e.target.value)}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none"
                          placeholder="홍"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        사용자명
                      </label>
                      <div className="relative">
                        <FiUser className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type="text"
                          value={formData.username}
                          onChange={(e) => handleInputChange('username', e.target.value)}
                          className={`w-full pl-12 pr-4 py-3.5 border-2 rounded-xl focus:outline-none transition-colors ${
                            validation.username.valid
                              ? 'border-green-400 focus:border-green-500'
                              : formData.username && !validation.username.valid
                              ? 'border-red-400 focus:border-red-500'
                              : 'border-gray-200 focus:border-[#8B6F47]'
                          }`}
                          placeholder="username"
                          required
                        />
                        {validation.username.valid && (
                          <FiCheck className="absolute right-4 top-4 text-green-500" />
                        )}
                      </div>
                      {validation.username.message && (
                        <p className="mt-1 text-xs text-red-500">{validation.username.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        이메일
                      </label>
                      <div className="relative">
                        <FiMail className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => handleInputChange('email', e.target.value)}
                          className={`w-full pl-12 pr-4 py-3.5 border-2 rounded-xl focus:outline-none transition-colors ${
                            validation.email.valid
                              ? 'border-green-400 focus:border-green-500'
                              : formData.email && !validation.email.valid
                              ? 'border-red-400 focus:border-red-500'
                              : 'border-gray-200 focus:border-[#8B6F47]'
                          }`}
                          placeholder="your@email.com"
                          required
                        />
                        {validation.email.valid && (
                          <FiCheck className="absolute right-4 top-4 text-green-500" />
                        )}
                      </div>
                      {validation.email.message && (
                        <p className="mt-1 text-xs text-red-500">{validation.email.message}</p>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        비밀번호
                      </label>
                      <div className="relative">
                        <FiLock className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={formData.password}
                          onChange={(e) => handleInputChange('password', e.target.value)}
                          className="w-full pl-12 pr-12 py-3.5 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none"
                          placeholder="••••••••"
                          required
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-4 top-4 text-gray-400 hover:text-gray-600"
                        >
                          {showPassword ? <FiEyeOff /> : <FiEye />}
                        </button>
                      </div>
                      {formData.password && (
                        <div className="mt-2">
                          <PasswordStrength password={formData.password} />
                        </div>
                      )}
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        비밀번호 확인
                      </label>
                      <div className="relative">
                        <FiLock className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type="password"
                          value={formData.confirmPassword}
                          onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                          className={`w-full pl-12 pr-4 py-3.5 border-2 rounded-xl focus:outline-none transition-colors ${
                            validation.confirmPassword.valid
                              ? 'border-green-400 focus:border-green-500'
                              : formData.confirmPassword && !validation.confirmPassword.valid
                              ? 'border-red-400 focus:border-red-500'
                              : 'border-gray-200 focus:border-[#8B6F47]'
                          }`}
                          placeholder="••••••••"
                          required
                        />
                        {validation.confirmPassword.valid && (
                          <FiCheck className="absolute right-4 top-4 text-green-500" />
                        )}
                      </div>
                      {validation.confirmPassword.message && (
                        <p className="mt-1 text-xs text-red-500">{validation.confirmPassword.message}</p>
                      )}
                    </div>

                    <div className="space-y-3">
                      <label className="flex items-start gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={acceptTerms}
                          onChange={(e) => setAcceptTerms(e.target.checked)}
                          className="w-4 h-4 mt-0.5 text-[#8B6F47] border-gray-300 rounded focus:ring-[#8B6F47]"
                          required
                        />
                        <span className="text-sm text-gray-600">
                          <Link href="/terms" className="text-[#8B6F47] hover:underline">
                            이용약관
                          </Link>
                          {' 및 '}
                          <Link href="/privacy" className="text-[#8B6F47] hover:underline">
                            개인정보처리방침
                          </Link>
                          에 동의합니다
                        </span>
                      </label>
                      <label className="flex items-start gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={marketingConsent}
                          onChange={(e) => setMarketingConsent(e.target.checked)}
                          className="w-4 h-4 mt-0.5 text-[#8B6F47] border-gray-300 rounded focus:ring-[#8B6F47]"
                        />
                        <span className="text-sm text-gray-600">
                          마케팅 정보 수신에 동의합니다 (선택)
                        </span>
                      </label>
                    </div>

                    <button
                      type="submit"
                      disabled={loading || !acceptTerms || !validation.email.valid ||
                               !validation.password.valid || !validation.confirmPassword.valid ||
                               !validation.username.valid}
                      className="w-full py-4 bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white rounded-xl font-medium hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? '가입 중...' : '회원가입'}
                    </button>
                  </form>
                )}

                {/* 비밀번호 재설정 폼 */}
                {mode === 'reset' && (
                  <form onSubmit={handlePasswordReset} className="space-y-6">
                    <div>
                      <h2 className="text-2xl font-semibold mb-2">비밀번호 재설정</h2>
                      <p className="text-gray-600 text-sm">
                        가입하신 이메일로 비밀번호 재설정 링크를 보내드립니다
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        이메일
                      </label>
                      <div className="relative">
                        <FiMail className="absolute left-4 top-4 text-gray-400" />
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => handleInputChange('email', e.target.value)}
                          className="w-full pl-12 pr-4 py-3.5 border-2 border-gray-200 rounded-xl focus:border-[#8B6F47] focus:outline-none"
                          placeholder="your@email.com"
                          required
                        />
                      </div>
                    </div>

                    <button
                      type="submit"
                      disabled={loading || !validation.email.valid}
                      className="w-full py-4 bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white rounded-xl font-medium hover:shadow-lg transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? '전송 중...' : '재설정 링크 전송'}
                    </button>

                    <button
                      type="button"
                      onClick={() => setMode('login')}
                      className="w-full text-gray-600 hover:text-[#8B6F47] transition-colors text-sm"
                    >
                      로그인으로 돌아가기
                    </button>
                  </form>
                )}
              </>
            )}
          </motion.div>
        </div>
      </div>

      {/* 푸터 */}
      <footer className="relative z-10 mt-20 py-8 text-center text-sm text-gray-500">
        <p>&copy; 2025 Deulsoom. All rights reserved.</p>
      </footer>
    </div>
  );
}