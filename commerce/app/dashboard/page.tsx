'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { authService, User } from '@/lib/auth';
import {
  FiUser, FiSettings, FiPackage, FiHeart, FiClock, FiLogOut,
  FiShield, FiBell, FiCreditCard, FiMail, FiSmartphone,
  FiDatabase, FiTrendingUp, FiAward, FiBarChart, FiCalendar,
  FiEdit3, FiSave, FiX, FiCheck, FiAlertCircle, FiPlus
} from 'react-icons/fi';
import { IoSparklesSharp } from 'react-icons/io5';

// 대시보드 카드 컴포넌트
function DashboardCard({
  title,
  value,
  icon: Icon,
  change,
  color = 'indigo'
}: {
  title: string;
  value: string | number;
  icon: any;
  change?: string;
  color?: string;
}) {
  const colorClasses = {
    indigo: 'from-indigo-500 to-indigo-600',
    emerald: 'from-emerald-500 to-emerald-600',
    amber: 'from-amber-500 to-amber-600',
    rose: 'from-rose-500 to-rose-600'
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100"
    >
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-xl bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} text-white`}>
          <Icon size={24} />
        </div>
        {change && (
          <span className={`text-sm font-medium ${change.startsWith('+') ? 'text-green-600' : 'text-red-600'}`}>
            {change}
          </span>
        )}
      </div>
      <h3 className="text-gray-600 text-sm font-medium mb-1">{title}</h3>
      <p className="text-2xl font-bold text-gray-900">{value}</p>
    </motion.div>
  );
}

// 향수 히스토리 아이템
function FragranceHistoryItem({
  fragrance
}: {
  fragrance: {
    id: string;
    name: string;
    createdAt: string;
    rating: number;
    dnaId: string;
    thumbnail?: string;
  }
}) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      whileHover={{ scale: 1.02 }}
      className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors cursor-pointer"
    >
      <div className="w-16 h-16 rounded-lg bg-gradient-to-br from-[#8B6F47] to-[#6B5637] flex items-center justify-center text-white">
        <IoSparklesSharp size={24} />
      </div>
      <div className="flex-1">
        <h4 className="font-medium text-gray-900">{fragrance.name}</h4>
        <p className="text-sm text-gray-600">DNA ID: {fragrance.dnaId}</p>
        <p className="text-xs text-gray-500">{fragrance.createdAt}</p>
      </div>
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span
            key={i}
            className={`text-sm ${i < fragrance.rating ? 'text-amber-400' : 'text-gray-300'}`}
          >
            ★
          </span>
        ))}
      </div>
    </motion.div>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'profile' | 'settings' | 'billing'>('overview');
  const [editMode, setEditMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // 프로필 편집 폼
  const [profileForm, setProfileForm] = useState({
    firstName: '',
    lastName: '',
    email: '',
    language: 'ko',
    currency: 'KRW',
    theme: 'light' as 'light' | 'dark' | 'auto',
    notifications: {
      email: true,
      push: false,
      sms: false,
      marketing: false
    }
  });

  // 보안 설정
  const [securitySettings, setSecuritySettings] = useState({
    twoFactorEnabled: false,
    biometricEnabled: false,
    passwordChanged: ''
  });

  // 향수 관련 통계 (실제로는 API에서 가져올 데이터)
  const [stats] = useState({
    totalFragrances: 24,
    uniqueDNA: 156,
    favoriteAccord: 'Floral Oriental',
    scentComplexity: 8.5,
    topNote: 'Bergamot',
    heartNote: 'Rose',
    baseNote: 'Sandalwood',
    olfactoryFamily: 'Woody Floral',
    creationStreak: 7,
    rareIngredients: 12
  });

  // 향수 히스토리 (실제로는 API에서 가져올 데이터)
  const [fragranceHistory] = useState([
    {
      id: '1',
      name: '봄날의 산책',
      createdAt: '2025-01-25',
      rating: 5,
      dnaId: 'DNA-2025-0125-001',
      thumbnail: ''
    },
    {
      id: '2',
      name: '여름밤의 꿈',
      createdAt: '2025-01-20',
      rating: 4,
      dnaId: 'DNA-2025-0120-002',
      thumbnail: ''
    },
    {
      id: '3',
      name: '가을 숲의 향기',
      createdAt: '2025-01-15',
      rating: 5,
      dnaId: 'DNA-2025-0115-003',
      thumbnail: ''
    }
  ]);

  useEffect(() => {
    loadUserData();
  }, []);

  const loadUserData = async () => {
    try {
      const currentUser = await authService.getCurrentUser();

      if (!currentUser) {
        router.push('/auth/login');
        return;
      }

      setUser(currentUser);
      setProfileForm({
        firstName: currentUser.firstName,
        lastName: currentUser.lastName,
        email: currentUser.email,
        language: currentUser.preferences?.language || 'ko',
        currency: currentUser.preferences?.currency || 'KRW',
        theme: currentUser.preferences?.theme || 'light',
        notifications: currentUser.preferences?.notifications || {
          email: true,
          push: false,
          sms: false,
          marketing: false
        }
      });
      setSecuritySettings({
        twoFactorEnabled: currentUser.twoFactorEnabled || false,
        biometricEnabled: currentUser.biometricEnabled || false,
        passwordChanged: ''
      });
    } catch (error) {
      console.error('Failed to load user data:', error);
      router.push('/auth/login');
    } finally {
      setLoading(false);
    }
  };

  const handleProfileSave = async () => {
    setSaving(true);
    setMessage(null);

    try {
      const updatedUser = await authService.updateProfile({
        firstName: profileForm.firstName,
        lastName: profileForm.lastName,
        preferences: {
          language: profileForm.language,
          currency: profileForm.currency,
          theme: profileForm.theme,
          notifications: profileForm.notifications,
          fragranceIntensity: user?.preferences?.fragranceIntensity || 'moderate'
        }
      });

      if (updatedUser) {
        setUser(updatedUser);
        setEditMode(false);
        setMessage({ type: 'success', text: '프로필이 성공적으로 업데이트되었습니다!' });
      } else {
        setMessage({ type: 'error', text: '프로필 업데이트에 실패했습니다.' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: '오류가 발생했습니다. 다시 시도해주세요.' });
    } finally {
      setSaving(false);
    }
  };

  const handleLogout = async () => {
    await authService.logout();
    router.push('/');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#FAF7F2] to-white">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-[#8B6F47] border-t-transparent"></div>
          <p className="mt-4 text-gray-600">로딩 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FAF7F2] via-white to-[#F5EDE4]">
      {/* 헤더 */}
      <header className="bg-white/80 backdrop-blur-lg border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-8">
              <Link href="/" className="text-2xl font-serif tracking-wider text-[#8B6F47]">
                Deulsoom
              </Link>
              <nav className="hidden md:flex gap-6">
                <Link href="/create" className="text-gray-600 hover:text-[#8B6F47] transition-colors">
                  향수 만들기
                </Link>
                <Link href="/products" className="text-gray-600 hover:text-[#8B6F47] transition-colors">
                  제품 보기
                </Link>
                <Link href="/about" className="text-gray-600 hover:text-[#8B6F47] transition-colors">
                  소개
                </Link>
              </nav>
            </div>
            <div className="flex items-center gap-4">
              <button className="p-2 text-gray-600 hover:text-[#8B6F47] transition-colors">
                <FiBell size={20} />
              </button>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-900">
                    {user?.firstName} {user?.lastName}
                  </p>
                  <p className="text-xs text-gray-600 capitalize">
                    {user?.subscription === 'premium' && '✨ '}
                    {user?.subscription} 멤버
                  </p>
                </div>
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#8B6F47] to-[#6B5637] flex items-center justify-center text-white font-medium">
                  {user?.firstName?.[0]}{user?.lastName?.[0]}
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-12 gap-8">
          {/* 사이드바 */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <nav className="space-y-2">
                {[
                  { id: 'overview', label: '대시보드', icon: FiBarChart },
                  { id: 'profile', label: '프로필', icon: FiUser },
                  { id: 'settings', label: '설정', icon: FiSettings },
                  { id: 'billing', label: '구독 관리', icon: FiCreditCard }
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id as typeof activeTab)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                      activeTab === item.id
                        ? 'bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white'
                        : 'text-gray-600 hover:bg-gray-50'
                    }`}
                  >
                    <item.icon size={20} />
                    <span className="font-medium">{item.label}</span>
                  </button>
                ))}
              </nav>

              <hr className="my-6" />

              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-3 px-4 py-3 text-red-600 hover:bg-red-50 rounded-xl transition-colors"
              >
                <FiLogOut size={20} />
                <span className="font-medium">로그아웃</span>
              </button>
            </div>
          </div>

          {/* 메인 콘텐츠 */}
          <div className="lg:col-span-9">
            {/* 메시지 표시 */}
            <AnimatePresence>
              {message && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={`mb-6 p-4 rounded-xl flex items-center gap-3 ${
                    message.type === 'success'
                      ? 'bg-green-50 border border-green-200 text-green-700'
                      : 'bg-red-50 border border-red-200 text-red-700'
                  }`}
                >
                  {message.type === 'success' ? <FiCheck /> : <FiAlertCircle />}
                  <span>{message.text}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* 대시보드 개요 */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-3xl font-bold text-gray-900 mb-2">
                    안녕하세요, {user?.firstName}님! 👋
                  </h1>
                  <p className="text-gray-600">
                    오늘도 특별한 향기와 함께 멋진 하루 되세요
                  </p>
                </div>

                {/* 향수 통계 카드 */}
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <DashboardCard
                    title="내 향수 DNA"
                    value={`${stats.uniqueDNA} 가지`}
                    icon={FiDatabase}
                    change="+23 신규"
                    color="indigo"
                  />
                  <DashboardCard
                    title="향 복잡도"
                    value={`${stats.scentComplexity}/10`}
                    icon={FiTrendingUp}
                    change="+0.5"
                    color="emerald"
                  />
                  <DashboardCard
                    title="주 향료 계열"
                    value={stats.favoriteAccord}
                    icon={IoSparklesSharp}
                    color="amber"
                  />
                  <DashboardCard
                    title="희귀 원료"
                    value={`${stats.rareIngredients} 종`}
                    icon={FiAward}
                    change="+3"
                    color="rose"
                  />
                </div>

                {/* 향료 피라미드 */}
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <h2 className="text-xl font-bold text-gray-900 mb-6">나의 향료 피라미드 분석</h2>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className="w-20 h-20 mx-auto mb-3 rounded-full bg-gradient-to-br from-yellow-300 to-yellow-400 flex items-center justify-center">
                        <span className="text-2xl font-bold text-white">탑</span>
                      </div>
                      <h3 className="font-semibold text-gray-900">탑 노트</h3>
                      <p className="text-sm text-gray-600 mt-1">{stats.topNote}</p>
                      <p className="text-xs text-gray-500 mt-2">첫인상 • 15-30분</p>
                    </div>
                    <div className="text-center">
                      <div className="w-20 h-20 mx-auto mb-3 rounded-full bg-gradient-to-br from-pink-400 to-rose-500 flex items-center justify-center">
                        <span className="text-2xl font-bold text-white">하트</span>
                      </div>
                      <h3 className="font-semibold text-gray-900">하트 노트</h3>
                      <p className="text-sm text-gray-600 mt-1">{stats.heartNote}</p>
                      <p className="text-xs text-gray-500 mt-2">본향 • 2-4시간</p>
                    </div>
                    <div className="text-center">
                      <div className="w-20 h-20 mx-auto mb-3 rounded-full bg-gradient-to-br from-amber-600 to-amber-700 flex items-center justify-center">
                        <span className="text-2xl font-bold text-white">베이스</span>
                      </div>
                      <h3 className="font-semibold text-gray-900">베이스 노트</h3>
                      <p className="text-sm text-gray-600 mt-1">{stats.baseNote}</p>
                      <p className="text-xs text-gray-500 mt-2">잔향 • 4시간 이상</p>
                    </div>
                  </div>
                </div>

                {/* 최근 향수 */}
                <div className="bg-white rounded-2xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-gray-900">최근 제작한 향수</h2>
                    <Link
                      href="/fragrances"
                      className="text-[#8B6F47] hover:underline text-sm font-medium"
                    >
                      모두 보기 →
                    </Link>
                  </div>
                  <div className="space-y-3">
                    {fragranceHistory.map((fragrance) => (
                      <FragranceHistoryItem key={fragrance.id} fragrance={fragrance} />
                    ))}
                  </div>
                  <Link
                    href="/create"
                    className="mt-6 w-full flex items-center justify-center gap-2 py-3 border-2 border-dashed border-gray-300 rounded-xl text-gray-600 hover:border-[#8B6F47] hover:text-[#8B6F47] transition-colors"
                  >
                    <FiPlus />
                    <span>새 향수 만들기</span>
                  </Link>
                </div>
              </div>
            )}

            {/* 프로필 관리 */}
            {activeTab === 'profile' && (
              <div className="bg-white rounded-2xl shadow-lg p-8">
                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-2xl font-bold text-gray-900">프로필 정보</h2>
                  {!editMode ? (
                    <button
                      onClick={() => setEditMode(true)}
                      className="flex items-center gap-2 px-4 py-2 bg-[#8B6F47] text-white rounded-lg hover:bg-[#6B5637] transition-colors"
                    >
                      <FiEdit3 size={16} />
                      <span>수정</span>
                    </button>
                  ) : (
                    <div className="flex gap-2">
                      <button
                        onClick={() => setEditMode(false)}
                        className="px-4 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50 transition-colors"
                      >
                        취소
                      </button>
                      <button
                        onClick={handleProfileSave}
                        disabled={saving}
                        className="flex items-center gap-2 px-4 py-2 bg-[#8B6F47] text-white rounded-lg hover:bg-[#6B5637] transition-colors disabled:opacity-50"
                      >
                        <FiSave size={16} />
                        <span>{saving ? '저장 중...' : '저장'}</span>
                      </button>
                    </div>
                  )}
                </div>

                <div className="space-y-6">
                  {/* 기본 정보 */}
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        이름
                      </label>
                      <input
                        type="text"
                        value={profileForm.firstName}
                        onChange={(e) => setProfileForm({ ...profileForm, firstName: e.target.value })}
                        disabled={!editMode}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#8B6F47] focus:outline-none disabled:bg-gray-50 disabled:text-gray-600"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        성
                      </label>
                      <input
                        type="text"
                        value={profileForm.lastName}
                        onChange={(e) => setProfileForm({ ...profileForm, lastName: e.target.value })}
                        disabled={!editMode}
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#8B6F47] focus:outline-none disabled:bg-gray-50 disabled:text-gray-600"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      이메일
                    </label>
                    <input
                      type="email"
                      value={profileForm.email}
                      disabled
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg bg-gray-50 text-gray-600"
                    />
                    <p className="mt-1 text-xs text-gray-500">이메일은 변경할 수 없습니다</p>
                  </div>

                  {/* 환경 설정 */}
                  <div className="pt-6 border-t border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">환경 설정</h3>
                    <div className="grid md:grid-cols-3 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          언어
                        </label>
                        <select
                          value={profileForm.language}
                          onChange={(e) => setProfileForm({ ...profileForm, language: e.target.value })}
                          disabled={!editMode}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#8B6F47] focus:outline-none disabled:bg-gray-50"
                        >
                          <option value="ko">한국어</option>
                          <option value="en">English</option>
                          <option value="ja">日本語</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          통화
                        </label>
                        <select
                          value={profileForm.currency}
                          onChange={(e) => setProfileForm({ ...profileForm, currency: e.target.value })}
                          disabled={!editMode}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#8B6F47] focus:outline-none disabled:bg-gray-50"
                        >
                          <option value="KRW">KRW (₩)</option>
                          <option value="USD">USD ($)</option>
                          <option value="EUR">EUR (€)</option>
                          <option value="JPY">JPY (¥)</option>
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          테마
                        </label>
                        <select
                          value={profileForm.theme}
                          onChange={(e) => setProfileForm({ ...profileForm, theme: e.target.value as 'light' | 'dark' | 'auto' })}
                          disabled={!editMode}
                          className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#8B6F47] focus:outline-none disabled:bg-gray-50"
                        >
                          <option value="light">라이트</option>
                          <option value="dark">다크</option>
                          <option value="auto">자동</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* 알림 설정 */}
                  <div className="pt-6 border-t border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">알림 설정</h3>
                    <div className="space-y-3">
                      {Object.entries({
                        email: '이메일 알림',
                        push: '푸시 알림',
                        sms: 'SMS 알림',
                        marketing: '마케팅 정보 수신'
                      }).map(([key, label]) => (
                        <label key={key} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100">
                          <span className="text-gray-700">{label}</span>
                          <input
                            type="checkbox"
                            checked={profileForm.notifications[key as keyof typeof profileForm.notifications]}
                            onChange={(e) => setProfileForm({
                              ...profileForm,
                              notifications: {
                                ...profileForm.notifications,
                                [key]: e.target.checked
                              }
                            })}
                            disabled={!editMode}
                            className="w-5 h-5 text-[#8B6F47] border-gray-300 rounded focus:ring-[#8B6F47]"
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 보안 설정 */}
            {activeTab === 'settings' && (
              <div className="space-y-6">
                {/* 보안 설정 */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">보안 설정</h2>

                  <div className="space-y-6">
                    {/* 2FA 설정 */}
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-[#8B6F47]/10 rounded-lg">
                          <FiShield className="text-[#8B6F47]" size={24} />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">2단계 인증</h3>
                          <p className="text-sm text-gray-600">계정 보안을 강화합니다</p>
                        </div>
                      </div>
                      <button
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                          securitySettings.twoFactorEnabled
                            ? 'bg-green-100 text-green-700 hover:bg-green-200'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        {securitySettings.twoFactorEnabled ? '활성화됨' : '설정하기'}
                      </button>
                    </div>

                    {/* 생체 인증 */}
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-4">
                        <div className="p-3 bg-[#8B6F47]/10 rounded-lg">
                          <FiSmartphone className="text-[#8B6F47]" size={24} />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">생체 인증</h3>
                          <p className="text-sm text-gray-600">지문이나 Face ID로 로그인</p>
                        </div>
                      </div>
                      <button
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                          securitySettings.biometricEnabled
                            ? 'bg-green-100 text-green-700 hover:bg-green-200'
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        {securitySettings.biometricEnabled ? '활성화됨' : '설정하기'}
                      </button>
                    </div>

                    {/* 비밀번호 변경 */}
                    <div className="pt-6 border-t border-gray-200">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">비밀번호 변경</h3>
                      <button className="px-6 py-3 bg-[#8B6F47] text-white rounded-lg hover:bg-[#6B5637] transition-colors">
                        비밀번호 변경하기
                      </button>
                    </div>
                  </div>
                </div>

                {/* 연결된 기기 */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">연결된 기기</h2>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center gap-4">
                        <FiSmartphone className="text-gray-600" size={24} />
                        <div>
                          <p className="font-medium text-gray-900">Windows PC - Chrome</p>
                          <p className="text-sm text-gray-600">현재 기기 • 서울, 한국</p>
                        </div>
                      </div>
                      <span className="text-sm text-green-600 font-medium">활성</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 구독 관리 */}
            {activeTab === 'billing' && (
              <div className="space-y-6">
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">구독 관리</h2>

                  {/* 현재 플랜 */}
                  <div className="p-6 bg-gradient-to-br from-[#8B6F47]/10 to-[#6B5637]/10 rounded-xl mb-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {user?.subscription === 'premium' ? '프리미엄' :
                         user?.subscription === 'enterprise' ? '엔터프라이즈' : '무료'} 플랜
                      </h3>
                      {user?.subscription !== 'free' && (
                        <span className="px-3 py-1 bg-green-100 text-green-700 text-sm rounded-full font-medium">
                          활성
                        </span>
                      )}
                    </div>
                    <p className="text-gray-600 mb-4">
                      {user?.subscription === 'premium' ?
                        '모든 프리미엄 기능을 이용할 수 있습니다' :
                       user?.subscription === 'enterprise' ?
                        '기업용 고급 기능을 모두 이용할 수 있습니다' :
                        '기본 기능을 무료로 이용 중입니다'}
                    </p>
                    {user?.subscription === 'free' && (
                      <button className="px-6 py-3 bg-gradient-to-r from-[#8B6F47] to-[#6B5637] text-white rounded-lg hover:shadow-lg transition-all">
                        프리미엄으로 업그레이드
                      </button>
                    )}
                  </div>

                  {/* 플랜 비교 */}
                  <div className="grid md:grid-cols-3 gap-6">
                    {[
                      {
                        name: '무료',
                        price: '₩0',
                        features: [
                          '기본 향수 제작',
                          '월 3개 제작 제한',
                          '기본 DNA 분석',
                          '커뮤니티 접근'
                        ]
                      },
                      {
                        name: '프리미엄',
                        price: '₩29,900',
                        popular: true,
                        features: [
                          '무제한 향수 제작',
                          '고급 DNA 분석',
                          '3D 시각화',
                          '협업 기능',
                          '우선 지원'
                        ]
                      },
                      {
                        name: '엔터프라이즈',
                        price: '문의',
                        features: [
                          '프리미엄 모든 기능',
                          '팀 관리',
                          'API 접근',
                          '전담 매니저',
                          '맞춤 개발'
                        ]
                      }
                    ].map((plan) => (
                      <div
                        key={plan.name}
                        className={`p-6 border-2 rounded-xl ${
                          plan.popular ? 'border-[#8B6F47] relative' : 'border-gray-200'
                        }`}
                      >
                        {plan.popular && (
                          <span className="absolute -top-3 left-1/2 transform -translate-x-1/2 px-3 py-1 bg-[#8B6F47] text-white text-xs rounded-full">
                            인기
                          </span>
                        )}
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">{plan.name}</h3>
                        <p className="text-2xl font-bold text-gray-900 mb-4">
                          {plan.price}
                          {plan.price !== '문의' && <span className="text-sm text-gray-600">/월</span>}
                        </p>
                        <ul className="space-y-2 mb-6">
                          {plan.features.map((feature, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-sm text-gray-600">
                              <FiCheck className="text-green-500 mt-0.5 flex-shrink-0" />
                              <span>{feature}</span>
                            </li>
                          ))}
                        </ul>
                        <button
                          className={`w-full py-2 rounded-lg font-medium transition-colors ${
                            plan.popular
                              ? 'bg-[#8B6F47] text-white hover:bg-[#6B5637]'
                              : 'border border-gray-300 text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          {user?.subscription === plan.name.toLowerCase() ? '현재 플랜' :
                           plan.price === '문의' ? '문의하기' : '선택하기'}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}