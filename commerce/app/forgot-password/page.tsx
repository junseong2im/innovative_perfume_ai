'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // 백엔드 API 호출
      const response = await fetch('/api/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });

      if (!response.ok) {
        throw new Error('Failed to send reset email');
      }

      setIsSubmitted(true);
    } catch (err) {
      setError('비밀번호 재설정 요청 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'var(--ivory-light)' }}>
      <div className="w-full max-w-md">
        <div className="bg-white rounded-xl shadow-xl p-8 border border-opacity-20" style={{ borderColor: 'var(--vintage-gold)' }}>
          {!isSubmitted ? (
            <>
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
                <h2 className="text-xl font-light mt-4 mb-2" style={{ color: 'var(--vintage-navy)' }}>
                  비밀번호 찾기
                </h2>
                <p className="text-sm" style={{ color: 'var(--vintage-gray)' }}>
                  가입하신 이메일 주소를 입력해 주세요.<br />
                  비밀번호 재설정 링크를 보내드립니다.
                </p>
              </div>

              {error && (
                <div className="mb-6 p-4 rounded-lg" style={{ backgroundColor: 'var(--vintage-rose)', color: 'var(--deep-brown)' }}>
                  <p className="text-sm font-medium">{error}</p>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2" style={{ color: 'var(--vintage-navy)' }}>
                    이메일 주소
                  </label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full px-4 py-3 border rounded-lg focus:outline-none transition-all"
                    style={{
                      borderColor: 'var(--vintage-gray-light)',
                      backgroundColor: 'var(--ivory-light)',
                      color: 'var(--vintage-navy)'
                    }}
                    placeholder="example@email.com"
                  />
                </div>

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
                      전송 중...
                    </div>
                  ) : (
                    '비밀번호 재설정 링크 전송'
                  )}
                </button>
              </form>

              <div className="mt-8 text-center">
                <Link
                  href="/login"
                  className="text-sm transition-colors"
                  style={{ color: 'var(--vintage-gray)' }}
                  onMouseEnter={(e) => e.target.style.color = 'var(--vintage-navy)'}
                  onMouseLeave={(e) => e.target.style.color = 'var(--vintage-gray)'}
                >
                  ← 로그인으로 돌아가기
                </Link>
              </div>
            </>
          ) : (
            <div className="text-center">
              <div className="mb-6">
                <div
                  className="w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-4"
                  style={{ backgroundColor: 'var(--vintage-sage)' }}
                >
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>

                <h2 className="text-xl font-light mb-2" style={{ color: 'var(--vintage-navy)' }}>
                  이메일이 전송되었습니다
                </h2>
                <p className="text-sm mb-4" style={{ color: 'var(--vintage-gray)' }}>
                  <strong>{email}</strong>로<br />
                  비밀번호 재설정 링크를 보내드렸습니다.
                </p>
                <p className="text-xs" style={{ color: 'var(--vintage-gray)' }}>
                  이메일이 도착하지 않았다면<br />
                  스팸함을 확인해 주세요.
                </p>
              </div>

              <div className="space-y-3">
                <button
                  onClick={() => {
                    setIsSubmitted(false);
                    setEmail('');
                  }}
                  className="w-full py-3 px-6 rounded-lg font-medium transition-all border"
                  style={{
                    borderColor: 'var(--vintage-gray-light)',
                    color: 'var(--vintage-navy)',
                    backgroundColor: 'white'
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--ivory-light)'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
                >
                  다른 이메일로 재전송
                </button>

                <Link
                  href="/login"
                  className="block w-full py-3 px-6 text-white font-medium rounded-lg transition-all text-center"
                  style={{ backgroundColor: 'var(--vintage-gold)' }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--vintage-gold-dark)'}
                  onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--vintage-gold)'}
                >
                  로그인으로 돌아가기
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}