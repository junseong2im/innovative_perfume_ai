'use client';

import Link from 'next/link';

export default function FragranceCreatorButton() {
  return (
    <section className="py-16 lg:py-20" style={{backgroundColor: 'var(--ivory-light)'}}>
      <div className="mx-auto max-w-4xl px-4 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-light text-neutral-900 mb-4 lg:text-4xl">
            당신의 언어가 향이 되는 과정
          </h2>
          <p className="text-lg text-neutral-600 mb-8 max-w-2xl mx-auto">
            당신의 기억, 감정, 꿈의 조각들을 들려주세요.
            들숨의 AI 아티스트가 세상에 단 하나뿐인 당신만의 향의 서사를 조율합니다.
          </p>

          <div className="flex justify-center">
            <Link
              href="/create"
              className="inline-flex items-center justify-center px-12 py-5 text-xl font-medium text-white rounded-lg transition-all transform hover:scale-105"
              style={{backgroundColor: 'var(--vintage-gold)'}}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--vintage-gold-dark)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--vintage-gold)'}
            >
              <svg
                className="w-6 h-6 mr-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
              시그니처 향수 만들기
            </Link>
          </div>

        </div>
      </div>
    </section>
  );
}