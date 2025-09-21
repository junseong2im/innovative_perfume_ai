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

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/create"
              className="inline-flex items-center justify-center px-8 py-4 text-lg font-medium text-white rounded-lg transition-all transform hover:scale-105"
              style={{backgroundColor: 'var(--light-brown)'}}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown-dark)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown)'}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
              나의 향기 시작하기
            </Link>

            <Link
              href="/create?mode=chat"
              className="inline-flex items-center justify-center px-8 py-4 text-lg font-medium text-neutral-700 bg-white border-2 border-neutral-300 rounded-lg transition-all hover:border-neutral-400 transform hover:scale-105"
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
              이야기로 만들기
            </Link>
          </div>

        </div>
      </div>
    </section>
  );
}