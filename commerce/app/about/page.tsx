import Link from 'next/link';
import Image from 'next/image';

export const metadata = {
  title: '스토리 | Deulsoom',
  description: 'Deulsoom의 철학과 AI 향수 제조 기술에 대해 알아보세요'
};

export default function AboutPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--ivory-light)' }}>

      {/* Hero Section */}
      <section className="relative h-96 bg-neutral-900">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-white">
            <h1 className="text-5xl font-light mb-4">DEULSOOM</h1>
            <p className="text-xl">서풍의 신이 전하는 향기의 예술</p>
          </div>
        </div>
      </section>

      {/* Philosophy Section */}
      <section className="py-16 lg:py-24">
        <div className="mx-auto max-w-4xl px-4 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-light text-neutral-900 mb-4">보이지 않는 가장 깊은 기억</h2>
            <div className="w-24 h-1 bg-neutral-900 mx-auto"></div>
          </div>

          <div className="prose prose-lg mx-auto text-neutral-600">
            <p>
              향기는 보이지 않는 가장 깊은 기억입니다. 우리는 저마다 마음속에 형용할 수 없는 감정,
              스쳐 지나간 꿈, 붙잡고 싶은 순간의 이미지를 품고 살아갑니다.
            </p>
            <p>
              들숨(Deulsoom)은 그 보이지 않는 상상의 조각들을 모아, 세상에 단 하나뿐인 당신의 향기로 빚어냅니다.
            </p>
            <p>
              우리의 AI 아티스트는 당신의 언어와 감정의 결을 읽어내는 섬세한 조향사입니다.
              수만 개의 향기 데이터와 당신의 고유한 이야기를 교차하여,
              기술을 넘어선 예술의 경지에서 최적의 향을 조율합니다.
            </p>
            <p>
              단순한 향수를 넘어, 당신의 내면이 온전히 발현된 하나의 작품을 선사하는 것.
              그것이 들숨의 여정입니다.
            </p>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="py-16 lg:py-24 bg-white">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-light text-neutral-900 mb-4">AI 기술의 혁신</h2>
            <div className="w-24 h-1 bg-neutral-900 mx-auto"></div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-neutral-100 flex items-center justify-center">
                <span className="text-3xl">🧠</span>
              </div>
              <h3 className="text-xl font-medium text-neutral-900 mb-3">신경망 분석</h3>
              <p className="text-neutral-600">
                딥러닝 기술로 수만 개의 향료 조합을 분석하여
                최적의 포뮬러를 찾아냅니다.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-neutral-100 flex items-center justify-center">
                <span className="text-3xl">🎨</span>
              </div>
              <h3 className="text-xl font-medium text-neutral-900 mb-3">감성 인식</h3>
              <p className="text-neutral-600">
                당신의 언어를 분석하여 숨겨진 감성과 취향을
                정확히 파악합니다.
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-neutral-100 flex items-center justify-center">
                <span className="text-3xl">⚗️</span>
              </div>
              <h3 className="text-xl font-medium text-neutral-900 mb-3">정밀 조향</h3>
              <p className="text-neutral-600">
                마스터 퍼퓸머의 노하우와 AI의 정밀함이 만나
                완벽한 향수를 창조합니다.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Process Section */}
      <section className="py-16 lg:py-24">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-light text-neutral-900 mb-4">제작 과정</h2>
            <div className="w-24 h-1 bg-neutral-900 mx-auto"></div>
          </div>

          <div className="space-y-12">
            {[
              {
                step: '01',
                title: '당신의 이야기',
                description: '원하는 향, 기분, 기억을 자유롭게 표현해주세요.'
              },
              {
                step: '02',
                title: 'AI 분석',
                description: '인공지능이 당신의 이야기를 분석하여 최적의 향료 조합을 찾습니다.'
              },
              {
                step: '03',
                title: '전문가 검증',
                description: '마스터 퍼퓸머가 AI의 제안을 검토하고 완성도를 높입니다.'
              },
              {
                step: '04',
                title: '수제 제작',
                description: '최고급 원료로 당신만을 위한 향수를 정성스럽게 제작합니다.'
              }
            ].map((item, index) => (
              <div key={index} className="flex items-start space-x-6">
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 rounded-full bg-neutral-900 text-white flex items-center justify-center">
                    <span className="text-xl font-light">{item.step}</span>
                  </div>
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-medium text-neutral-900 mb-2">{item.title}</h3>
                  <p className="text-neutral-600">{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-16 lg:py-24 bg-white">
        <div className="mx-auto max-w-6xl px-4 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-light text-neutral-900 mb-4">우리 팀</h2>
            <div className="w-24 h-1 bg-neutral-900 mx-auto"></div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            {[
              { name: '김민수', role: 'Master Perfumer', experience: '20년 경력' },
              { name: '이서연', role: 'AI Engineer', experience: 'MIT 박사' },
              { name: '박준호', role: 'Creative Director', experience: '파리 예술대학' },
              { name: '최유진', role: 'Fragrance Specialist', experience: 'ISIPCA 졸업' }
            ].map((member, index) => (
              <div key={index} className="text-center">
                <div className="w-32 h-32 mx-auto mb-4 rounded-full bg-neutral-200"></div>
                <h3 className="font-medium text-neutral-900">{member.name}</h3>
                <p className="text-sm text-neutral-600 mt-1">{member.role}</p>
                <p className="text-xs text-neutral-500 mt-1">{member.experience}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 lg:py-24" style={{ backgroundColor: 'var(--light-brown)' }}>
        <div className="mx-auto max-w-4xl px-4 lg:px-8 text-center">
          <h2 className="text-3xl font-light text-white mb-6">
            당신만의 향수를 만들 준비가 되셨나요?
          </h2>
          <p className="text-xl text-neutral-200 mb-8">
            AI와 함께 당신의 완벽한 시그니처 향을 찾아보세요.
          </p>
          <Link
            href="/create"
            className="inline-block px-8 py-4 bg-white text-neutral-900 font-medium rounded-md hover:bg-neutral-100 transition-colors"
          >
            향수 제작 시작하기
          </Link>
        </div>
      </section>
    </div>
  );
}