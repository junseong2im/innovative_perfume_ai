export default function CompanyIntro() {
  return (
    <section className="text-white py-16 lg:py-24" style={{backgroundColor: 'var(--light-brown)'}}>
      <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="text-center mb-16">
            <h2 className="mb-6 text-3xl font-light tracking-wide lg:text-4xl">
              AI 향수 예술
            </h2>
            <p className="text-xl font-light text-neutral-300">
              기술과 영원한 향수 제조 예술의 만남
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-6">
              <h3 className="text-2xl font-light text-white">우리의 철학</h3>
              <p className="text-lg text-neutral-300 leading-relaxed">
                제피루스 AI는 완벽한 향수는 개인적인 것이라고 믿습니다.
                당신의 기억, 감정, 고유한 정체성을 반영한 향수.
                우리의 쳊단 인공지능은 전통적인 향수 제조 예술을
                대체하는 것이 아니라 더욱 풍부하게 합니다.
              </p>
              <p className="text-lg text-neutral-300 leading-relaxed">
                당신이 설명하는 이상적인 향에 담긴 미묘한 언어를 분석하여,
                우리의 AI는 당신이 진정으로 원하는 향의 본질을 담은
                맞춤형 포뮤러를 만들어, 당신의 말을 후각의 시로 번역합니다.
              </p>
            </div>

            <div className="space-y-8">
              <div className="border-l-2 pl-6" style={{borderColor: 'var(--light-brown-dark)'}}>
                <h4 className="text-lg font-medium text-white mb-2">
                  쳊단 AI 기술
                </h4>
                <p className="text-neutral-400">
                  자체 개발한 신경망은 향 설명과 분자 구성 사이의
                  복잡한 관계를 이해합니다.
                </p>
              </div>

              <div className="border-l-2 pl-6" style={{borderColor: 'var(--light-brown-dark)'}}>
                <h4 className="text-lg font-medium text-white mb-2">
                  마스터 퍼퓸머 협업
                </h4>
                <p className="text-neutral-400">
                  모든 AI 생성 포뮤러는 숙련된 퍼퓸머와
                  향료 전문가 팀의 검증을 거쳐 완성됩니다.
                </p>
              </div>

              <div className="border-l-2 pl-6" style={{borderColor: 'var(--light-brown-dark)'}}>
                <h4 className="text-lg font-medium text-white mb-2">
                  지속 가능한 실천
                </h4>
                <p className="text-neutral-400">
                  최고급 천연 지속 가능 원료만을 사용하여
                  고급스럽고 책임감 있는 향수를 만들어냅니다.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-3xl font-light text-white mb-2">10,000+</div>
              <div className="text-neutral-400">제작된 고유 향수</div>
            </div>
            <div>
              <div className="text-3xl font-light text-white mb-2">98%</div>
              <div className="text-neutral-400">고객 만족도</div>
            </div>
            <div>
              <div className="text-3xl font-light text-white mb-2">50+</div>
              <div className="text-neutral-400">프리미엄 원료</div>
            </div>
          </div>

          <div className="mt-16 text-center">
            <h3 className="text-2xl font-light text-white mb-6">
              당신만의 시그니처 향을 만들 준비가 되셨나요?
            </h3>
            <p className="text-lg text-neutral-300 mb-8 max-w-2xl mx-auto">
              AI 기반 향수 제조의 마법을 통해 자신만의 완벽한 향을
              발견한 수천 명의 향수 애호가들과 함께하세요.
            </p>
            <button
              className="px-8 py-4 text-neutral-900 font-medium rounded-md transition-colors hover:opacity-90"
              style={{backgroundColor: 'var(--ivory-light)'}}
            >
              시작하기
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}