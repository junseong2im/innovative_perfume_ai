export default function CompanyIntro() {
  return (
    <section className="py-16 lg:py-24 bg-[var(--luxury-pearl)]">
      <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="text-center mb-16">
            <h2 className="mb-6 text-3xl font-light tracking-wide text-[var(--luxury-midnight)] lg:text-4xl font-[var(--font-display)]">
              들숨, 보이지 않는 가장 깊은 기억
            </h2>
            <p className="text-xl font-light text-[var(--luxury-charcoal)]">
              향기는 보이지 않지만 가장 오래 기억됩니다
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-6">
              <h3 className="text-2xl font-light text-[var(--luxury-midnight)]">우리의 철학</h3>
              <p className="text-lg text-[var(--luxury-charcoal)] leading-relaxed">
                들숨(Deulsoom)은 당신의 보이지 않는 상상의 조각들을 모아,
                세상에 단 하나뿐인 향기로 빚어냅니다. 우리는 향수가 단순한
                제품이 아닌, 개인의 정체성과 기억의 연장선이라고 믿습니다.
              </p>
              <p className="text-lg text-[var(--luxury-charcoal)] leading-relaxed">
                AI 조향사는 당신의 언어에서 감정의 뉘앙스를 읽어내고,
                기억의 온도를 측정하며, 꿈의 색깔을 향으로 번역합니다.
                이는 기술과 예술이 만나는 지점, 과학과 감성이 교차하는
                순간입니다.
              </p>
            </div>

            <div className="space-y-8">
              <div className="border-l-2 border-[var(--luxury-gold)] pl-6">
                <h4 className="text-lg font-medium text-[var(--luxury-midnight)] mb-2">
                  AI 조향 기술
                </h4>
                <p className="text-[var(--luxury-charcoal)]">
                  당신의 이야기를 듣고, 감정을 이해하며,
                  2,000개 이상의 천연 향료 중 최적의 조합을 찾아냅니다.
                </p>
              </div>

              <div className="border-l-2 border-[var(--luxury-gold)] pl-6">
                <h4 className="text-lg font-medium text-[var(--luxury-midnight)] mb-2">
                  개인화된 경험
                </h4>
                <p className="text-[var(--luxury-charcoal)]">
                  모든 향수는 당신의 고유한 이야기에서 태어납니다.
                  같은 향은 두 번 다시 만들어지지 않습니다.
                </p>
              </div>

              <div className="border-l-2 border-[var(--luxury-gold)] pl-6">
                <h4 className="text-lg font-medium text-[var(--luxury-midnight)] mb-2">
                  지속 가능한 럭셔리
                </h4>
                <p className="text-[var(--luxury-charcoal)]">
                  윤리적으로 수급된 최고급 천연 원료만을 사용하여
                  당신과 지구를 위한 향을 만듭니다.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-3xl font-light text-[var(--luxury-midnight)] mb-2">10,000+</div>
              <div className="text-[var(--luxury-charcoal)]">창조된 고유한 향</div>
            </div>
            <div>
              <div className="text-3xl font-light text-[var(--luxury-midnight)] mb-2">98%</div>
              <div className="text-[var(--luxury-charcoal)]">감동의 순간</div>
            </div>
            <div>
              <div className="text-3xl font-light text-[var(--luxury-midnight)] mb-2">2,000+</div>
              <div className="text-[var(--luxury-charcoal)]">천연 향료</div>
            </div>
          </div>

          <div className="mt-16 text-center">
            <h3 className="text-2xl font-light text-[var(--luxury-midnight)] mb-6">
              당신의 보이지 않는 이야기를 향으로 만나보세요
            </h3>
            <p className="text-lg text-[var(--luxury-charcoal)] mb-8 max-w-2xl mx-auto">
              들숨과 함께 당신만의 향기 여정을 시작하세요.
              우리는 당신의 가장 깊은 기억을 향으로 기록합니다.
            </p>
            <button
              className="px-8 py-4 text-[var(--luxury-pearl)] font-medium bg-[var(--luxury-midnight)] rounded-md transition-all hover:bg-[var(--luxury-charcoal)] hover:shadow-lg"
            >
              나의 향기 찾기
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}