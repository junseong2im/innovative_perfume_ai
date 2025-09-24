import ProductGrid from 'components/product-grid';
import Link from 'next/link';

export const metadata = {
  title: '신제품 & 추천 | Deulsoom',
  description: '최신 출시 향수와 추천 제품을 만나보세요'
};

export default async function NewProductsPage() {
  // 추천 제품 더미 데이터 (상품 준비중)
  const recommendedProducts = [
    {
      id: '1',
      handle: 'deulsoom-signature-1',
      title: '들숨 시그니처 No.1',
      price: '165,000',
      description: '곧 출시될 들숨의 첫 번째 시그니처 향수입니다.'
    },
    {
      id: '2',
      handle: 'deulsoom-signature-2',
      title: '들숨 시그니처 No.2',
      price: '185,000',
      description: '세련된 우디 계열의 깊이 있는 향수입니다.'
    },
    {
      id: '3',
      handle: 'deulsoom-limited',
      title: '들숨 리미티드 에디션',
      price: '220,000',
      description: '한정판으로 출시될 특별한 향수입니다.'
    }
  ];

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--ivory-light)' }}>

      {/* Page Title */}
      <div className="bg-white border-b border-neutral-200 py-8">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8 text-center">
          <h1 className="text-3xl font-light text-neutral-900">신제품 & 추천</h1>
        </div>
      </div>

      {/* Recommended Section */}
      <section className="py-12 lg:py-16">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-light text-neutral-900">
              이달의 추천 향수
            </h2>
            <p className="text-lg text-neutral-600">
              전문가가 엄선한 시즌 베스트 컬렉션
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
            {recommendedProducts.map((product) => (
              <div key={product.handle} className="relative group cursor-not-allowed">
                <div className="absolute top-4 left-4 z-10">
                  <span className="bg-amber-600 text-white px-3 py-1 text-xs font-medium rounded">
                    준비중
                  </span>
                </div>
                <div className="aspect-square bg-gradient-to-b from-neutral-50 to-neutral-100 rounded-lg overflow-hidden relative">
                  {/* 향수 병 SVG */}
                  <div className="w-full h-full flex items-center justify-center">
                    <svg
                      className="w-20 h-28 text-neutral-300"
                      fill="currentColor"
                      viewBox="0 0 100 150"
                    >
                      <rect x="35" y="20" width="30" height="120" rx="4" />
                      <rect x="30" y="10" width="40" height="15" rx="2" />
                      <rect x="32" y="5" width="36" height="10" rx="3" />
                      <rect x="40" y="50" width="20" height="30" rx="1" fill="white" opacity="0.7" />
                    </svg>
                  </div>

                  {/* 상품 준비중 오버레이 */}
                  <div className="absolute inset-0 bg-white bg-opacity-80 flex items-center justify-center">
                    <div className="text-center">
                      <div className="text-sm font-medium text-neutral-800 mb-1">
                        상품 준비중
                      </div>
                      <div className="text-xs text-neutral-600">
                        Coming Soon
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 text-center opacity-75">
                  <h3 className="font-medium text-neutral-900">{product.title}</h3>
                  <p className="mt-1 text-neutral-600">
                    {product.price} KRW
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* New Products Section */}
      <section className="py-12 lg:py-16 bg-white">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
          <div className="mb-12 text-center">
            <h2 className="mb-4 text-3xl font-light text-neutral-900">
              신제품
            </h2>
            <p className="text-lg text-neutral-600">
              방금 도착한 최신 향수 컬렉션
            </p>
          </div>

          <ProductGrid products={[]} />
        </div>
      </section>

      {/* 사전 예약 안내 섹션 */}
      <section className="py-16 bg-neutral-50">
        <div className="mx-auto max-w-screen-xl px-4 lg:px-8 text-center">
          <h3 className="text-2xl font-light text-neutral-900 mb-4">
            사전 예약 안내
          </h3>
          <p className="text-lg text-neutral-600 mb-8 max-w-2xl mx-auto">
            들숨의 첫 번째 컬렉션이 곧 출시됩니다. <br />
            사전 예약을 통해 특별한 혜택을 받아보세요.
          </p>
          <div className="space-y-4">
            <div className="inline-flex items-center space-x-2 text-sm text-neutral-700 bg-white px-4 py-2 rounded-full">
              <span>📧</span>
              <span>출시 알림 신청</span>
            </div>
            <div className="inline-flex items-center space-x-2 text-sm text-neutral-700 bg-white px-4 py-2 rounded-full ml-4">
              <span>🎁</span>
              <span>사전 예약 특가</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}