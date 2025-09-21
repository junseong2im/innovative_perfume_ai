import { getProducts } from 'lib/shopify';
import ProductGrid from 'components/product-grid';
import Link from 'next/link';

export const metadata = {
  title: '신제품 & 추천 | Deulsoom',
  description: '최신 출시 향수와 추천 제품을 만나보세요'
};

export default async function NewProductsPage() {
  const products = await getProducts({ sortKey: 'CREATED_AT', reverse: true });
  const recommendedProducts = products.slice(0, 3);
  const newProducts = products.slice(3, 9);

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
            {recommendedProducts.map((product: any) => (
              <div key={product.handle} className="relative group">
                <div className="absolute top-4 left-4 z-10">
                  <span className="bg-red-600 text-white px-3 py-1 text-xs font-medium rounded">
                    BEST
                  </span>
                </div>
                <Link href={`/product/${product.handle}`}>
                  <div className="aspect-square bg-neutral-100 rounded-lg overflow-hidden">
                    <img
                      src={product.featuredImage?.url || '/placeholder.jpg'}
                      alt={product.title}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  </div>
                  <div className="mt-4 text-center">
                    <h3 className="font-medium text-neutral-900">{product.title}</h3>
                    <p className="mt-1 text-neutral-600">
                      {product.priceRange.minVariantPrice.amount} {product.priceRange.minVariantPrice.currencyCode}
                    </p>
                  </div>
                </Link>
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

          <ProductGrid products={newProducts} />
        </div>
      </section>
    </div>
  );
}