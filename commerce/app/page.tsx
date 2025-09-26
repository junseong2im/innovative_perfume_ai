import { Suspense } from 'react';
import { getProducts } from 'lib/shopify';
import Footer from 'components/layout/footer';
import FragranceCreatorButton from 'components/fragrance-creator-button';
import ProductGrid from 'components/product-grid';
import CompanyIntro from 'components/company-intro';
import HeroSection from 'components/hero-section';

export const metadata = {
  title: 'Deulsoom - AI Fragrance Collection',
  description: 'Create your personalized fragrance with AI. Discover distinctive, memorable fragrances crafted with sophisticated sensibility. Experience the essence of a deep breath.',
  openGraph: {
    type: 'website',
    title: 'Deulsoom - AI Fragrance Collection',
    description: 'Create your personalized fragrance with AI, inspired by the art of breathing'
  }
};

export default async function HomePage() {
  // Get products for the showcase section
  const products = await getProducts({ sortKey: 'CREATED_AT' });

  return (
    <>
      {/* Hero Section with Navigation */}
      <HeroSection />

      {/* Fragrance Creator Button Section */}
      <FragranceCreatorButton />

      {/* Featured Products Section */}
      <section className="py-16 lg:py-24 bg-[var(--luxury-cream)] border-t border-[var(--luxury-silk)]">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
          <div className="mb-16 text-center">
            <h2 className="mb-6 text-3xl font-light tracking-wide text-[var(--luxury-midnight)] lg:text-4xl font-[var(--font-display)]">
              시그니처 컬렉션
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-[var(--luxury-charcoal)] leading-relaxed">
              후각의 우수성에 세심한 주의를 기울여 제작된
              독특한 향수들을 만나보세요.
            </p>
          </div>
          <Suspense fallback={<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-96 bg-neutral-100 animate-pulse" />
            ))}
          </div>}>
            <ProductGrid products={products} />
          </Suspense>
        </div>
      </section>

      {/* Company Introduction */}
      <CompanyIntro />

      <Footer />
    </>
  );
}