import { AddToCart } from 'components/cart/add-to-cart';
import Price from 'components/price';
import Prose from 'components/prose';
import { Product } from 'lib/shopify/types';
import { VariantSelector } from './variant-selector';
import { NotePyramid } from './note-pyramid';
import { FragranceInfo } from './fragrance-info';

export function ProductDescription({ product }: { product: Product }) {
  // 임시 향수 노트 데이터 (실제로는 product 데이터에서 가져와야 함)
  const sampleNotes = {
    topNotes: [
      { name: '베르가못' },
      { name: '레몬' },
      { name: '자몽' }
    ],
    heartNotes: [
      { name: '장미' },
      { name: '자스민' },
      { name: '일랑일랑' }
    ],
    baseNotes: [
      { name: '샌달우드' },
      { name: '앰버' },
      { name: '머스크' }
    ]
  };

  return (
    <>
      <div className="mb-6 flex flex-col border-b pb-6 dark:border-neutral-700">
        <h1 className="mb-2 text-5xl font-medium">{product.title}</h1>
        <div className="mr-auto w-auto rounded-full bg-[var(--luxury-midnight)] p-2 text-sm text-[var(--luxury-cream)]">
          <Price
            amount={product.priceRange.maxVariantPrice.amount}
            currencyCode={product.priceRange.maxVariantPrice.currencyCode}
          />
        </div>
      </div>

      {/* 향 피라미드 추가 */}
      <NotePyramid
        topNotes={sampleNotes.topNotes}
        heartNotes={sampleNotes.heartNotes}
        baseNotes={sampleNotes.baseNotes}
        className="mb-8"
      />

      {/* 향수 특성 정보 추가 */}
      <FragranceInfo
        longevity={4}
        sillage={3}
        season={['봄', '여름']}
        gender="unisex"
        className="mb-8"
      />

      <VariantSelector options={product.options} variants={product.variants} />

      {product.descriptionHtml ? (
        <Prose
          className="mb-6 text-sm leading-tight dark:text-white/[60%]"
          html={product.descriptionHtml}
        />
      ) : null}

      <AddToCart product={product} />
    </>
  );
}
