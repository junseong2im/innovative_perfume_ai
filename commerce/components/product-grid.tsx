import Image from 'next/image';
import Link from 'next/link';
import { Product } from 'lib/shopify/types';

interface ProductGridProps {
  products: Product[];
}

export default function ProductGrid({ products }: ProductGridProps) {
  return (
    <div className="grid grid-cols-1 gap-x-6 gap-y-16 sm:grid-cols-2 lg:grid-cols-3">
      {products.map((product) => (
        <Link
          key={product.id}
          href={`/product/${product.handle}`}
          className="group"
        >
          <div className="aspect-[3/4] w-full overflow-hidden bg-neutral-100 mb-6">
            {product.featuredImage ? (
              <Image
                src={product.featuredImage.url}
                alt={product.featuredImage.altText || product.title}
                width={400}
                height={533}
                className="h-full w-full object-cover object-center group-hover:opacity-90 transition-opacity"
              />
            ) : (
              <div className="h-full w-full bg-neutral-200 flex items-center justify-center">
                {/* Bottle SVG placeholder */}
                <svg
                  className="w-24 h-32 text-neutral-400"
                  fill="currentColor"
                  viewBox="0 0 100 150"
                >
                  <rect x="35" y="20" width="30" height="120" rx="4" />
                  <rect x="30" y="10" width="40" height="15" rx="2" />
                  <rect x="32" y="5" width="36" height="10" rx="3" />
                  <rect x="40" y="50" width="20" height="30" rx="1" fill="white" opacity="0.7" />
                </svg>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <h3 className="text-lg font-light text-neutral-900 group-hover:text-neutral-600 transition-colors">
              {product.title}
            </h3>

            <p className="text-sm text-neutral-600 line-clamp-2">
              {product.description}
            </p>

            <p className="text-sm font-medium text-neutral-900">
              {new Intl.NumberFormat('ko-KR').format(
                parseInt(product.priceRange.minVariantPrice.amount)
              )}{' '}
              {product.priceRange.minVariantPrice.currencyCode}
            </p>

            {/* Notes display */}
            {product.tags && product.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {product.tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 text-xs bg-neutral-100 text-neutral-600 rounded"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </Link>
      ))}
    </div>
  );
}