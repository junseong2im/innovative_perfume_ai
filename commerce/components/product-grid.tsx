'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Product } from 'lib/shopify/types';

interface ProductGridProps {
  products: Product[];
}

// Mock data for demonstration
const mockProducts = [
  {
    id: '1',
    handle: 'midnight-memory',
    title: 'Midnight Memory',
    description: '깊은 밤의 기억, 우디하고 신비로운 향',
    keywords: ['우디', '머스크', '앰버', '은은한'],
    notes: {
      top: '베르가못, 라벤더',
      middle: '시더우드, 샌달우드',
      base: '앰버, 머스크'
    },
    imageUrl: '/images/product-1.jpg',
    price: '280,000'
  },
  {
    id: '2',
    handle: 'dawn-breath',
    title: 'Dawn Breath',
    description: '새벽공기의 청량함, 상쾌한 시트러스',
    keywords: ['시트러스', '프레시', '민트', '활력'],
    notes: {
      top: '레몬, 자몽',
      middle: '민트, 바질',
      base: '화이트 머스크'
    },
    imageUrl: '/images/product-2.jpg',
    price: '260,000'
  },
  {
    id: '3',
    handle: 'library-whisper',
    title: 'Library Whisper',
    description: '도서관의 고요함, 종이와 가죽의 향',
    keywords: ['레더', '바닐라', '도서관', '지적인'],
    notes: {
      top: '페이퍼, 바이올렛',
      middle: '레더, 아이리스',
      base: '바닐라, 통카빈'
    },
    imageUrl: '/images/product-3.jpg',
    price: '320,000'
  },
  {
    id: '4',
    handle: 'forest-dream',
    title: 'Forest Dream',
    description: '숲속의 꿈, 초록빛 자연의 향',
    keywords: ['그린', '이끼', '허브', '자연'],
    notes: {
      top: '그린티, 버베나',
      middle: '오크모스, 제라늄',
      base: '베티버, 파출리'
    },
    imageUrl: '/images/product-4.jpg',
    price: '275,000'
  },
  {
    id: '5',
    handle: 'silk-touch',
    title: 'Silk Touch',
    description: '실크의 감촉, 부드럽고 파우더리한',
    keywords: ['파우더리', '플로럴', '부드러운', '우아한'],
    notes: {
      top: '화이트 로즈, 프리지아',
      middle: '파우더, 헬리오트로프',
      base: '화이트 머스크, 캐시미어 우드'
    },
    imageUrl: '/images/product-5.jpg',
    price: '295,000'
  },
  {
    id: '6',
    handle: 'golden-hour',
    title: 'Golden Hour',
    description: '황금빛 시간, 따뜻하고 감미로운',
    keywords: ['앰버', '꿀', '따뜻한', '감미로운'],
    notes: {
      top: '허니, 사프란',
      middle: '앰버그리스, 재스민',
      base: '벤조인, 라브다넘'
    },
    imageUrl: '/images/product-6.jpg',
    price: '340,000'
  }
];

export default function ProductGrid({ products }: ProductGridProps) {
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
      {mockProducts.map((product) => (
        <div
          key={product.id}
          className="group relative"
          onMouseEnter={() => setHoveredCard(product.id)}
          onMouseLeave={() => setHoveredCard(null)}
        >
          <Link href={`/product/${product.handle}`} className="block">
            {/* Card Container */}
            <div className="relative bg-[var(--luxury-pearl)] overflow-hidden transition-all duration-500 group-hover:transform group-hover:-translate-y-2 group-hover:shadow-2xl">
              {/* Image Container */}
              <div className="aspect-[3/4] relative overflow-hidden bg-gradient-to-br from-[var(--luxury-silk)] to-[var(--luxury-pearl)]">
                {/* Placeholder for product image */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <svg
                    className="w-24 h-32 text-[var(--luxury-stone)] opacity-20 transition-transform duration-500 group-hover:scale-110"
                    fill="currentColor"
                    viewBox="0 0 100 150"
                  >
                    <rect x="35" y="20" width="30" height="120" rx="4" />
                    <rect x="30" y="10" width="40" height="15" rx="2" />
                    <rect x="32" y="5" width="36" height="10" rx="3" />
                    <rect x="40" y="50" width="20" height="30" rx="1" fill="white" opacity="0.3" />
                  </svg>
                </div>

                {/* Floating Keywords - Appear on Hover */}
                <div className={`absolute inset-0 flex flex-wrap items-center justify-center gap-2 p-4 transition-all duration-500 ${
                  hoveredCard === product.id ? 'opacity-100' : 'opacity-0'
                }`}>
                  {product.keywords.map((keyword, index) => (
                    <span
                      key={keyword}
                      className="px-3 py-1 text-xs tracking-wider text-[var(--luxury-midnight)] bg-white/80 backdrop-blur-sm rounded-full transform transition-all duration-300"
                      style={{
                        animationDelay: `${index * 100}ms`,
                        animation: hoveredCard === product.id ? 'fadeInUp 0.5s ease-out forwards' : 'none'
                      }}
                    >
                      #{keyword}
                    </span>
                  ))}
                </div>

                {/* Gradient Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-[var(--luxury-midnight)]/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              </div>

              {/* Content */}
              <div className="p-6">
                {/* Title */}
                <h3 className="text-lg font-light tracking-wider text-[var(--luxury-midnight)] mb-2 transition-colors duration-300 group-hover:text-[var(--luxury-gold)]">
                  {product.title}
                </h3>

                {/* Description */}
                <p className="text-sm text-[var(--luxury-charcoal)] mb-4 line-clamp-2">
                  {product.description}
                </p>

                {/* Notes Preview - Shows on Hover */}
                <div className={`overflow-hidden transition-all duration-500 ${
                  hoveredCard === product.id ? 'max-h-24 opacity-100' : 'max-h-0 opacity-0'
                }`}>
                  <div className="text-xs space-y-1 text-[var(--luxury-stone)]">
                    <div>Top: {product.notes.top}</div>
                    <div>Middle: {product.notes.middle}</div>
                    <div>Base: {product.notes.base}</div>
                  </div>
                </div>

                {/* Price */}
                <div className="flex items-center justify-between mt-4">
                  <span className="text-lg font-light text-[var(--luxury-midnight)]">
                    ₩{product.price}
                  </span>
                  <span className="text-xs text-[var(--luxury-gold)] opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    자세히 보기 →
                  </span>
                </div>
              </div>

              {/* Hover Border Effect */}
              <div className="absolute inset-0 border border-[var(--luxury-gold)] opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
            </div>
          </Link>
        </div>
      ))}
    </div>
  );
}