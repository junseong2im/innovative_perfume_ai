'use client';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import clsx from 'clsx';

interface FilterSection {
  title: string;
  key: string;
  options: { value: string; label: string }[];
  type: 'checkbox' | 'radio' | 'range';
}

const filterSections: FilterSection[] = [
  {
    title: '향조',
    key: 'fragrance_family',
    type: 'checkbox',
    options: [
      { value: 'floral', label: '플로럴' },
      { value: 'citrus', label: '시트러스' },
      { value: 'woody', label: '우디' },
      { value: 'fruity', label: '프루티' },
      { value: 'spicy', label: '스파이시' },
      { value: 'oriental', label: '오리엔탈' },
      { value: 'fresh', label: '프레시' },
      { value: 'aromatic', label: '아로마틱' }
    ]
  },
  {
    title: '무드',
    key: 'mood',
    type: 'checkbox',
    options: [
      { value: 'daily', label: '데일리' },
      { value: 'romantic', label: '로맨틱' },
      { value: 'calm', label: '차분한' },
      { value: 'energetic', label: '활기찬' },
      { value: 'sensual', label: '관능적인' },
      { value: 'elegant', label: '우아한' },
      { value: 'modern', label: '모던한' },
      { value: 'classic', label: '클래식' }
    ]
  },
  {
    title: '계절',
    key: 'season',
    type: 'radio',
    options: [
      { value: 'all', label: '사계절' },
      { value: 'spring-summer', label: '봄/여름' },
      { value: 'fall-winter', label: '가을/겨울' }
    ]
  }
];

export function FragranceFilter() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [priceRange, setPriceRange] = useState<[number, number]>([0, 500000]);
  const [selectedFilters, setSelectedFilters] = useState<Record<string, string[]>>({});

  const handleFilterChange = (key: string, value: string, type: 'checkbox' | 'radio') => {
    if (type === 'checkbox') {
      setSelectedFilters((prev) => {
        const current = prev[key] || [];
        const updated = current.includes(value)
          ? current.filter((v) => v !== value)
          : [...current, value];

        if (updated.length === 0) {
          const { [key]: removed, ...rest } = prev;
          return rest;
        }

        return { ...prev, [key]: updated };
      });
    } else {
      setSelectedFilters((prev) => ({ ...prev, [key]: [value] }));
    }
  };

  const applyFilters = () => {
    const params = new URLSearchParams(searchParams.toString());

    // 필터 파라미터 설정
    Object.entries(selectedFilters).forEach(([key, values]) => {
      if (values.length > 0) {
        params.set(key, values.join(','));
      } else {
        params.delete(key);
      }
    });

    // 가격 범위 설정
    if (priceRange[0] > 0 || priceRange[1] < 500000) {
      params.set('price_min', priceRange[0].toString());
      params.set('price_max', priceRange[1].toString());
    } else {
      params.delete('price_min');
      params.delete('price_max');
    }

    router.push(`?${params.toString()}`);
  };

  const clearFilters = () => {
    setSelectedFilters({});
    setPriceRange([0, 500000]);
    router.push(window.location.pathname);
  };

  const formatPrice = (value: number) => {
    return new Intl.NumberFormat('ko-KR', {
      style: 'currency',
      currency: 'KRW'
    }).format(value);
  };

  return (
    <div className="w-full space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-[var(--luxury-midnight)] font-[var(--font-display)]">
          필터
        </h2>
        <button
          onClick={clearFilters}
          className="text-sm text-[var(--luxury-stone)] hover:text-[var(--luxury-midnight)] transition-colors"
        >
          초기화
        </button>
      </div>

      {/* 필터 섹션들 */}
      {filterSections.map((section) => (
        <div key={section.key} className="space-y-3">
          <h3 className="font-medium text-[var(--luxury-midnight)]">{section.title}</h3>
          <div className="space-y-2">
            {section.options.map((option) => {
              const isChecked = selectedFilters[section.key]?.includes(option.value) || false;

              return (
                <label
                  key={option.value}
                  className="flex items-center gap-3 cursor-pointer group"
                >
                  <input
                    type={section.type}
                    name={section.key}
                    value={option.value}
                    checked={isChecked}
                    onChange={() => handleFilterChange(section.key, option.value, section.type)}
                    className={clsx(
                      'peer',
                      section.type === 'checkbox'
                        ? 'w-4 h-4 rounded border-[var(--luxury-stone)] text-[var(--luxury-gold)] focus:ring-[var(--luxury-gold)]'
                        : 'w-4 h-4 text-[var(--luxury-gold)] focus:ring-[var(--luxury-gold)]'
                    )}
                  />
                  <span className="text-sm text-[var(--luxury-charcoal)] group-hover:text-[var(--luxury-midnight)] transition-colors">
                    {option.label}
                  </span>
                </label>
              );
            })}
          </div>
        </div>
      ))}

      {/* 가격 범위 필터 */}
      <div className="space-y-3">
        <h3 className="font-medium text-[var(--luxury-midnight)]">가격대</h3>
        <div className="space-y-4">
          <div className="px-3">
            <input
              type="range"
              min="0"
              max="500000"
              step="10000"
              value={priceRange[1]}
              onChange={(e) => setPriceRange([priceRange[0], parseInt(e.target.value)])}
              className="w-full h-2 bg-[var(--luxury-silk)] rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, var(--luxury-silk) 0%, var(--luxury-gold) ${(priceRange[1] / 500000) * 100}%, var(--luxury-silk) ${(priceRange[1] / 500000) * 100}%)`
              }}
            />
          </div>
          <div className="flex justify-between text-sm text-[var(--luxury-stone)]">
            <span>{formatPrice(priceRange[0])}</span>
            <span>{formatPrice(priceRange[1])}</span>
          </div>
        </div>
      </div>

      {/* 필터 적용 버튼 */}
      <button
        onClick={applyFilters}
        className="w-full py-3 px-4 bg-[var(--luxury-midnight)] text-[var(--luxury-cream)] rounded-lg hover:bg-[var(--luxury-charcoal)] transition-colors font-medium"
      >
        필터 적용
      </button>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          background: var(--luxury-gold);
          border-radius: 50%;
          cursor: pointer;
        }

        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: var(--luxury-gold);
          border-radius: 50%;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
}