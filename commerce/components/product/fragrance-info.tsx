'use client';

import clsx from 'clsx';

interface FragranceInfoProps {
  longevity: number; // 1-5 scale
  sillage: number; // 1-5 scale
  season: string[];
  gender: 'masculine' | 'feminine' | 'unisex';
  className?: string;
}

export function FragranceInfo({
  longevity,
  sillage,
  season,
  gender,
  className
}: FragranceInfoProps) {
  const renderBar = (value: number, maxValue: number = 5) => {
    return (
      <div className="flex gap-1">
        {Array.from({ length: maxValue }).map((_, index) => (
          <div
            key={index}
            className={clsx(
              'h-2 w-8 rounded-full transition-all duration-300',
              index < value
                ? 'bg-[var(--luxury-gold)]'
                : 'bg-[var(--luxury-silk)]'
            )}
          />
        ))}
      </div>
    );
  };

  const getGenderIcon = (gender: string) => {
    switch (gender) {
      case 'masculine':
        return '♂';
      case 'feminine':
        return '♀';
      case 'unisex':
        return '⚥';
      default:
        return '⚥';
    }
  };

  const getSeasonIcon = (season: string) => {
    switch (season.toLowerCase()) {
      case '봄':
      case 'spring':
        return '🌸';
      case '여름':
      case 'summer':
        return '☀️';
      case '가을':
      case 'autumn':
      case 'fall':
        return '🍂';
      case '겨울':
      case 'winter':
        return '❄️';
      default:
        return '🌿';
    }
  };

  return (
    <div className={clsx('w-full', className)}>
      <h3 className="text-lg font-semibold mb-6 text-[var(--luxury-midnight)] font-[var(--font-display)]">
        향수 특성
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 지속력 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-2xl">⏱️</span>
            <span className="font-medium text-[var(--luxury-midnight)]">지속력</span>
          </div>
          <div className="flex items-center gap-4">
            {renderBar(longevity)}
            <span className="text-sm text-[var(--luxury-stone)]">
              {longevity === 1 && '1-2시간'}
              {longevity === 2 && '2-4시간'}
              {longevity === 3 && '4-6시간'}
              {longevity === 4 && '6-8시간'}
              {longevity === 5 && '8시간 이상'}
            </span>
          </div>
        </div>

        {/* 확산력 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-2xl">💨</span>
            <span className="font-medium text-[var(--luxury-midnight)]">확산력</span>
          </div>
          <div className="flex items-center gap-4">
            {renderBar(sillage)}
            <span className="text-sm text-[var(--luxury-stone)]">
              {sillage === 1 && '매우 가까움'}
              {sillage === 2 && '가까움'}
              {sillage === 3 && '보통'}
              {sillage === 4 && '강함'}
              {sillage === 5 && '매우 강함'}
            </span>
          </div>
        </div>

        {/* 추천 계절 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🗓️</span>
            <span className="font-medium text-[var(--luxury-midnight)]">추천 계절</span>
          </div>
          <div className="flex gap-3">
            {season.map((s, index) => (
              <div
                key={index}
                className="flex items-center gap-1 px-3 py-1.5 rounded-full bg-[var(--luxury-pearl)] border border-[var(--luxury-silk)]"
              >
                <span className="text-lg">{getSeasonIcon(s)}</span>
                <span className="text-sm text-[var(--luxury-charcoal)]">{s}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 성별 */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-2xl">👤</span>
            <span className="font-medium text-[var(--luxury-midnight)]">성별</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-3xl text-[var(--luxury-gold)]">
              {getGenderIcon(gender)}
            </span>
            <span className="text-[var(--luxury-charcoal)]">
              {gender === 'masculine' && '남성용'}
              {gender === 'feminine' && '여성용'}
              {gender === 'unisex' && '공용'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}