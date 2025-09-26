'use client';

import { useState } from 'react';
import clsx from 'clsx';

interface Note {
  name: string;
  description?: string;
  icon?: string;
}

interface NotePyramidProps {
  topNotes: Note[];
  heartNotes: Note[];
  baseNotes: Note[];
  className?: string;
}

export function NotePyramid({
  topNotes,
  heartNotes,
  baseNotes,
  className
}: NotePyramidProps) {
  const [activeLayer, setActiveLayer] = useState<'top' | 'heart' | 'base' | null>(null);

  const renderNotes = (notes: Note[], label: string) => {
    return notes.map((note, index) => (
      <span key={index} className="text-sm">
        {note.name}{index < notes.length - 1 ? ', ' : ''}
      </span>
    ));
  };

  return (
    <div className={clsx('w-full', className)}>
      <h3 className="text-lg font-semibold mb-6 text-[var(--luxury-midnight)] font-[var(--font-display)]">
        향의 구성
      </h3>

      <div className="relative">
        {/* 피라미드 시각화 */}
        <div className="flex flex-col items-center gap-1">
          {/* Top Notes - 삼각형 상단 */}
          <div
            className={clsx(
              'relative w-0 h-0 cursor-pointer transition-all duration-500',
              'border-l-[60px] border-l-transparent',
              'border-r-[60px] border-r-transparent',
              'border-b-[40px]',
              activeLayer === 'top'
                ? 'border-b-[var(--luxury-gold)]'
                : 'border-b-[var(--luxury-silk)] hover:border-b-[var(--luxury-rose-gold)]'
            )}
            onMouseEnter={() => setActiveLayer('top')}
            onMouseLeave={() => setActiveLayer(null)}
          />

          {/* Heart Notes - 사각형 중간 */}
          <div
            className={clsx(
              'w-[160px] h-[50px] cursor-pointer transition-all duration-500',
              activeLayer === 'heart'
                ? 'bg-[var(--luxury-gold)]'
                : 'bg-[var(--luxury-silk)] hover:bg-[var(--luxury-rose-gold)]'
            )}
            onMouseEnter={() => setActiveLayer('heart')}
            onMouseLeave={() => setActiveLayer(null)}
          />

          {/* Base Notes - 사다리꼴 하단 */}
          <div
            className={clsx(
              'relative w-[200px] h-[60px] cursor-pointer transition-all duration-500',
              'transform perspective-100',
              activeLayer === 'base'
                ? 'bg-[var(--luxury-gold)]'
                : 'bg-[var(--luxury-silk)] hover:bg-[var(--luxury-rose-gold)]'
            )}
            style={{
              clipPath: 'polygon(10% 0%, 90% 0%, 100% 100%, 0% 100%)'
            }}
            onMouseEnter={() => setActiveLayer('base')}
            onMouseLeave={() => setActiveLayer(null)}
          />
        </div>

        {/* 노트 설명 */}
        <div className="mt-8 space-y-4">
          {/* Top Notes */}
          <div
            className={clsx(
              'p-4 rounded-lg border transition-all duration-300',
              activeLayer === 'top'
                ? 'border-[var(--luxury-gold)] bg-[var(--luxury-pearl)]'
                : 'border-[var(--luxury-silk)] bg-white'
            )}
          >
            <div className="flex items-baseline justify-between mb-2">
              <h4 className="font-semibold text-[var(--luxury-midnight)]">탑 노트</h4>
              <span className="text-xs text-[var(--luxury-stone)]">첫인상 · 0-15분</span>
            </div>
            <div className="text-[var(--luxury-charcoal)]">
              {renderNotes(topNotes, 'top')}
            </div>
          </div>

          {/* Heart Notes */}
          <div
            className={clsx(
              'p-4 rounded-lg border transition-all duration-300',
              activeLayer === 'heart'
                ? 'border-[var(--luxury-gold)] bg-[var(--luxury-pearl)]'
                : 'border-[var(--luxury-silk)] bg-white'
            )}
          >
            <div className="flex items-baseline justify-between mb-2">
              <h4 className="font-semibold text-[var(--luxury-midnight)]">하트 노트</h4>
              <span className="text-xs text-[var(--luxury-stone)]">핵심 · 15분-1시간</span>
            </div>
            <div className="text-[var(--luxury-charcoal)]">
              {renderNotes(heartNotes, 'heart')}
            </div>
          </div>

          {/* Base Notes */}
          <div
            className={clsx(
              'p-4 rounded-lg border transition-all duration-300',
              activeLayer === 'base'
                ? 'border-[var(--luxury-gold)] bg-[var(--luxury-pearl)]'
                : 'border-[var(--luxury-silk)] bg-white'
            )}
          >
            <div className="flex items-baseline justify-between mb-2">
              <h4 className="font-semibold text-[var(--luxury-midnight)]">베이스 노트</h4>
              <span className="text-xs text-[var(--luxury-stone)]">잔향 · 1시간 이상</span>
            </div>
            <div className="text-[var(--luxury-charcoal)]">
              {renderNotes(baseNotes, 'base')}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}