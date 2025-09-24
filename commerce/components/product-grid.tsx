import Image from 'next/image';
import Link from 'next/link';
import { Product } from 'lib/shopify/types';

interface ProductGridProps {
  products: Product[];
}

export default function ProductGrid({ products }: ProductGridProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 lg:py-24">
      {/* 상품 준비중 메인 표시 */}
      <div className="text-center max-w-lg mx-auto">
        {/* 향수 병 아이콘 */}
        <div className="mb-8 flex justify-center">
          <svg
            className="w-32 h-40 text-neutral-300"
            fill="currentColor"
            viewBox="0 0 100 150"
          >
            <rect x="35" y="20" width="30" height="120" rx="4" />
            <rect x="30" y="10" width="40" height="15" rx="2" />
            <rect x="32" y="5" width="36" height="10" rx="3" />
            <rect x="40" y="50" width="20" height="30" rx="1" fill="white" opacity="0.7" />

            {/* 라벨 */}
            <rect x="42" y="70" width="16" height="8" rx="1" fill="white" opacity="0.9" />
          </svg>
        </div>

        {/* 메인 메시지 */}
        <h3 className="text-2xl font-light text-neutral-900 mb-4">
          상품 준비중
        </h3>

        <p className="text-lg text-neutral-600 mb-6 leading-relaxed">
          들숨의 특별한 향수 컬렉션이<br />
          곧 여러분을 찾아갑니다
        </p>

        {/* 부가 설명 */}
        <div className="space-y-3 text-sm text-neutral-500">
          <p>✨ AI 기반 맞춤형 향수 제작</p>
          <p>🌿 자연에서 영감을 받은 고유한 향조</p>
          <p>🎯 개인의 취향을 반영한 특별한 레시피</p>
        </div>

        {/* 알림 신청 버튼 */}
        <div className="mt-8">
          <button
            className="inline-flex items-center px-6 py-3 text-sm font-medium text-white bg-neutral-800 rounded-lg hover:bg-neutral-700 transition-colors duration-200"
            disabled
          >
            <span className="mr-2">📧</span>
            출시 알림 신청
          </button>
        </div>

        {/* 예상 출시일 */}
        <div className="mt-6 pt-6 border-t border-neutral-200">
          <p className="text-xs text-neutral-400">
            예상 출시: 2024년 하반기
          </p>
        </div>
      </div>
    </div>
  );
}