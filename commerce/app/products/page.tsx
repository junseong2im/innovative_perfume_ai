'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import ProductGrid from 'components/product-grid';

export default function ProductsPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [sortBy, setSortBy] = useState('featured');
  const [products, setProducts] = useState([]);

  const categories = [
    { id: 'all', name: '전체' },
    { id: 'fragrance', name: '향수' },
    { id: 'body', name: '바디 & 핸드' },
    { id: 'face', name: '페이스' },
    { id: 'hair', name: '헤어' },
    { id: 'home', name: '홈' }
  ];

  const sortOptions = [
    { id: 'featured', name: '추천순' },
    { id: 'newest', name: '최신순' },
    { id: 'price-low', name: '낮은 가격순' },
    { id: 'price-high', name: '높은 가격순' },
    { id: 'name', name: '이름순' }
  ];

  useEffect(() => {
    // Load products from API or localStorage
    const loadProducts = async () => {
      try {
        const response = await fetch('/api/products');
        const data = await response.json();
        setProducts(data.products || []);
      } catch (error) {
        // Fallback to mock data
        setProducts([]);
      }
    };
    loadProducts();
  }, [selectedCategory, sortBy]);

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--ivory-light)' }}>
      {/* Page Title */}
      <div className="bg-white border-b border-neutral-200 py-8">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8 text-center">
          <h1 className="text-3xl font-light text-neutral-900">제품</h1>
        </div>
      </div>

      <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
        <div className="flex flex-col lg:flex-row gap-8 py-8">
          {/* Sidebar Filters */}
          <aside className="lg:w-64 space-y-8">
            {/* Categories */}
            <div>
              <h3 className="font-medium text-neutral-900 mb-4">카테고리</h3>
              <ul className="space-y-2">
                {categories.map((category) => (
                  <li key={category.id}>
                    <button
                      onClick={() => setSelectedCategory(category.id)}
                      className={`w-full text-left px-3 py-2 rounded transition-colors ${
                        selectedCategory === category.id
                          ? 'bg-neutral-900 text-white'
                          : 'hover:bg-neutral-100 text-neutral-700'
                      }`}
                    >
                      {category.name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>

            {/* Price Range */}
            <div>
              <h3 className="font-medium text-neutral-900 mb-4">가격대</h3>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">₩0 - ₩50,000</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">₩50,000 - ₩100,000</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">₩100,000 - ₩200,000</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">₩200,000+</span>
                </label>
              </div>
            </div>

            {/* Fragrance Family */}
            <div>
              <h3 className="font-medium text-neutral-900 mb-4">향 계열</h3>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">플로럴</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">우디</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">시트러스</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">오리엔탈</span>
                </label>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-2" />
                  <span className="text-neutral-700">프레시</span>
                </label>
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1">
            {/* Sort Options */}
            <div className="flex items-center justify-between mb-6">
              <p className="text-neutral-600">
                {products.length}개의 제품
              </p>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-4 py-2 border border-neutral-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-neutral-900"
              >
                {sortOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Products Grid */}
            {products.length > 0 ? (
              <ProductGrid products={products} />
            ) : (
              <div className="text-center py-12">
                <p className="text-neutral-600">제품을 불러오는 중...</p>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}