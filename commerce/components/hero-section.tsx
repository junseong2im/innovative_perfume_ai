'use client';

import Link from 'next/link';
import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';

export default function HeroSection() {
  const router = useRouter();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [wishlistCount, setWishlistCount] = useState(0);
  const [cartCount, setCartCount] = useState(0);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState('');
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showWishlist, setShowWishlist] = useState(false);
  const [showCart, setShowCart] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Check login status
    const user = localStorage.getItem('user');
    if (user) {
      const userData = JSON.parse(user);
      setIsLoggedIn(true);
      setUserName(userData.name || userData.email.split('@')[0]);
    }

    // Load wishlist and cart counts
    const wishlist = JSON.parse(localStorage.getItem('wishlist') || '[]');
    const cart = JSON.parse(localStorage.getItem('cart') || '[]');
    setWishlistCount(wishlist.length);
    setCartCount(cart.reduce((sum: number, item: any) => sum + item.quantity, 0));

    // Parallax scroll effect
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    // Mouse tracking for subtle interactions
    const handleMouseMove = (e: MouseEvent) => {
      if (heroRef.current) {
        const rect = heroRef.current.getBoundingClientRect();
        setMousePosition({
          x: (e.clientX - rect.left - rect.width / 2) / rect.width,
          y: (e.clientY - rect.top - rect.height / 2) / rect.height
        });
      }
    };

    window.addEventListener('scroll', handleScroll);
    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery)}`);
      setIsSearchOpen(false);
      setSearchQuery('');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    setIsLoggedIn(false);
    setUserName('');
    setShowUserMenu(false);
    router.push('/');
  };

  const toggleWishlist = () => {
    setShowWishlist(!showWishlist);
    setShowCart(false);
  };

  const toggleCart = () => {
    setShowCart(!showCart);
    setShowWishlist(false);
  };

  return (
    <section className="bg-[var(--luxury-midnight)] text-[var(--luxury-pearl)]">
      {/* Extended Navigation */}
      <nav className="border-b border-[var(--luxury-gold)] border-opacity-30">
        <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Left side - Navigation items */}
            <div className="flex items-center space-x-6">
              <Link
                href="/new-products"
                className="text-sm font-light tracking-wide text-[var(--luxury-pearl)] transition-colors hover:text-[var(--luxury-gold)]"
              >
                신제품 & 추천
              </Link>
              <Link
                href="/products"
                className="text-sm font-light tracking-wide text-[var(--luxury-pearl)] transition-colors hover:text-[var(--luxury-gold)]"
              >
                제품
              </Link>
              <Link
                href="/about"
                className="text-sm font-light tracking-wide text-[var(--luxury-pearl)] transition-colors hover:text-[var(--luxury-gold)]"
              >
                스토리
              </Link>

              {/* Search Button */}
              <button
                onClick={() => setIsSearchOpen(!isSearchOpen)}
                className="text-[var(--luxury-pearl)] hover:text-[var(--luxury-gold)] transition-colors ml-2"
                aria-label="검색"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
            </div>

            {/* Right side - Login/User, Wishlist, Cart */}
            <div className="flex items-center space-x-5">
              {isLoggedIn ? (
                <div className="relative">
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="text-sm font-light tracking-wide text-[var(--luxury-pearl)] transition-colors hover:text-[var(--luxury-gold)] flex items-center space-x-1"
                  >
                    <span>{userName}님</span>
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                  {showUserMenu && (
                    <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50">
                      <Link href="/account" className="block px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100">
                        내 계정
                      </Link>
                      <Link href="/orders" className="block px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100">
                        주문 내역
                      </Link>
                      <button
                        onClick={handleLogout}
                        className="block w-full text-left px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
                      >
                        로그아웃
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <Link
                  href="/login"
                  className="text-sm font-light tracking-wide text-[var(--luxury-pearl)] transition-colors hover:text-[var(--luxury-gold)]"
                >
                  로그인
                </Link>
              )}

              <button
                onClick={toggleWishlist}
                className="text-[var(--luxury-pearl)] hover:text-[var(--luxury-gold)] transition-colors relative"
                aria-label="위시리스트"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
                {wishlistCount > 0 && (
                  <span className="absolute -top-1 -right-1 h-4 w-4 bg-[var(--luxury-gold)] text-[var(--luxury-midnight)] text-xs rounded-full flex items-center justify-center">
                    {wishlistCount}
                  </span>
                )}
              </button>

              <button
                onClick={toggleCart}
                className="text-[var(--luxury-pearl)] hover:text-[var(--luxury-gold)] transition-colors relative"
                aria-label="카트"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                </svg>
                {cartCount > 0 && (
                  <span className="absolute -top-1 -right-1 h-4 w-4 bg-[var(--luxury-gold)] text-[var(--luxury-midnight)] text-xs rounded-full flex items-center justify-center">
                    {cartCount}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* Search Bar - Hidden by default */}
          {isSearchOpen && (
            <div className="pb-4">
              <form onSubmit={handleSearch} className="relative">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="제품, 카테고리, 향료를 검색하세요..."
                  className="w-full px-4 py-2 pr-10 text-sm text-neutral-900 bg-white rounded-md focus:outline-none focus:ring-2 focus:ring-white"
                  autoFocus
                />
                <button
                  type="submit"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-neutral-600 hover:text-neutral-900"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </button>
              </form>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Content - Optimized Height */}
      <div ref={heroRef} className="relative flex items-stretch h-[70vh] min-h-[500px] max-h-[700px] overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-2 w-full relative">
          {/* Left side - Text content with parallax */}
          <div className="px-4 lg:px-8 xl:px-16 2xl:px-24 flex items-center py-8 lg:py-0 z-10">
            <div
              className="transform transition-transform duration-700 ease-out"
              style={{
                transform: `translateY(${Math.min(scrollY * -0.1, 0)}px) translateX(${mousePosition.x * 5}px)`,
              }}
            >
              <h1
                className="mb-8 text-5xl lg:text-7xl font-light tracking-[0.2em] text-[var(--luxury-gold)] opacity-0 animate-fadeInUp"
                style={{
                  fontFamily: 'var(--font-display)',
                  animationDelay: '0.3s',
                  animationFillMode: 'forwards'
                }}
              >
                Deulsoom
              </h1>
              <div className="space-y-6">
                <p
                  className="text-xs font-normal tracking-[0.3em] text-[var(--luxury-rose-gold)] uppercase opacity-0 animate-fadeInUp"
                  style={{
                    fontFamily: 'var(--font-body)',
                    animationDelay: '0.6s',
                    animationFillMode: 'forwards'
                  }}
                >
                  보이지 않는 가장 깊은 기억
                </p>
                <p
                  className="max-w-xl text-base lg:text-lg font-light text-[var(--luxury-pearl)] leading-relaxed opacity-0 animate-fadeInUp"
                  style={{
                    fontFamily: 'var(--font-body)',
                    letterSpacing: '0.03em',
                    animationDelay: '0.9s',
                    animationFillMode: 'forwards'
                  }}
                >
                  향기는 보이지 않는 가장 깊은 기억입니다.
                  우리는 당신의 보이지 않는 상상의 조각들을 모아,
                  세상에 단 하나뿐인 향기로 빚어냅니다.
                </p>
              </div>

              {/* CTA Buttons with hover effects */}
              <div className="mt-12 flex gap-4 opacity-0 animate-fadeInUp"
                style={{
                  animationDelay: '1.2s',
                  animationFillMode: 'forwards'
                }}
              >
                <a
                  href="#ai-creator"
                  className="group relative px-8 py-3 text-sm tracking-wider text-[var(--luxury-midnight)] bg-[var(--luxury-gold)] overflow-hidden transition-all duration-300 hover:text-[var(--luxury-pearl)]"
                  onClick={(e) => {
                    e.preventDefault();
                    document.querySelector('#ai-creator')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  <span className="relative z-10">나의 향기 시작하기</span>
                  <div className="absolute inset-0 bg-[var(--luxury-midnight)] transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left"></div>
                </a>
                <a
                  href="#collection"
                  className="px-8 py-3 text-sm tracking-wider text-[var(--luxury-pearl)] border border-[var(--luxury-gold)] transition-all duration-300 hover:bg-[var(--luxury-gold)] hover:text-[var(--luxury-midnight)]"
                  onClick={(e) => {
                    e.preventDefault();
                    document.querySelector('#collection')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  컬렉션 둘러보기
                </a>
              </div>
            </div>
          </div>

          {/* Right side - Multi-layered visual experience */}
          <div className="relative h-full">
            {/* Animated gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-r from-[var(--luxury-midnight)] via-transparent to-transparent z-10"></div>

            {/* Floating elements */}
            <div className="absolute inset-0 overflow-hidden">
              <div
                className="absolute top-20 right-20 w-32 h-32 bg-[var(--luxury-gold)] rounded-full opacity-10 blur-2xl"
                style={{
                  transform: `translate(${mousePosition.x * 30}px, ${mousePosition.y * 30}px)`,
                  transition: 'transform 0.3s ease-out'
                }}
              ></div>
              <div
                className="absolute bottom-40 right-40 w-48 h-48 bg-[var(--luxury-rose-gold)] rounded-full opacity-10 blur-3xl"
                style={{
                  transform: `translate(${mousePosition.x * -20}px, ${mousePosition.y * -20}px)`,
                  transition: 'transform 0.3s ease-out'
                }}
              ></div>
            </div>

            <div
              className="absolute inset-0 transition-transform duration-700 ease-out"
              style={{
                transform: `scale(1) translateY(0)`,
              }}
            >
              {/* 이미지 파일을 C:\Users\user\Desktop\새 폴더 (2)\Newss\commerce\public\images\ 폴더에 넣으세요 */}
              {/* 지원 형식: jpg, png, webp */}
              <img
                src="/images/image-5537275_1280.jpg"
                alt="Deulsoom Premium Fragrance"
                className="w-full h-full object-cover"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  const placeholder = document.getElementById('image-placeholder');
                  if (placeholder) placeholder.style.display = 'flex';
                }}
              />
              <div
                id="image-placeholder"
                className="absolute inset-0 bg-gradient-to-br from-[var(--luxury-charcoal)] to-[var(--luxury-obsidian)] items-center justify-center hidden"
              >
                <div className="text-center text-white/60">
                  <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-sm">이미지를 추가해주세요</p>
                  <p className="text-xs mt-1 opacity-60">public/images/hero-image.jpg</p>
                </div>
              </div>

              {/* Scroll indicator */}
              <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 animate-bounce">
                <div className="flex flex-col items-center">
                  <p className="text-xs tracking-[0.2em] text-[var(--luxury-pearl)] opacity-60 mb-2">SCROLL</p>
                  <svg className="w-6 h-6 text-[var(--luxury-gold)] opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}