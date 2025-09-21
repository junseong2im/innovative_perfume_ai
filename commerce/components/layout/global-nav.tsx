'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';

export default function GlobalNav() {
  const router = useRouter();
  const pathname = usePathname();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [wishlistCount, setWishlistCount] = useState(0);
  const [cartCount, setCartCount] = useState(0);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState('');
  const [showUserMenu, setShowUserMenu] = useState(false);

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

    // Listen for storage changes
    const handleStorageChange = () => {
      const wishlist = JSON.parse(localStorage.getItem('wishlist') || '[]');
      const cart = JSON.parse(localStorage.getItem('cart') || '[]');
      setWishlistCount(wishlist.length);
      setCartCount(cart.reduce((sum: number, item: any) => sum + item.quantity, 0));
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
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

  // Don't show on home page as it has its own hero navigation
  if (pathname === '/') {
    return null;
  }

  return (
    <nav className="sticky top-0 z-50 border-b text-white" style={{backgroundColor: 'var(--light-brown)', borderColor: 'var(--light-brown-dark)'}}>
      <div className="mx-auto max-w-screen-2xl px-4 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Left side - Logo and Navigation items */}
          <div className="flex items-center space-x-8">
            {/* Logo - Home Link */}
            <Link
              href="/"
              className="text-lg font-normal tracking-[0.12em] text-white hover:opacity-80 transition-opacity"
              style={{fontFamily: 'Playfair Display, Didot, Garamond, Times New Roman, serif'}}
            >
              Deulsoom
            </Link>

            {/* Divider */}
            <div className="h-6 w-px bg-neutral-400 opacity-50"></div>

            {/* Navigation Links */}
            <div className="flex items-center space-x-6">
              <Link
                href="/new-products"
                className={`text-sm font-light tracking-wide transition-colors ${
                  pathname === '/new-products'
                    ? 'text-white font-medium'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                신제품 & 추천
              </Link>
              <Link
                href="/products"
                className={`text-sm font-light tracking-wide transition-colors ${
                  pathname === '/products'
                    ? 'text-white font-medium'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                제품
              </Link>
              <Link
                href="/about"
                className={`text-sm font-light tracking-wide transition-colors ${
                  pathname === '/about'
                    ? 'text-white font-medium'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                스토리
              </Link>

              {/* Search Button */}
              <button
                onClick={() => setIsSearchOpen(!isSearchOpen)}
                className="text-neutral-300 hover:text-white transition-colors ml-2"
                aria-label="검색"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </button>
            </div>
          </div>

          {/* Right side - Login/User, Wishlist, Cart */}
          <div className="flex items-center space-x-5">
            {isLoggedIn ? (
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="text-sm font-light tracking-wide text-neutral-300 transition-colors hover:text-white flex items-center space-x-1"
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
                className="text-sm font-light tracking-wide text-neutral-300 transition-colors hover:text-white"
              >
                로그인
              </Link>
            )}

            <Link
              href="/wishlist"
              className="text-neutral-300 hover:text-white transition-colors relative"
              aria-label="위시리스트"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
              {wishlistCount > 0 && (
                <span className="absolute -top-1 -right-1 h-4 w-4 bg-white text-neutral-900 text-xs rounded-full flex items-center justify-center">
                  {wishlistCount}
                </span>
              )}
            </Link>

            <Link
              href="/cart"
              className="text-neutral-300 hover:text-white transition-colors relative"
              aria-label="카트"
            >
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
              </svg>
              {cartCount > 0 && (
                <span className="absolute -top-1 -right-1 h-4 w-4 bg-white text-neutral-900 text-xs rounded-full flex items-center justify-center">
                  {cartCount}
                </span>
              )}
            </Link>
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
  );
}