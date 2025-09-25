import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ReactQueryDevtools } from 'react-query/devtools';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { Helmet } from 'react-helmet';

import theme from './theme/theme';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import Home from './pages/Home';
import Search from './pages/Search';
import Generate from './pages/Generate';
import About from './pages/About';
import Contact from './pages/Contact';
import Dashboard from './pages/Dashboard';
import NotFound from './pages/NotFound';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { LoadingProvider } from './context/LoadingContext';
import { AuthProvider } from './context/AuthContext';

// React Query 클라이언트 설정
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5분
      cacheTime: 10 * 60 * 1000, // 10분
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <Helmet>
        <title>Fragrance AI - 향수 검색 & 레시피 생성 플랫폼</title>
        <meta name="description" content="AI 기술로 완벽한 향수를 찾고, 나만의 향수 레시피를 생성하세요. 한국어 특화 향수 검색 및 추천 시스템." />
        <meta name="keywords" content="향수, 퍼퓨메, 향수 추천, 향수 검색, AI, 머신러닝, 레시피 생성" />
        <meta property="og:title" content="Fragrance AI - 향수 검색 & 레시피 생성" />
        <meta property="og:description" content="AI 기술로 완벽한 향수를 찾고, 나만의 향수 레시피를 생성하세요." />
        <meta property="og:type" content="website" />
        <link rel="canonical" href="/" />
      </Helmet>

      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AuthProvider>
            <LoadingProvider>
              <Router>
                <div className="App">
                  <Header />
                  <main style={{ minHeight: 'calc(100vh - 200px)' }}>
                    <Routes>
                      <Route path="/" element={<Home />} />
                      <Route path="/search" element={<Search />} />
                      <Route path="/generate" element={<Generate />} />
                      <Route path="/about" element={<About />} />
                      <Route path="/contact" element={<Contact />} />
                      <Route path="/dashboard" element={<Dashboard />} />
                      <Route path="*" element={<NotFound />} />
                    </Routes>
                  </main>
                  <Footer />
                </div>
              </Router>
            </LoadingProvider>
          </AuthProvider>
        </ThemeProvider>

        {/* Toast 알림 */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#333',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#4caf50',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#f44336',
                secondary: '#fff',
              },
            },
          }}
        />

        {/* React Query Devtools (개발 모드에서만) */}
        {process.env.NODE_ENV === 'development' && (
          <ReactQueryDevtools initialIsOpen={false} />
        )}
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;