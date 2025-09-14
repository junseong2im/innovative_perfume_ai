import React, { useState, useEffect, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  TextField,
  InputAdornment,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Alert,
  Pagination,
  Rating,
  Avatar,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  ExpandMore as ExpandMoreIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
  Share as ShareIcon,
  TrendingUp as TrendingIcon,
  LocalOffer as PriceIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import debounce from 'lodash.debounce';

import { useFragranceSearch } from '../hooks/useFragranceSearch';
import SearchResultCard from '../components/search/SearchResultCard';
import SearchFilters from '../components/search/SearchFilters';
import SearchSuggestions from '../components/search/SearchSuggestions';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatPrice } from '../utils/formatters';

const Search = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // 검색 상태
  const [query, setQuery] = useState(searchParams.get('q') || '');
  const [debouncedQuery, setDebouncedQuery] = useState(query);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);

  // 필터 상태
  const [filters, setFilters] = useState({
    gender: searchParams.get('gender') || '',
    priceRange: [0, 300000],
    brands: searchParams.get('brands')?.split(',').filter(Boolean) || [],
    notes: searchParams.get('notes')?.split(',').filter(Boolean) || [],
    season: searchParams.get('season') || '',
    rating: 0,
    sortBy: 'relevance',
  });

  // UI 상태
  const [showFilters, setShowFilters] = useState(false);
  const [favorites, setFavorites] = useState(new Set());

  // 디바운스된 검색어 업데이트
  const debouncedSetQuery = useMemo(
    () => debounce((value) => {
      setDebouncedQuery(value);
      setCurrentPage(1);
    }, 500),
    []
  );

  useEffect(() => {
    debouncedSetQuery(query);
    return () => debouncedSetQuery.cancel();
  }, [query, debouncedSetQuery]);

  // 검색 API 호출
  const {
    data: searchResults,
    isLoading,
    error,
    refetch,
  } = useFragranceSearch({
    query: debouncedQuery,
    filters,
    page: currentPage,
    limit: itemsPerPage,
  });

  // URL 파라미터 업데이트
  useEffect(() => {
    const params = new URLSearchParams();
    if (debouncedQuery) params.set('q', debouncedQuery);
    if (filters.gender) params.set('gender', filters.gender);
    if (filters.brands.length) params.set('brands', filters.brands.join(','));
    if (filters.notes.length) params.set('notes', filters.notes.join(','));
    if (filters.season) params.set('season', filters.season);

    setSearchParams(params, { replace: true });
  }, [debouncedQuery, filters, setSearchParams]);

  // 검색 제출
  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim()) {
      setDebouncedQuery(query.trim());
      setCurrentPage(1);
    }
  };

  // 필터 변경
  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setCurrentPage(1);
  };

  // 즐겨찾기 토글
  const toggleFavorite = (fragranceId) => {
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(fragranceId)) {
        newFavorites.delete(fragranceId);
      } else {
        newFavorites.add(fragranceId);
      }
      return newFavorites;
    });
  };

  // 페이지 변경
  const handlePageChange = (event, page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // 인기 검색어
  const popularSearches = [
    '톰포드 블랙 오키드',
    '샤넬 넘버 5',
    '디올 사바주',
    '조말론 라임 바질',
    '크리드 아벤투스',
  ];

  const totalPages = Math.ceil((searchResults?.total || 0) / itemsPerPage);

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* 검색 헤더 */}
      <Box sx={{ mb: 4 }}>
        <Typography
          variant="h3"
          align="center"
          gutterBottom
          sx={{
            fontWeight: 700,
            background: 'linear-gradient(135deg, #6B73FF 0%, #FF6B9D 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            mb: 3,
          }}
        >
          향수 검색
        </Typography>

        {/* 검색 바 */}
        <Box
          component="form"
          onSubmit={handleSearch}
          sx={{
            maxWidth: 800,
            mx: 'auto',
            mb: 3,
          }}
        >
          <TextField
            fullWidth
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="원하는 향수를 자세히 설명해주세요 (예: 따뜻하고 달콤한 겨울 향수)"
            variant="outlined"
            size="large"
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: 'background.paper',
                borderRadius: 3,
                '&:hover fieldset': {
                  borderColor: 'primary.main',
                },
                '&.Mui-focused fieldset': {
                  borderColor: 'primary.main',
                  borderWidth: 2,
                },
              },
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="primary" sx={{ fontSize: 28 }} />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <Button
                    type="submit"
                    variant="contained"
                    sx={{ mr: 1, borderRadius: 2 }}
                    disabled={isLoading}
                  >
                    {isLoading ? <CircularProgress size={20} color="inherit" /> : '검색'}
                  </Button>
                  <IconButton
                    onClick={() => setShowFilters(!showFilters)}
                    color={showFilters ? 'primary' : 'default'}
                  >
                    <FilterIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
        </Box>

        {/* 인기 검색어 (검색 결과가 없을 때만 표시) */}
        {!debouncedQuery && (
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              인기 검색어
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 1 }}>
              {popularSearches.map((search) => (
                <Chip
                  key={search}
                  label={search}
                  variant="outlined"
                  onClick={() => {
                    setQuery(search);
                    setDebouncedQuery(search);
                  }}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: 'primary.main',
                      color: 'white',
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* 사이드바 필터 */}
        <Grid item xs={12} md={3}>
          <AnimatePresence>
            {(showFilters || window.innerWidth >= 960) && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <SearchFilters
                  filters={filters}
                  onFilterChange={handleFilterChange}
                  searchResults={searchResults}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </Grid>

        {/* 검색 결과 */}
        <Grid item xs={12} md={9}>
          {/* 결과 헤더 */}
          {searchResults && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" color="text.secondary">
                총 <strong>{searchResults.total.toLocaleString()}</strong>개의 향수를 찾았습니다
              </Typography>

              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>정렬</InputLabel>
                <Select
                  value={filters.sortBy}
                  onChange={(e) => handleFilterChange({ sortBy: e.target.value })}
                  label="정렬"
                >
                  <MenuItem value="relevance">관련도순</MenuItem>
                  <MenuItem value="rating">평점순</MenuItem>
                  <MenuItem value="price_asc">가격 낮은순</MenuItem>
                  <MenuItem value="price_desc">가격 높은순</MenuItem>
                  <MenuItem value="newest">최신순</MenuItem>
                </Select>
              </FormControl>
            </Box>
          )}

          {/* 로딩 상태 */}
          {isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <LoadingSpinner message="향수를 검색하고 있습니다..." />
            </Box>
          )}

          {/* 에러 상태 */}
          {error && (
            <Alert
              severity="error"
              action={
                <Button color="inherit" size="small" onClick={refetch}>
                  다시 시도
                </Button>
              }
              sx={{ mb: 3 }}
            >
              검색 중 오류가 발생했습니다: {error.message}
            </Alert>
          )}

          {/* 검색 결과 */}
          {searchResults?.results && (
            <>
              <Grid container spacing={3}>
                <AnimatePresence>
                  {searchResults.results.map((fragrance, index) => (
                    <Grid item xs={12} sm={6} lg={4} key={fragrance.id}>
                      <motion.div
                        layout
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{
                          duration: 0.4,
                          delay: index * 0.05,
                        }}
                        whileHover={{ y: -4 }}
                      >
                        <SearchResultCard
                          fragrance={fragrance}
                          isFavorite={favorites.has(fragrance.id)}
                          onToggleFavorite={() => toggleFavorite(fragrance.id)}
                          onViewDetail={() => navigate(`/fragrance/${fragrance.id}`)}
                        />
                      </motion.div>
                    </Grid>
                  ))}
                </AnimatePresence>
              </Grid>

              {/* 페이지네이션 */}
              {totalPages > 1 && (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 6 }}>
                  <Pagination
                    count={totalPages}
                    page={currentPage}
                    onChange={handlePageChange}
                    color="primary"
                    size="large"
                    showFirstButton
                    showLastButton
                  />
                </Box>
              )}
            </>
          )}

          {/* 검색 결과 없음 */}
          {searchResults?.results?.length === 0 && debouncedQuery && (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Typography variant="h5" gutterBottom color="text.secondary">
                검색 결과가 없습니다
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                다른 검색어를 시도해보시거나 필터를 조정해보세요
              </Typography>

              <SearchSuggestions
                query={debouncedQuery}
                onSuggestionClick={(suggestion) => {
                  setQuery(suggestion);
                  setDebouncedQuery(suggestion);
                }}
              />

              <Button
                variant="outlined"
                onClick={() => {
                  setQuery('');
                  setDebouncedQuery('');
                  setFilters({
                    gender: '',
                    priceRange: [0, 300000],
                    brands: [],
                    notes: [],
                    season: '',
                    rating: 0,
                    sortBy: 'relevance',
                  });
                }}
                sx={{ mt: 2 }}
              >
                필터 초기화
              </Button>
            </Box>
          )}

          {/* 검색 제안사항 (초기 상태) */}
          {!debouncedQuery && !isLoading && (
            <Box sx={{ py: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                추천 검색 방법
              </Typography>

              <Grid container spacing={3}>
                {[
                  {
                    title: '상황별 검색',
                    description: '데이트, 직장, 파티 등 상황을 설명해보세요',
                    example: '로맨틱한 데이트에 어울리는 향수',
                    icon: <TrendingIcon />,
                  },
                  {
                    title: '계절별 검색',
                    description: '봄, 여름, 가을, 겨울 등 계절감을 표현해보세요',
                    example: '따뜻한 겨울 밤에 어울리는 향수',
                    icon: <FilterIcon />,
                  },
                  {
                    title: '감정별 검색',
                    description: '기분이나 감정을 자유롭게 표현해보세요',
                    example: '우아하고 신비로운 느낌의 향수',
                    icon: <FavoriteIcon />,
                  },
                ].map((tip, index) => (
                  <Grid item xs={12} md={4} key={index}>
                    <Card
                      sx={{
                        height: '100%',
                        cursor: 'pointer',
                        transition: 'transform 0.2s',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                        },
                      }}
                      onClick={() => {
                        setQuery(tip.example);
                        setDebouncedQuery(tip.example);
                      }}
                    >
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                          <Avatar
                            sx={{
                              backgroundColor: 'primary.main',
                              mr: 2,
                            }}
                          >
                            {tip.icon}
                          </Avatar>
                          <Typography variant="h6">
                            {tip.title}
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          {tip.description}
                        </Typography>
                        <Typography variant="body2" color="primary.main" sx={{ fontStyle: 'italic' }}>
                          예: "{tip.example}"
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default Search;