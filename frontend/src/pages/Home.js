import React from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  Chip,
  Avatar,
  Rating,
  Fade,
  Grow,
} from '@mui/material';
import {
  Search as SearchIcon,
  AutoAwesome as GenerateIcon,
  TrendingUp as TrendingIcon,
  Star as StarIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import HeroSection from '../components/home/HeroSection';
import FeatureCard from '../components/home/FeatureCard';
import PopularFragrances from '../components/home/PopularFragrances';
import TestimonialSection from '../components/home/TestimonialSection';
import StatsSection from '../components/home/StatsSection';

const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <SearchIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'AI 향수 검색',
      description: '자연어로 원하는 향수를 설명하면 AI가 최적의 향수를 찾아드립니다.',
      path: '/search',
      color: 'primary',
    },
    {
      icon: <GenerateIcon sx={{ fontSize: 40, color: 'secondary.main' }} />,
      title: '레시피 생성',
      description: '개인의 취향과 상황에 맞는 맞춤형 향수 레시피를 AI가 생성합니다.',
      path: '/generate',
      color: 'secondary',
    },
    {
      icon: <TrendingIcon sx={{ fontSize: 40, color: 'success.main' }} />,
      title: '트렌드 분석',
      description: '실시간 향수 트렌드와 인기 브랜드 분석 정보를 제공합니다.',
      path: '/dashboard',
      color: 'success',
    },
  ];

  const recentSearches = [
    '따뜻하고 달콤한 겨울 향수',
    '상쾌한 시트러스 여름 향수',
    '우아한 플로럴 데일리 향수',
    '깊고 신비로운 오리엔탈 향수',
  ];

  return (
    <Box>
      {/* 히어로 섹션 */}
      <HeroSection />

      {/* 주요 기능 섹션 */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
        >
          <Typography
            variant="h2"
            align="center"
            gutterBottom
            sx={{ mb: 6, fontWeight: 700 }}
          >
            AI 기반 향수 플랫폼
          </Typography>
        </motion.div>

        <Grid container spacing={4}>
          {features.map((feature, index) => (
            <Grid item xs={12} md={4} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <FeatureCard
                  {...feature}
                  onClick={() => navigate(feature.path)}
                />
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>

      {/* 인기 검색어 섹션 */}
      <Box sx={{ backgroundColor: 'grey.50', py: 6 }}>
        <Container maxWidth="lg">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <Typography variant="h4" align="center" gutterBottom sx={{ mb: 4 }}>
              인기 검색어
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: 2 }}>
              {recentSearches.map((search, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  viewport={{ once: true }}
                >
                  <Chip
                    label={search}
                    onClick={() => navigate(`/search?q=${encodeURIComponent(search)}`)}
                    variant="outlined"
                    sx={{
                      fontSize: '0.9rem',
                      padding: '8px 12px',
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'primary.main',
                        color: 'white',
                        transform: 'translateY(-2px)',
                        transition: 'all 0.2s ease',
                      },
                    }}
                  />
                </motion.div>
              ))}
            </Box>
          </motion.div>
        </Container>
      </Box>

      {/* 인기 향수 섹션 */}
      <PopularFragrances />

      {/* 통계 섹션 */}
      <StatsSection />

      {/* 사용자 후기 섹션 */}
      <TestimonialSection />

      {/* CTA 섹션 */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #6B73FF 0%, #9C98FF 100%)',
          color: 'white',
          py: 8,
        }}
      >
        <Container maxWidth="md">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <Typography variant="h3" align="center" gutterBottom sx={{ fontWeight: 700 }}>
              완벽한 향수를 찾아보세요
            </Typography>
            <Typography variant="h6" align="center" sx={{ mb: 4, opacity: 0.9 }}>
              AI가 당신의 취향을 분석하여 최적의 향수를 추천해드립니다
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
              <Button
                component={RouterLink}
                to="/search"
                variant="contained"
                size="large"
                sx={{
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  border: '2px solid rgba(255, 255, 255, 0.3)',
                  backdropFilter: 'blur(10px)',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.3)',
                    transform: 'translateY(-2px)',
                  },
                }}
                startIcon={<SearchIcon />}
              >
                향수 검색하기
              </Button>
              <Button
                component={RouterLink}
                to="/generate"
                variant="outlined"
                size="large"
                sx={{
                  color: 'white',
                  borderColor: 'rgba(255, 255, 255, 0.5)',
                  '&:hover': {
                    borderColor: 'white',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    transform: 'translateY(-2px)',
                  },
                }}
                startIcon={<GenerateIcon />}
              >
                레시피 생성하기
              </Button>
            </Box>
          </motion.div>
        </Container>
      </Box>

      {/* 최근 업데이트 섹션 */}
      <Container maxWidth="lg" sx={{ py: 6 }}>
        <Typography variant="h4" align="center" gutterBottom sx={{ mb: 4 }}>
          최근 업데이트
        </Typography>
        <Grid container spacing={3}>
          {[
            {
              title: 'AI 모델 성능 향상',
              description: '한국어 향수 검색 정확도가 15% 향상되었습니다.',
              date: '2024.01.15',
              badge: 'AI 업데이트',
            },
            {
              title: '새로운 브랜드 추가',
              description: '국내외 50개 브랜드의 향수 데이터가 추가되었습니다.',
              date: '2024.01.12',
              badge: '데이터 확장',
            },
            {
              title: '모바일 최적화',
              description: '모바일에서 더욱 편리하게 이용할 수 있습니다.',
              date: '2024.01.10',
              badge: 'UX 개선',
            },
          ].map((update, index) => (
            <Grid item xs={12} md={4} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Chip
                        label={update.badge}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                      <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto' }}>
                        {update.date}
                      </Typography>
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {update.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {update.description}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default Home;