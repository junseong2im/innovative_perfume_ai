import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  TextField,
  Button,
  InputAdornment,
  Fade,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Search as SearchIcon,
  AutoAwesome as MagicIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

const HeroSection = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPlaceholder, setCurrentPlaceholder] = useState(0);

  const placeholders = [
    '따뜻하고 달콤한 겨울 향수를 찾고 있어요',
    '상쾌한 시트러스 계열의 여름 향수',
    '우아하고 로맨틱한 데이트 향수',
    '깊고 신비로운 밤 향수',
    '가벼운 일상 향수 추천해주세요',
  ];

  // 플레이스홀더 자동 변경
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPlaceholder((prev) => (prev + 1) % placeholders.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [placeholders.length]);

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.6,
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6, ease: 'easeOut' },
    },
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        background: `linear-gradient(135deg,
          rgba(107, 115, 255, 0.1) 0%,
          rgba(156, 152, 255, 0.1) 50%,
          rgba(255, 107, 157, 0.1) 100%
        )`,
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `url('data:image/svg+xml,${encodeURIComponent(`
            <svg width="60" height="60" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
                  <path d="M 60 0 L 0 0 0 60" fill="none" stroke="rgba(107, 115, 255, 0.1)" stroke-width="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          `)})`,
          opacity: 0.3,
        },
      }}
    >
      {/* 배경 애니메이션 요소들 */}
      <Box
        component={motion.div}
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.1, 0.3, 0.1],
          rotate: [0, 180, 360],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: 'linear',
        }}
        sx={{
          position: 'absolute',
          top: '10%',
          right: '10%',
          width: 200,
          height: 200,
          borderRadius: '50%',
          background: 'linear-gradient(45deg, #6B73FF, #FF6B9D)',
          filter: 'blur(40px)',
          zIndex: 0,
        }}
      />

      <Box
        component={motion.div}
        animate={{
          scale: [1, 0.8, 1],
          opacity: [0.1, 0.2, 0.1],
          rotate: [360, 180, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: 'linear',
        }}
        sx={{
          position: 'absolute',
          bottom: '20%',
          left: '5%',
          width: 150,
          height: 150,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #9C98FF, #6B73FF)',
          filter: 'blur(30px)',
          zIndex: 0,
        }}
      />

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <Box sx={{ textAlign: 'center', py: 8 }}>
            {/* 메인 타이틀 */}
            <motion.div variants={itemVariants}>
              <Typography
                variant={isMobile ? 'h3' : 'h1'}
                sx={{
                  fontWeight: 800,
                  mb: 3,
                  background: 'linear-gradient(135deg, #6B73FF 0%, #FF6B9D 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                완벽한 향수를 찾는
                <br />
                가장 스마트한 방법
              </Typography>
            </motion.div>

            {/* 서브 타이틀 */}
            <motion.div variants={itemVariants}>
              <Typography
                variant={isMobile ? 'h6' : 'h5'}
                color="text.secondary"
                sx={{
                  mb: 5,
                  maxWidth: 600,
                  mx: 'auto',
                  lineHeight: 1.6,
                }}
              >
                AI가 당신의 취향을 분석하여 최적의 향수를 찾아드리고,
                <br />
                개인 맞춤 레시피까지 생성해드립니다.
              </Typography>
            </motion.div>

            {/* 검색 바 */}
            <motion.div variants={itemVariants}>
              <Box
                component="form"
                onSubmit={handleSearch}
                sx={{
                  maxWidth: 600,
                  mx: 'auto',
                  mb: 4,
                }}
              >
                <TextField
                  fullWidth
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={placeholders[currentPlaceholder]}
                  variant="outlined"
                  size="large"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      backgroundColor: 'rgba(255, 255, 255, 0.9)',
                      backdropFilter: 'blur(20px)',
                      borderRadius: 3,
                      fontSize: '1.1rem',
                      padding: '8px 0',
                      border: '1px solid rgba(107, 115, 255, 0.2)',
                      '&:hover': {
                        border: '1px solid rgba(107, 115, 255, 0.4)',
                      },
                      '&.Mui-focused': {
                        border: '2px solid #6B73FF',
                        boxShadow: '0 0 20px rgba(107, 115, 255, 0.3)',
                      },
                    },
                    '& .MuiOutlinedInput-input': {
                      padding: '16px 20px',
                    },
                  }}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <SearchIcon
                          sx={{
                            color: 'primary.main',
                            fontSize: 28,
                            ml: 1,
                          }}
                        />
                      </InputAdornment>
                    ),
                    endAdornment: (
                      <InputAdornment position="end">
                        <Button
                          type="submit"
                          variant="contained"
                          sx={{
                            mr: 1,
                            borderRadius: 2,
                            padding: '8px 20px',
                            background: 'linear-gradient(135deg, #6B73FF 0%, #9C98FF 100%)',
                            '&:hover': {
                              background: 'linear-gradient(135deg, #3D5AFE 0%, #6B73FF 100%)',
                              transform: 'translateY(-1px)',
                            },
                          }}
                        >
                          검색
                        </Button>
                      </InputAdornment>
                    ),
                  }}
                />
              </Box>
            </motion.div>

            {/* 액션 버튼들 */}
            <motion.div variants={itemVariants}>
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate('/search')}
                  startIcon={<SearchIcon />}
                  sx={{
                    borderRadius: 3,
                    padding: '12px 30px',
                    fontSize: '1rem',
                    background: 'linear-gradient(135deg, #6B73FF 0%, #9C98FF 100%)',
                    boxShadow: '0 8px 25px rgba(107, 115, 255, 0.3)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #3D5AFE 0%, #6B73FF 100%)',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 12px 35px rgba(107, 115, 255, 0.4)',
                    },
                  }}
                >
                  향수 탐색하기
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => navigate('/generate')}
                  startIcon={<MagicIcon />}
                  sx={{
                    borderRadius: 3,
                    padding: '12px 30px',
                    fontSize: '1rem',
                    borderColor: 'primary.main',
                    color: 'primary.main',
                    backgroundColor: 'rgba(255, 255, 255, 0.8)',
                    backdropFilter: 'blur(20px)',
                    '&:hover': {
                      backgroundColor: 'primary.main',
                      color: 'white',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 12px 35px rgba(107, 115, 255, 0.3)',
                    },
                  }}
                >
                  레시피 생성하기
                </Button>
              </Box>
            </motion.div>

            {/* 통계 정보 */}
            <motion.div
              variants={itemVariants}
              style={{
                marginTop: 60,
                display: 'flex',
                justifyContent: 'center',
                gap: 40,
                flexWrap: 'wrap',
              }}
            >
              {[
                { label: '등록된 향수', value: '50,000+' },
                { label: '만족도', value: '98.5%' },
                { label: '생성된 레시피', value: '100,000+' },
              ].map((stat, index) => (
                <Box key={index} sx={{ textAlign: 'center' }}>
                  <Typography
                    variant="h4"
                    sx={{
                      fontWeight: 700,
                      background: 'linear-gradient(135deg, #6B73FF 0%, #FF6B9D 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                    }}
                  >
                    {stat.value}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {stat.label}
                  </Typography>
                </Box>
              ))}
            </motion.div>
          </Box>
        </motion.div>
      </Container>

      {/* 스크롤 인디케이터 */}
      <Box
        component={motion.div}
        animate={{ y: [0, 10, 0] }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        sx={{
          position: 'absolute',
          bottom: 30,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          opacity: 0.6,
          cursor: 'pointer',
        }}
        onClick={() => {
          window.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
        }}
      >
        <Typography variant="body2" sx={{ mb: 1 }}>
          더 알아보기
        </Typography>
        <Box
          sx={{
            width: 2,
            height: 30,
            backgroundColor: 'primary.main',
            borderRadius: 1,
          }}
        />
      </Box>
    </Box>
  );
};

export default HeroSection;