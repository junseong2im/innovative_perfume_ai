import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#6B73FF', // 부드러운 보라색
      light: '#9C98FF',
      dark: '#3D5AFE',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#FF6B9D', // 따뜻한 핑크색
      light: '#FFB3D1',
      dark: '#E91E63',
      contrastText: '#ffffff',
    },
    background: {
      default: '#FAFBFF', // 매우 연한 보라색 배경
      paper: '#FFFFFF',
    },
    text: {
      primary: '#2D3748', // 진한 회색
      secondary: '#4A5568',
    },
    success: {
      main: '#48BB78',
      light: '#9AE6B4',
      dark: '#2F855A',
    },
    warning: {
      main: '#ED8936',
      light: '#FBD38D',
      dark: '#C05621',
    },
    error: {
      main: '#F56565',
      light: '#FEB2B2',
      dark: '#C53030',
    },
    info: {
      main: '#4299E1',
      light: '#90CDF4',
      dark: '#2B6CB0',
    },
    grey: {
      50: '#F7FAFC',
      100: '#EDF2F7',
      200: '#E2E8F0',
      300: '#CBD5E0',
      400: '#A0AEC0',
      500: '#718096',
      600: '#4A5568',
      700: '#2D3748',
      800: '#1A202C',
      900: '#171923',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Apple SD Gothic Neo"',
      '"Noto Sans KR"',
      '"Malgun Gothic"',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      lineHeight: 1.2,
      marginBottom: '1rem',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
      marginBottom: '0.8rem',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
      marginBottom: '0.6rem',
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.4,
      marginBottom: '0.5rem',
    },
    h5: {
      fontSize: '1.1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      marginBottom: '0.4rem',
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      marginBottom: '0.3rem',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      marginBottom: '1rem',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
      marginBottom: '0.8rem',
    },
    button: {
      fontWeight: 600,
      textTransform: 'none', // 버튼 텍스트 대문자 변환 비활성화
    },
  },
  shape: {
    borderRadius: 12, // 기본 border radius
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0,0,0,0.1)',
    '0px 4px 8px rgba(0,0,0,0.12)',
    '0px 8px 16px rgba(0,0,0,0.14)',
    '0px 12px 24px rgba(0,0,0,0.16)',
    '0px 16px 32px rgba(0,0,0,0.18)',
    '0px 20px 40px rgba(0,0,0,0.20)',
    // ... 더 많은 그림자 정의
  ],
  components: {
    // 글로벌 컴포넌트 스타일
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '8px 20px',
          fontSize: '0.9rem',
          fontWeight: 600,
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 4px 12px rgba(107, 115, 255, 0.3)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #6B73FF 0%, #9C98FF 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #3D5AFE 0%, #6B73FF 100%)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.08)',
          border: '1px solid #E2E8F0',
          '&:hover': {
            boxShadow: '0px 8px 30px rgba(0, 0, 0, 0.12)',
            transform: 'translateY(-2px)',
            transition: 'all 0.3s ease',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            '&:hover fieldset': {
              borderColor: '#6B73FF',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#6B73FF',
              borderWidth: 2,
            },
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          fontWeight: 500,
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 16,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid #E2E8F0',
          color: '#2D3748',
        },
      },
    },
  },
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 900,
      lg: 1200,
      xl: 1536,
    },
  },
  spacing: 8, // 기본 spacing unit (8px)
});

// 다크 테마 (추후 확장용)
export const darkTheme = createTheme({
  ...theme,
  palette: {
    ...theme.palette,
    mode: 'dark',
    background: {
      default: '#0F0F1A',
      paper: '#1A1B2E',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#A0AEC0',
    },
  },
});

export default theme;