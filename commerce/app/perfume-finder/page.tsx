'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Button from 'components/ui/button';
import clsx from 'clsx';

interface Question {
  id: string;
  question: string;
  options: {
    value: string;
    label: string;
    image?: string;
    mood?: string;
  }[];
}

const questions: Question[] = [
  {
    id: 'season',
    question: '어떤 계절의 산책을 가장 좋아하시나요?',
    options: [
      { value: 'spring', label: '봄', mood: '벚꽃이 흩날리는 따스한 봄날' },
      { value: 'summer', label: '여름', mood: '햇살 가득한 여름 해변' },
      { value: 'autumn', label: '가을', mood: '단풍잎이 떨어지는 고요한 가을' },
      { value: 'winter', label: '겨울', mood: '눈 내리는 겨울 밤' }
    ]
  },
  {
    id: 'time',
    question: '오늘 당신의 기분은 어떤 색에 가까운가요?',
    options: [
      { value: 'pink', label: '핑크', mood: '로맨틱하고 부드러운' },
      { value: 'blue', label: '블루', mood: '차분하고 신비로운' },
      { value: 'yellow', label: '옐로우', mood: '밝고 활기찬' },
      { value: 'purple', label: '퍼플', mood: '우아하고 고급스러운' }
    ]
  },
  {
    id: 'place',
    question: '향수를 뿌리고 가고 싶은 곳은?',
    options: [
      { value: 'office', label: '오피스', mood: '전문적이고 세련된 공간' },
      { value: 'date', label: '데이트', mood: '로맨틱한 저녁 식사' },
      { value: 'party', label: '파티', mood: '활기찬 사교 모임' },
      { value: 'nature', label: '자연', mood: '평화로운 숲속 산책' }
    ]
  },
  {
    id: 'personality',
    question: '당신을 가장 잘 표현하는 단어는?',
    options: [
      { value: 'confident', label: '자신감', mood: '당당하고 카리스마 있는' },
      { value: 'gentle', label: '부드러움', mood: '따뜻하고 포근한' },
      { value: 'mysterious', label: '신비로움', mood: '알 수 없는 매력적인' },
      { value: 'playful', label: '발랄함', mood: '즐겁고 경쾌한' }
    ]
  },
  {
    id: 'memory',
    question: '가장 좋아하는 향의 기억은?',
    options: [
      { value: 'flower', label: '꽃향기', mood: '정원의 싱그러운 꽃들' },
      { value: 'ocean', label: '바다향', mood: '시원한 바닷바람' },
      { value: 'wood', label: '나무향', mood: '따뜻한 나무 냄새' },
      { value: 'sweet', label: '달콤한 향', mood: '달콤한 바닐라와 과일' }
    ]
  }
];

export default function PerfumeFinderPage() {
  const router = useRouter();
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnswer = (value: string) => {
    const newAnswers = { ...answers, [questions[currentQuestion].id]: value };
    setAnswers(newAnswers);

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
    } else {
      // 모든 질문에 답변 완료 - 분석 시작
      analyzeAndRecommend(newAnswers);
    }
  };

  const analyzeAndRecommend = async (finalAnswers: Record<string, string>) => {
    setIsAnalyzing(true);

    // AI 분석 시뮬레이션 (실제로는 API 호출)
    setTimeout(() => {
      // 답변 기반 추천 로직
      const queryParams = new URLSearchParams(finalAnswers);
      router.push(`/search?${queryParams.toString()}&recommendation=true`);
    }, 2000);
  };

  const goBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1);
    }
  };

  const progress = ((currentQuestion + 1) / questions.length) * 100;

  if (isAnalyzing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-[var(--luxury-cream)] to-[var(--luxury-pearl)]">
        <div className="text-center space-y-6">
          <div className="animate-pulse">
            <div className="w-24 h-24 mx-auto rounded-full bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)]" />
          </div>
          <h2 className="text-2xl font-[var(--font-display)] text-[var(--luxury-midnight)]">
            당신만의 향수를 찾고 있습니다...
          </h2>
          <p className="text-[var(--luxury-stone)]">
            AI가 당신의 답변을 분석하고 있어요
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[var(--luxury-cream)] to-[var(--luxury-pearl)]">
      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* 헤더 */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-[var(--font-display)] text-[var(--luxury-midnight)] mb-4">
            퍼퓸 파인더
          </h1>
          <p className="text-lg text-[var(--luxury-charcoal)]">
            몇 가지 질문으로 당신에게 완벽한 향수를 찾아드려요
          </p>
        </div>

        {/* 진행 바 */}
        <div className="mb-12">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-[var(--luxury-stone)]">
              질문 {currentQuestion + 1} / {questions.length}
            </span>
            <span className="text-sm text-[var(--luxury-stone)]">
              {Math.round(progress)}% 완료
            </span>
          </div>
          <div className="w-full bg-[var(--luxury-silk)] rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)] transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* 질문 섹션 */}
        <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12">
          <h2 className="text-2xl md:text-3xl font-[var(--font-display)] text-[var(--luxury-midnight)] mb-8 text-center">
            {questions[currentQuestion].question}
          </h2>

          {/* 옵션들 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {questions[currentQuestion].options.map((option) => (
              <button
                key={option.value}
                onClick={() => handleAnswer(option.value)}
                className={clsx(
                  'group relative p-6 rounded-xl border-2 transition-all duration-300',
                  'hover:border-[var(--luxury-gold)] hover:shadow-lg',
                  'bg-gradient-to-br from-white to-[var(--luxury-pearl)]',
                  'border-[var(--luxury-silk)]'
                )}
              >
                <div className="text-left">
                  <h3 className="text-xl font-semibold text-[var(--luxury-midnight)] mb-2">
                    {option.label}
                  </h3>
                  {option.mood && (
                    <p className="text-sm text-[var(--luxury-charcoal)]">
                      {option.mood}
                    </p>
                  )}
                </div>
                <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-2xl">→</span>
                </div>
              </button>
            ))}
          </div>

          {/* 네비게이션 */}
          <div className="flex justify-between items-center mt-8">
            {currentQuestion > 0 ? (
              <Button
                variant="secondary"
                onClick={goBack}
                className="px-6"
              >
                이전
              </Button>
            ) : (
              <div />
            )}

            <button
              onClick={() => router.push('/search')}
              className="text-sm text-[var(--luxury-stone)] hover:text-[var(--luxury-midnight)] transition-colors"
            >
              건너뛰기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}