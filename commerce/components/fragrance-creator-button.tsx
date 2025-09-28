'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

// 키워드에 따른 배경 이미지 매핑
const keywordBackgrounds: { [key: string]: string } = {
  '봄': 'linear-gradient(135deg, #FFB6C1 0%, #FFE4E1 100%)',
  '여름': 'linear-gradient(135deg, #87CEEB 0%, #98FB98 100%)',
  '가을': 'linear-gradient(135deg, #DEB887 0%, #D2691E 100%)',
  '겨울': 'linear-gradient(135deg, #E0FFFF 0%, #B0C4DE 100%)',
  '바다': 'linear-gradient(135deg, #1E90FF 0%, #00CED1 100%)',
  '숲': 'linear-gradient(135deg, #228B22 0%, #90EE90 100%)',
  '도시': 'linear-gradient(135deg, #708090 0%, #C0C0C0 100%)',
  '사랑': 'linear-gradient(135deg, #FF69B4 0%, #FFB6C1 100%)',
  '기억': 'linear-gradient(135deg, #9370DB 0%, #DDA0DD 100%)',
  '꿈': 'linear-gradient(135deg, #6A5ACD 0%, #9370DB 100%)',
  '아침': 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)',
  '밤': 'linear-gradient(135deg, #191970 0%, #483D8B 100%)',
};

const memoryPrompts = [
  "당신의 기억은 어떤 향기인가요?",
  "첫사랑의 순간을 향으로 표현한다면?",
  "가장 행복했던 날의 공기는?",
  "고요한 새벽의 숨결을 담아보세요",
  "당신만의 비밀스러운 정원은?"
];

export default function FragranceCreatorButton() {
  const [userInput, setUserInput] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState(memoryPrompts[0]);
  const [backgroundGradient, setBackgroundGradient] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [relatedImages, setRelatedImages] = useState<string[]>([]);

  // Rotate prompts
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPrompt(prev => {
        const currentIndex = memoryPrompts.indexOf(prev);
        return memoryPrompts[(currentIndex + 1) % memoryPrompts.length];
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update background based on keywords
  useEffect(() => {
    if (!userInput) {
      setBackgroundGradient('');
      return;
    }

    const words = userInput.toLowerCase().split(' ');
    for (const word of words) {
      for (const keyword in keywordBackgrounds) {
        if (word.includes(keyword)) {
          setBackgroundGradient(keywordBackgrounds[keyword]);
          // Generate related visual elements
          setRelatedImages(getRelatedImages(keyword));
          return;
        }
      }
    }
    // Default gradient if no keyword matches
    setBackgroundGradient('linear-gradient(135deg, var(--luxury-pearl) 0%, var(--luxury-silk) 100%)');
  }, [userInput]);

  const getRelatedImages = (keyword: string): string[] => {
    // Placeholder for visual elements based on keywords
    const imageMap: { [key: string]: string[] } = {
      '봄': ['🌸', '🌷', '🌺'],
      '여름': ['☀️', '🌊', '🏖️'],
      '가을': ['🍂', '🍁', '🌰'],
      '겨울': ['❄️', '⛄', '🎄'],
      '바다': ['🌊', '🐚', '⛵'],
      '숲': ['🌲', '🍃', '🦌'],
      '도시': ['🏙️', '🌃', '🚕'],
      '사랑': ['💝', '🌹', '💕'],
      '기억': ['📷', '📖', '🕰️'],
      '꿈': ['✨', '🌙', '💫'],
      '아침': ['🌅', '☕', '🥐'],
      '밤': ['🌙', '⭐', '🌌'],
    };
    return imageMap[keyword] || ['✨'];
  };

  return (
    <section className="relative py-24 lg:py-32 overflow-hidden bg-[var(--luxury-pearl)]">
      {/* Dynamic Background */}
      <div
        className="absolute inset-0 transition-all duration-1000 opacity-20"
        style={{ background: backgroundGradient }}
      />

      {/* Floating Visual Elements */}
      {relatedImages.map((emoji, index) => (
        <div
          key={index}
          className="absolute text-6xl opacity-10 animate-float"
          style={{
            left: `${20 + index * 30}%`,
            top: `${20 + index * 20}%`,
            animationDelay: `${index * 0.5}s`
          }}
        >
          {emoji}
        </div>
      ))}

      <div className="relative mx-auto max-w-6xl px-4 lg:px-8">
        {/* Main Question */}
        <div className="text-center mb-12">
          <h2 className="text-4xl lg:text-6xl font-light text-[var(--luxury-midnight)] mb-6 font-[var(--font-display)] tracking-wide animate-fadeInUp">
            {currentPrompt}
          </h2>
          <p className="text-lg text-[var(--luxury-charcoal)] max-w-3xl mx-auto leading-relaxed">
            단어가 아닌 감정으로, 설명이 아닌 기억으로 들려주세요.
            AI 조향사가 당신의 이야기를 세상에서 가장 개인적인 향수로 빚어냅니다.
          </p>
        </div>

        {/* Interactive Input Area */}
        <div className="max-w-4xl mx-auto">
          <div className="relative group">
            {/* Glowing border effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)] rounded-lg blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>

            {/* Input Field */}
            <div className="relative bg-white/90 backdrop-blur-sm rounded-lg p-8 shadow-2xl">
              <textarea
                value={userInput}
                onChange={(e) => {
                  setUserInput(e.target.value);
                  setIsTyping(true);
                  setTimeout(() => setIsTyping(false), 500);
                }}
                placeholder="이곳에 당신의 이야기를 담아주세요..."
                className="w-full h-32 lg:h-40 p-4 text-lg text-[var(--luxury-midnight)] bg-transparent border-none outline-none resize-none placeholder:text-[var(--luxury-stone)] placeholder:opacity-50"
                style={{ fontFamily: 'var(--font-body)' }}
              />

              {/* Dynamic Keywords Display */}
              {userInput && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {userInput.split(' ').slice(0, 5).map((word, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 text-sm bg-[var(--luxury-gold)]/10 text-[var(--luxury-midnight)] rounded-full animate-fadeInUp"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      {word}
                    </span>
                  ))}
                </div>
              )}

              {/* CTA Buttons */}
              <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  href={{
                    pathname: '/create',
                    query: { inspiration: userInput }
                  }}
                  className="group relative px-8 py-4 text-center text-[var(--luxury-cream)] bg-[var(--luxury-midnight)] overflow-hidden transition-all duration-300 hover:shadow-xl"
                >
                  <span className="relative z-10">나만의 향수 제작 시작하기</span>
                  <div className="absolute inset-0 bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)] transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left"></div>
                </Link>

                <button className="px-8 py-4 text-[var(--luxury-midnight)] border-2 border-[var(--luxury-midnight)] hover:bg-[var(--luxury-midnight)] hover:text-[var(--luxury-cream)] transition-all duration-300">
                  영감 더 찾아보기
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Process Preview */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {[
            { step: '01', title: '이야기', desc: '당신의 기억과 감정을 들려주세요' },
            { step: '02', title: '조합', desc: 'AI가 최적의 향료 조합을 찾습니다' },
            { step: '03', title: '탄생', desc: '세상에 하나뿐인 향수가 완성됩니다' }
          ].map((item, index) => (
            <div
              key={item.step}
              className="text-center opacity-0 animate-fadeInUp"
              style={{ animationDelay: `${1000 + index * 200}ms`, animationFillMode: 'forwards' }}
            >
              <div className="text-3xl font-light text-[var(--luxury-gold)] mb-2">{item.step}</div>
              <h3 className="text-lg font-medium text-[var(--luxury-midnight)] mb-2">{item.title}</h3>
              <p className="text-sm text-[var(--luxury-charcoal)]">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}