'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  fragranceData?: {
    name: string;
    notes: {
      top: string[];
      middle: string[];
      base: string[];
    };
    description: string;
    intensity: string;
    season: string;
  };
}

export default function AIFragranceChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateFragrance = async (description: string) => {
    // 입력에 따른 향수 생성 로직
    const keywords = description.toLowerCase();

    let topNotes = [];
    let middleNotes = [];
    let baseNotes = [];
    let intensity = '중간';
    let season = '사계절';

    // 키워드 분석
    if (keywords.includes('상쾌') || keywords.includes('시원') || keywords.includes('fresh')) {
      topNotes = ['베르가못', '레몬', '그레이프프루트'];
      intensity = '가벼움';
    } else if (keywords.includes('달콤') || keywords.includes('sweet')) {
      topNotes = ['복숭아', '배', '블랙커런트'];
      intensity = '중간';
    } else {
      topNotes = ['핑크페퍼', '카다몬', '생강'];
      intensity = '강함';
    }

    if (keywords.includes('꽃') || keywords.includes('플로럴') || keywords.includes('floral')) {
      middleNotes = ['장미', '자스민', '일랑일랑'];
    } else if (keywords.includes('우디') || keywords.includes('나무')) {
      middleNotes = ['시더우드', '샌달우드', '베티버'];
    } else {
      middleNotes = ['라벤더', '제라늄', '네롤리'];
    }

    if (keywords.includes('따뜻') || keywords.includes('warm')) {
      baseNotes = ['앰버', '바닐라', '머스크'];
      season = '가을/겨울';
    } else if (keywords.includes('가벼운') || keywords.includes('light')) {
      baseNotes = ['화이트머스크', '앰브록산', '시더우드'];
      season = '봄/여름';
    } else {
      baseNotes = ['파출리', '톤카빈', '벤조인'];
      season = '사계절';
    }

    return {
      name: `${description.split(' ')[0]} 에센스`,
      notes: {
        top: topNotes,
        middle: middleNotes,
        base: baseNotes
      },
      description: `"${description}"의 느낌을 완벽하게 구현한 맞춤 향수`,
      intensity,
      season
    };
  };

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // AI 응답 시뮬레이션
    await new Promise(resolve => setTimeout(resolve, 1500));

    const fragranceData = await generateFragrance(input);

    const assistantMessage: Message = {
      role: 'assistant',
      content: `"${input}"를 바탕으로 향수를 제작했습니다.

당신이 원하시는 향의 특성을 분석하여, 최대한 비슷하게 구현했습니다.`,
      fragranceData
    };

    setMessages(prev => [...prev, assistantMessage]);
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <section className="py-16 lg:py-20" style={{backgroundColor: 'var(--ivory-light)'}}>
      <div className="mx-auto max-w-4xl px-4 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-light text-neutral-900 mb-3">AI 향수 제작 어시스턴트</h2>
          <p className="text-lg text-neutral-600 mb-4">원하는 향을 자유롭게 설명해주세요</p>
          <a
            href="/create"
            className="inline-block px-6 py-2 text-sm font-medium text-white rounded-md transition-colors"
            style={{backgroundColor: 'var(--light-brown)'}}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown-dark)'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown)'}
          >
            가이드 모드로 만들기 →
          </a>
        </div>

        {/* 채팅 영역 */}
        <div className="bg-white rounded-xl shadow-lg border border-neutral-200 mb-6" style={{minHeight: '400px'}}>
          <div className="p-6 space-y-4 max-h-96 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="text-center py-12 text-neutral-400">
                <p className="mb-4">어떤 향수를 찾고 계신가요?</p>
                <div className="space-y-2 text-sm">
                  <p>"봄날 아침의 상쾌한 꽃향기"</p>
                  <p>"따뜻하고 포근한 겨울 저녁"</p>
                  <p>"신비로운 동양의 향신료"</p>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-3xl ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                    <div className={`inline-block px-4 py-2 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-neutral-900 text-white'
                        : 'bg-neutral-100 text-neutral-900'
                    }`}>
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>

                    {message.fragranceData && (
                      <div className="mt-4 p-6 bg-white border border-neutral-200 rounded-lg">
                        <h3 className="text-xl font-medium text-neutral-900 mb-4">
                          {message.fragranceData.name}
                        </h3>

                        <p className="text-neutral-600 mb-6">{message.fragranceData.description}</p>

                        <div className="grid grid-cols-3 gap-4 mb-4">
                          <div>
                            <h4 className="font-medium text-neutral-900 mb-2">탑 노트</h4>
                            <ul className="text-sm text-neutral-600 space-y-1">
                              {message.fragranceData.notes.top.map((note, i) => (
                                <li key={i}>• {note}</li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <h4 className="font-medium text-neutral-900 mb-2">미들 노트</h4>
                            <ul className="text-sm text-neutral-600 space-y-1">
                              {message.fragranceData.notes.middle.map((note, i) => (
                                <li key={i}>• {note}</li>
                              ))}
                            </ul>
                          </div>
                          <div>
                            <h4 className="font-medium text-neutral-900 mb-2">베이스 노트</h4>
                            <ul className="text-sm text-neutral-600 space-y-1">
                              {message.fragranceData.notes.base.map((note, i) => (
                                <li key={i}>• {note}</li>
                              ))}
                            </ul>
                          </div>
                        </div>

                        <div className="flex justify-between text-sm text-neutral-600 pt-4 border-t border-neutral-200">
                          <span>강도: {message.fragranceData.intensity}</span>
                          <span>추천 시즌: {message.fragranceData.season}</span>
                        </div>

                        <button
                          className="mt-4 w-full px-6 py-3 text-white font-medium rounded-md transition-colors"
                          style={{backgroundColor: 'var(--light-brown)'}}
                          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown-dark)'}
                          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown)'}
                        >
                          이 향수 주문하기
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-neutral-100 text-neutral-900 px-4 py-2 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* 입력창 */}
        <div className="relative flex items-center">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="원하는 향을 자유롭게 설명해주세요... (예: 비 온 후 장미 정원의 상쾌한 향기)"
            className="w-full px-4 py-3 pr-12 text-lg text-neutral-900 placeholder-neutral-400 bg-white border border-neutral-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent"
            rows={1}
            style={{
              minHeight: '56px',
              maxHeight: '120px',
              overflowY: 'auto'
            }}
            disabled={isLoading}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading}
            className="absolute right-3 p-2 text-neutral-600 hover:text-neutral-900 disabled:text-neutral-300 disabled:cursor-not-allowed transition-colors"
            aria-label="전송"
          >
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}