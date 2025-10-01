'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant' | 'error';
  content: string;
  errorType?: 'CONNECTION' | 'SERVER_ERROR' | 'TIMEOUT' | 'INVALID_RESPONSE' | 'UNKNOWN';
  retryable?: boolean;
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
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

    try {
      const response = await fetch(`${API_URL}/api/v1/generate/recipe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: description,
          // AI가 설명을 분석하여 모든 파라미터를 자동으로 결정하도록 함
          // 하드코딩된 기본값 제거
        }),
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();

      // API 응답을 그대로 사용 (fallback 제거)
      if (!data.recipe || !data.recipe.name) {
        throw new Error('Invalid API response');
      }

      return {
        name: data.recipe.name,
        notes: {
          top: data.recipe.composition.top_notes.map((note: any) => note.name),
          middle: data.recipe.composition.heart_notes.map((note: any) => note.name),
          base: data.recipe.composition.base_notes.map((note: any) => note.name)
        },
        description: data.recipe.description,
        intensity: data.recipe.characteristics.intensity,
        season: data.recipe.characteristics.season
      };
    } catch (error: any) {
      console.error('AI 향수 생성 API 호출 실패:', error);

      // API 호출 실패 시 에러 타입 분류
      const errorInfo = {
        type: 'UNKNOWN' as const,
        message: 'AI 서버와 연결할 수 없습니다.',
        retryable: true
      };

      // 네트워크 에러
      if (error.message?.includes('fetch') || error.message?.includes('network')) {
        errorInfo.type = 'CONNECTION' as const;
        errorInfo.message = '네트워크 연결을 확인해주세요.';
      }
      // 서버 에러 (5xx)
      else if (error.message?.includes('500') || error.message?.includes('502') || error.message?.includes('503')) {
        errorInfo.type = 'SERVER_ERROR' as const;
        errorInfo.message = 'AI 서버에 일시적인 문제가 발생했습니다.';
      }
      // 타임아웃
      else if (error.message?.includes('timeout')) {
        errorInfo.type = 'TIMEOUT' as const;
        errorInfo.message = '응답 시간이 초과되었습니다.';
      }
      // 잘못된 응답
      else if (error.message?.includes('Invalid API response')) {
        errorInfo.type = 'INVALID_RESPONSE' as const;
        errorInfo.message = 'AI 응답을 처리할 수 없습니다.';
        errorInfo.retryable = false;
      }

      // 구조화된 에러 던지기
      const enhancedError = new Error(errorInfo.message);
      (enhancedError as any).errorType = errorInfo.type;
      (enhancedError as any).retryable = errorInfo.retryable;
      throw enhancedError;
    }
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

    try {
      const fragranceData = await generateFragrance(input);

      const assistantMessage: Message = {
        role: 'assistant',
        content: `"${input}"를 바탕으로 향수를 제작했습니다.

당신이 원하시는 향의 특성을 분석하여, 최대한 비슷하게 구현했습니다.`,
        fragranceData
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      // 에러 발생 시 사용자에게 명확한 피드백
      const errorMessage: Message = {
        role: 'error',
        content: error.message || 'AI 서버와 연결하는 중 문제가 발생했습니다.',
        errorType: error.errorType || 'UNKNOWN',
        retryable: error.retryable !== false
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
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
                    {message.role === 'error' ? (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div className="flex items-start">
                          <svg className="w-5 h-5 text-red-600 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                          </svg>
                          <div className="flex-1">
                            <h4 className="text-red-900 font-medium mb-1">
                              {message.errorType === 'CONNECTION' && '연결 오류'}
                              {message.errorType === 'SERVER_ERROR' && '서버 오류'}
                              {message.errorType === 'TIMEOUT' && '응답 시간 초과'}
                              {message.errorType === 'INVALID_RESPONSE' && '응답 처리 오류'}
                              {(!message.errorType || message.errorType === 'UNKNOWN') && '오류 발생'}
                            </h4>
                            <p className="text-red-700 text-sm mb-3">{message.content}</p>
                            <div className="space-y-2">
                              {message.retryable !== false && (
                                <button
                                  onClick={() => handleSubmit()}
                                  disabled={isLoading}
                                  className="text-sm px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50"
                                >
                                  다시 시도
                                </button>
                              )}
                              <div className="text-sm text-gray-600">
                                <p>다른 방법을 시도해보세요:</p>
                                <a href="/create" className="text-blue-600 hover:underline">• 가이드 모드 이용하기</a>
                                <br />
                                <a href="/products" className="text-blue-600 hover:underline">• 기존 제품 둘러보기</a>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className={`inline-block px-4 py-2 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-neutral-900 text-white'
                          : 'bg-neutral-100 text-neutral-900'
                      }`}>
                        <p className="whitespace-pre-wrap">{message.content}</p>
                      </div>
                    )}

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