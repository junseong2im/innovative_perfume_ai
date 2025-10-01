'use client';

import { useState, useRef, useEffect } from 'react';

// Error types for different failure scenarios
type ErrorType = 'CONNECTION' | 'SERVER_ERROR' | 'INVALID_RESPONSE' | 'TIMEOUT' | 'UNKNOWN';

interface ErrorDetails {
  type: ErrorType;
  message: string;
  userMessage: string;
  retryable: boolean;
  statusCode?: number;
}

interface Message {
  role: 'user' | 'assistant' | 'error';
  content: string;
  error?: ErrorDetails;
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
  timestamp?: Date;
}

// Error message component
const ErrorMessage: React.FC<{ error: ErrorDetails; onRetry?: () => void }> = ({ error, onRetry }) => {
  const getErrorIcon = () => {
    switch (error.type) {
      case 'CONNECTION':
        return '🔌';
      case 'SERVER_ERROR':
        return '⚠️';
      case 'TIMEOUT':
        return '⏰';
      default:
        return '❌';
    }
  };

  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-start">
        <span className="text-2xl mr-3">{getErrorIcon()}</span>
        <div className="flex-1">
          <h4 className="font-medium text-red-900 mb-1">
            {error.type === 'CONNECTION' && '연결 오류'}
            {error.type === 'SERVER_ERROR' && '서버 오류'}
            {error.type === 'TIMEOUT' && '응답 시간 초과'}
            {error.type === 'INVALID_RESPONSE' && '응답 오류'}
            {error.type === 'UNKNOWN' && '알 수 없는 오류'}
          </h4>
          <p className="text-red-700 text-sm mb-3">{error.userMessage}</p>

          {error.retryable && onRetry && (
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors"
            >
              다시 시도
            </button>
          )}

          {!error.retryable && (
            <div className="mt-3 p-3 bg-white rounded border border-red-100">
              <p className="text-sm text-gray-600 mb-2">다른 방법을 시도해보세요:</p>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• <a href="/create" className="text-blue-600 hover:underline">가이드 모드로 향수 만들기</a></li>
                <li>• <a href="/products" className="text-blue-600 hover:underline">기존 제품 둘러보기</a></li>
                <li>• 잠시 후 다시 시도해주세요</li>
              </ul>
            </div>
          )}
        </div>
      </div>

      {process.env.NODE_ENV === 'development' && (
        <details className="mt-3 text-xs text-gray-500">
          <summary className="cursor-pointer">개발자 정보</summary>
          <pre className="mt-2 p-2 bg-gray-100 rounded overflow-x-auto">
            {JSON.stringify({ type: error.type, statusCode: error.statusCode, message: error.message }, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
};

// Connection status indicator
const ConnectionStatus: React.FC<{ isOnline: boolean }> = ({ isOnline }) => {
  if (isOnline) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-yellow-100 border border-yellow-300 rounded-lg px-4 py-2 shadow-lg z-50">
      <div className="flex items-center">
        <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse mr-2"></span>
        <span className="text-sm text-yellow-800">연결 상태를 확인하고 있습니다...</span>
      </div>
    </div>
  );
};

export default function AIFragranceChatEnhanced() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Monitor connection status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const classifyError = (error: any): ErrorDetails => {
    // Network errors
    if (error.message?.includes('fetch') || error.message?.includes('network') || !isOnline) {
      return {
        type: 'CONNECTION',
        message: error.message,
        userMessage: 'AI 서버와 연결할 수 없습니다. 네트워크 연결을 확인해주세요.',
        retryable: true
      };
    }

    // Timeout errors
    if (error.message?.includes('timeout')) {
      return {
        type: 'TIMEOUT',
        message: error.message,
        userMessage: 'AI 서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.',
        retryable: true
      };
    }

    // API response errors
    if (error.message?.includes('API Error')) {
      const statusCode = parseInt(error.message.match(/\d+/)?.[0] || '0');

      if (statusCode >= 500) {
        return {
          type: 'SERVER_ERROR',
          message: error.message,
          userMessage: 'AI 서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
          retryable: true,
          statusCode
        };
      }

      if (statusCode >= 400 && statusCode < 500) {
        return {
          type: 'INVALID_RESPONSE',
          message: error.message,
          userMessage: '요청을 처리할 수 없습니다. 다른 표현으로 시도해보세요.',
          retryable: false,
          statusCode
        };
      }
    }

    // Invalid response format
    if (error.message?.includes('Invalid API response')) {
      return {
        type: 'INVALID_RESPONSE',
        message: error.message,
        userMessage: 'AI 응답을 처리하는 중 문제가 발생했습니다. 다시 시도해주세요.',
        retryable: true
      };
    }

    // Default unknown error
    return {
      type: 'UNKNOWN',
      message: error.message || 'Unknown error',
      userMessage: '예기치 않은 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
      retryable: true
    };
  };

  const generateFragrance = async (description: string, attemptNumber: number = 1): Promise<any> => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
    const MAX_TIMEOUT = 30000; // 30 seconds

    // Create timeout promise
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('timeout')), MAX_TIMEOUT);
    });

    try {
      // Race between fetch and timeout
      const response = await Promise.race([
        fetch(`${API_URL}/api/v1/generate/recipe`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Request-ID': `${Date.now()}-${attemptNumber}`,
            'X-Client-Version': '2.0.0'
          },
          body: JSON.stringify({
            description: description,
            request_metadata: {
              attempt: attemptNumber,
              timestamp: new Date().toISOString(),
              client: 'web-enhanced'
            }
          }),
        }),
        timeoutPromise
      ]) as Response;

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      // Validate response structure
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid API response: Empty or invalid data');
      }

      if (!data.recipe || !data.recipe.name) {
        throw new Error('Invalid API response: Missing recipe data');
      }

      // Validate required fields
      const requiredFields = ['name', 'description', 'composition', 'characteristics'];
      const missingFields = requiredFields.filter(field => !data.recipe[field]);

      if (missingFields.length > 0) {
        throw new Error(`Invalid API response: Missing fields: ${missingFields.join(', ')}`);
      }

      return {
        name: data.recipe.name,
        notes: {
          top: data.recipe.composition.top_notes?.map((note: any) => note.name) || [],
          middle: data.recipe.composition.heart_notes?.map((note: any) => note.name) || [],
          base: data.recipe.composition.base_notes?.map((note: any) => note.name) || []
        },
        description: data.recipe.description,
        intensity: data.recipe.characteristics.intensity || '보통',
        season: data.recipe.characteristics.season || '사계절'
      };
    } catch (error) {
      console.error(`AI 향수 생성 실패 (시도 ${attemptNumber}):`, error);
      throw error;
    }
  };

  const handleSubmit = async (retryLastMessage: boolean = false) => {
    const messageText = retryLastMessage ?
      messages.find(m => m.role === 'user' && m.content)?.content || input :
      input;

    if (!messageText.trim() || isLoading) return;

    if (!retryLastMessage) {
      const userMessage: Message = {
        role: 'user',
        content: messageText,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, userMessage]);
      setInput('');
    }

    setIsLoading(true);
    setRetryCount(prev => retryLastMessage ? prev + 1 : 0);

    try {
      const fragranceData = await generateFragrance(messageText, retryCount + 1);

      const assistantMessage: Message = {
        role: 'assistant',
        content: `"${messageText}"를 바탕으로 향수를 제작했습니다.

당신이 원하시는 향의 특성을 분석하여, AI가 최적의 조합을 찾았습니다.`,
        fragranceData,
        timestamp: new Date()
      };

      setMessages(prev => {
        // Remove any previous error messages for retry
        const filtered = retryLastMessage ?
          prev.filter(m => m.role !== 'error') :
          prev;
        return [...filtered, assistantMessage];
      });

      setRetryCount(0);
    } catch (error: any) {
      const errorDetails = classifyError(error);

      const errorMessage: Message = {
        role: 'error',
        content: '',
        error: errorDetails,
        timestamp: new Date()
      };

      setMessages(prev => {
        // Remove any previous error messages for retry
        const filtered = retryLastMessage ?
          prev.filter(m => m.role !== 'error') :
          prev;
        return [...filtered, errorMessage];
      });
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

  const handleRetry = () => {
    handleSubmit(true);
  };

  return (
    <section className="py-16 lg:py-20" style={{backgroundColor: 'var(--ivory-light)'}}>
      <div className="mx-auto max-w-4xl px-4 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-light text-neutral-900 mb-3">AI 향수 제작 어시스턴트</h2>
          <p className="text-lg text-neutral-600 mb-4">원하는 향을 자유롭게 설명해주세요</p>

          {/* Alternative options always visible */}
          <div className="flex justify-center gap-4">
            <a
              href="/create"
              className="inline-block px-6 py-2 text-sm font-medium text-white rounded-md transition-colors"
              style={{backgroundColor: 'var(--light-brown)'}}
              onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown-dark)'}
              onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--light-brown)'}
            >
              가이드 모드로 만들기 →
            </a>
            <a
              href="/products"
              className="inline-block px-6 py-2 text-sm font-medium text-neutral-700 bg-white border border-neutral-300 rounded-md transition-colors hover:bg-neutral-50"
            >
              제품 둘러보기 →
            </a>
          </div>
        </div>

        {/* Chat area */}
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
                    {message.role === 'error' && message.error ? (
                      <ErrorMessage error={message.error} onRetry={handleRetry} />
                    ) : (
                      <>
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
                      </>
                    )}
                  </div>
                </div>
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-neutral-100 text-neutral-900 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                      <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                      <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                    </div>
                    <span className="text-sm text-neutral-500">AI가 향수를 제작 중입니다...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="relative flex items-center">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isOnline ? "원하는 향을 자유롭게 설명해주세요... (예: 비 온 후 장미 정원의 상쾌한 향기)" : "네트워크 연결을 확인해주세요..."}
            className="w-full px-4 py-3 pr-12 text-lg text-neutral-900 placeholder-neutral-400 bg-white border border-neutral-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent"
            rows={1}
            style={{
              minHeight: '56px',
              maxHeight: '120px',
              overflowY: 'auto'
            }}
            disabled={isLoading || !isOnline}
          />
          <button
            onClick={() => handleSubmit()}
            disabled={!input.trim() || isLoading || !isOnline}
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

      {/* Connection status indicator */}
      <ConnectionStatus isOnline={isOnline} />
    </section>
  );
}