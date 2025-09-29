'use client';

import { useState, useEffect } from 'react';
import { AIPerfumerSystem, type CreatedFragrance } from '../lib/ai-perfumer';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

const memoryPrompts = [
  "당신의 기억은 어떤 향기인가요?",
  "첫사랑의 순간을 향으로 표현한다면?",
  "가장 행복했던 날의 공기는?",
  "고요한 새벽의 숨결을 담아보세요",
  "당신만의 비밀스러운 정원은?"
];

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
  '우주': 'linear-gradient(135deg, #000428 0%, #004e92 100%)',
  '다락방': 'linear-gradient(135deg, #8B7355 0%, #D2B48C 100%)',
  '첫눈': 'linear-gradient(135deg, #FFFFFF 0%, #E0E0E0 100%)',
  '음악': 'linear-gradient(135deg, #4B0082 0%, #8A2BE2 100%)',
  '지하철': 'linear-gradient(135deg, #424242 0%, #616161 100%)',
};

export default function FragranceCreatorExpanded() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [userInput, setUserInput] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState(memoryPrompts[0]);
  const [backgroundGradient, setBackgroundGradient] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: '안녕하세요. 들숨의 AI 조향사입니다. 당신의 가장 개인적인 기억, 감정, 혹은 상상을 들려주세요. "다락방의 냄새", "첫눈이 올 때의 설렘", "베토벤의 월광 소나타 3악장", "퇴근길 지하철의 고독함"... 어떤 이야기든 향으로 번역해드릴 수 있습니다.'
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [generatedFragrance, setGeneratedFragrance] = useState<CreatedFragrance | null>(null);
  const [conversationContext, setConversationContext] = useState<string[]>([]);
  const [aiSystem] = useState(() => new AIPerfumerSystem());

  // Rotate prompts
  useEffect(() => {
    if (!isExpanded) {
      const interval = setInterval(() => {
        setCurrentPrompt(prev => {
          const currentIndex = memoryPrompts.indexOf(prev);
          return memoryPrompts[(currentIndex + 1) % memoryPrompts.length];
        });
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [isExpanded]);

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
          return;
        }
      }
    }
    setBackgroundGradient('linear-gradient(135deg, var(--luxury-pearl) 0%, var(--luxury-silk) 100%)');
  }, [userInput]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const message = userInput.trim();
    const userMessage = { role: 'user' as const, content: message };
    setChatMessages(prev => [...prev, userMessage]);

    const newContext = [...conversationContext, message];
    setConversationContext(newContext);
    setUserInput('');
    setIsLoading(true);

    try {
      // 실제 API 호출
      const response = await fetch('/api/ai-perfumer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          context: conversationContext
        })
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();

      // AI 응답 추가
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response
      }]);

      // 향수가 생성된 경우 표시
      if (data.fragrance) {
        setGeneratedFragrance(data.fragrance);
        setChatMessages(prev => [...prev, {
          role: 'assistant',
          content: `✨ 당신만의 향수 "${data.fragrance.korean_name}"가 완성되었습니다.\n\n${data.fragrance.story}`
        }]);
      }
    } catch (error) {
      console.error('API call failed, falling back to local AI:', error);

      // 실패 시 로컬 AI 시스템 사용 (폴백)
      const aiResponse = aiSystem.generateConversationalResponse(message, conversationContext);

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: aiResponse
      }]);

      // 3-4번의 대화 후 향수 생성
      if (newContext.length >= 3) {
        setTimeout(() => {
          generateFragrance();
        }, 1500);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const generateFragrance = () => {
    setIsLoading(true);

    // 모든 대화 내용을 결합
    const fullContext = conversationContext.join(' ');

    // AI 시스템으로 향수 생성
    const attributes = aiSystem.deconstructInput(fullContext);
    const expandedAttributes = aiSystem.expandSensoryAssociations(attributes);
    const composition = aiSystem.translateToFragrance(expandedAttributes);
    const fragrance = aiSystem.createFragranceStory(fullContext, composition);

    setTimeout(() => {
      setGeneratedFragrance(fragrance);

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `✨ 당신만의 향수 "${fragrance.korean_name}"가 완성되었습니다.\n\n${fragrance.story}`
      }]);

      setIsLoading(false);
    }, 2000);
  };

  const resetCreation = () => {
    setGeneratedFragrance(null);
    setChatMessages([{
      role: 'assistant',
      content: '안녕하세요. 들숨의 AI 조향사입니다. 당신의 가장 개인적인 기억, 감정, 혹은 상상을 들려주세요. "다락방의 냄새", "첫눈이 올 때의 설렘", "베토벤의 월광 소나타 3악장", "퇴근길 지하철의 고독함"... 어떤 이야기든 향으로 번역해드릴 수 있습니다.'
    }]);
    setUserInput('');
    setConversationContext([]);
    setIsExpanded(false);
  };

  if (!isExpanded) {
    // Compact view
    return (
      <section id="ai-creator" className="relative py-24 lg:py-32 overflow-hidden bg-[var(--luxury-pearl)]">
        <div
          className="absolute inset-0 transition-all duration-1000 opacity-20"
          style={{ background: backgroundGradient }}
        />

        <div className="relative mx-auto max-w-6xl px-4 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-4xl lg:text-6xl font-light text-[var(--luxury-midnight)] mb-6 font-[var(--font-display)] tracking-wide animate-fadeInUp">
              {currentPrompt}
            </h2>
            <p className="text-lg text-[var(--luxury-charcoal)] max-w-3xl mx-auto leading-relaxed">
              단어가 아닌 감정으로, 설명이 아닌 기억으로 들려주세요.
              AI 조향사가 당신의 이야기를 세상에서 가장 개인적인 향수로 빚어냅니다.
            </p>
          </div>

          <div className="max-w-4xl mx-auto">
            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)] rounded-lg blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>

              <div className="relative bg-white/90 backdrop-blur-sm rounded-lg p-8 shadow-2xl">
                <textarea
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder='"다락방의 냄새", "첫눈이 올 때의 설렘", "베토벤의 월광 소나타 3악장"... 당신만의 이야기를 담아주세요.'
                  className="w-full h-32 lg:h-40 p-4 text-lg text-[var(--luxury-midnight)] bg-transparent border-none outline-none resize-none placeholder:text-[var(--luxury-stone)] placeholder:opacity-50"
                  style={{ fontFamily: 'var(--font-body)' }}
                />

                <div className="mt-8 flex justify-center">
                  <button
                    onClick={() => setIsExpanded(true)}
                    className="group relative px-8 py-4 text-[var(--luxury-cream)] bg-[var(--luxury-midnight)] overflow-hidden transition-all duration-300 hover:shadow-xl"
                  >
                    <span className="relative z-10">AI와 대화 시작하기</span>
                    <div className="absolute inset-0 bg-gradient-to-r from-[var(--luxury-gold)] to-[var(--luxury-rose-gold)] transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500 origin-left"></div>
                  </button>
                </div>
              </div>
            </div>

            {/* 예시 카드 */}
            <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto">
              {[
                { text: '다락방의 냄새', icon: '🏠' },
                { text: '첫눈의 설렘', icon: '❄️' },
                { text: '월광 소나타', icon: '🎹' },
                { text: '지하철의 고독', icon: '🚇' }
              ].map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setUserInput(example.text);
                    setIsExpanded(true);
                  }}
                  className="p-3 bg-white/50 backdrop-blur-sm rounded-lg hover:bg-white/70 transition-all text-sm text-[var(--luxury-charcoal)] border border-[var(--luxury-silk)]"
                >
                  <span className="text-2xl mb-2 block">{example.icon}</span>
                  {example.text}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>
    );
  }

  // Expanded view - Full chat interface
  return (
    <section id="ai-creator" className="relative py-24 lg:py-32 bg-[var(--luxury-pearl)]">
      <div className="mx-auto max-w-6xl px-4 lg:px-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl lg:text-4xl font-light text-[var(--luxury-midnight)] mb-4 font-[var(--font-display)]">
            AI 조향사와의 대화
          </h2>
          <p className="text-lg text-[var(--luxury-charcoal)]">
            당신의 이야기를 들려주세요. 향으로 기록하겠습니다.
          </p>
        </div>

        {!generatedFragrance ? (
          <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden">
            {/* Chat Messages */}
            <div className="h-96 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-white to-[var(--luxury-pearl)]">
              {chatMessages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                      message.role === 'user'
                        ? 'bg-[var(--luxury-midnight)] text-[var(--luxury-pearl)]'
                        : 'bg-white border border-[var(--luxury-silk)] text-[var(--luxury-charcoal)]'
                    }`}
                  >
                    <div className="text-xs mb-1 opacity-70">
                      {message.role === 'user' ? '나' : 'AI 조향사'}
                    </div>
                    <div className="text-sm whitespace-pre-line">{message.content}</div>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-[var(--luxury-silk)] px-4 py-3 rounded-lg">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-[var(--luxury-gold)] rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-[var(--luxury-gold)] rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-[var(--luxury-gold)] rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Form */}
            <form onSubmit={handleSubmit} className="p-6 border-t border-[var(--luxury-silk)] bg-white">
              <div className="flex space-x-4">
                <input
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder="당신의 이야기를 들려주세요..."
                  className="flex-1 px-4 py-3 border border-[var(--luxury-silk)] rounded-lg focus:ring-2 focus:ring-[var(--luxury-gold)] focus:border-transparent text-[var(--luxury-midnight)] placeholder:text-[var(--luxury-stone)]"
                  disabled={isLoading || !!generatedFragrance}
                  autoFocus
                />
                <button
                  type="submit"
                  disabled={!userInput.trim() || isLoading || !!generatedFragrance}
                  className="px-6 py-3 bg-[var(--luxury-midnight)] text-[var(--luxury-pearl)] rounded-lg hover:bg-[var(--luxury-charcoal)] disabled:opacity-50"
                >
                  전송
                </button>
              </div>
            </form>
          </div>
        ) : (
          // Generated Fragrance Display - Enhanced
          <div className="max-w-5xl mx-auto bg-white rounded-xl shadow-lg p-8">
            {/* Header */}
            <div className="text-center mb-8">
              <div className="text-6xl mb-4">✨</div>
              <h3 className="text-3xl font-light text-[var(--luxury-midnight)] mb-2 font-[var(--font-display)]">
                {generatedFragrance.korean_name}
              </h3>
              <p className="text-lg text-[var(--luxury-charcoal)] italic">
                {generatedFragrance.name}
              </p>
            </div>

            {/* Story */}
            <div className="bg-gradient-to-r from-[var(--luxury-pearl)] to-[var(--luxury-silk)] rounded-lg p-6 mb-8">
              <p className="text-[var(--luxury-charcoal)] text-center leading-relaxed text-lg">
                {generatedFragrance.story}
              </p>
            </div>

            {/* Composition Details */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Top Notes */}
              <div className="bg-[var(--luxury-pearl)] rounded-lg p-6">
                <h4 className="font-medium text-[var(--luxury-midnight)] mb-4 text-center">
                  탑 노트 (첫인상)
                </h4>
                <div className="space-y-3">
                  {generatedFragrance.composition.top_notes.map((note, idx) => (
                    <div key={idx} className="border-l-2 border-[var(--luxury-gold)] pl-3">
                      <div className="font-medium text-sm text-[var(--luxury-midnight)]">
                        {note.name}
                      </div>
                      <div className="text-xs text-[var(--luxury-charcoal)] mt-1">
                        {note.description}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Heart Notes */}
              <div className="bg-[var(--luxury-pearl)] rounded-lg p-6">
                <h4 className="font-medium text-[var(--luxury-midnight)] mb-4 text-center">
                  하트 노트 (중심)
                </h4>
                <div className="space-y-3">
                  {generatedFragrance.composition.heart_notes.map((note, idx) => (
                    <div key={idx} className="border-l-2 border-[var(--luxury-rose-gold)] pl-3">
                      <div className="font-medium text-sm text-[var(--luxury-midnight)]">
                        {note.name}
                      </div>
                      <div className="text-xs text-[var(--luxury-charcoal)] mt-1">
                        {note.description}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Base Notes */}
              <div className="bg-[var(--luxury-pearl)] rounded-lg p-6">
                <h4 className="font-medium text-[var(--luxury-midnight)] mb-4 text-center">
                  베이스 노트 (잔향)
                </h4>
                <div className="space-y-3">
                  {generatedFragrance.composition.base_notes.map((note, idx) => (
                    <div key={idx} className="border-l-2 border-[var(--luxury-charcoal)] pl-3">
                      <div className="font-medium text-sm text-[var(--luxury-midnight)]">
                        {note.name}
                      </div>
                      <div className="text-xs text-[var(--luxury-charcoal)] mt-1">
                        {note.description}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Characteristics */}
            <div className="bg-gradient-to-r from-white to-[var(--luxury-pearl)] rounded-lg p-6 mb-8">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                <div>
                  <div className="text-xs text-[var(--luxury-charcoal)] mb-1">강도</div>
                  <div className="font-medium text-[var(--luxury-midnight)]">
                    {generatedFragrance.characteristics.intensity}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-[var(--luxury-charcoal)] mb-1">계절</div>
                  <div className="font-medium text-[var(--luxury-midnight)]">
                    {generatedFragrance.characteristics.season}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-[var(--luxury-charcoal)] mb-1">성별</div>
                  <div className="font-medium text-[var(--luxury-midnight)]">
                    {generatedFragrance.characteristics.gender}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-[var(--luxury-charcoal)] mb-1">키워드</div>
                  <div className="font-medium text-[var(--luxury-midnight)]">
                    {generatedFragrance.characteristics.keywords.join(', ')}
                  </div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-center space-x-4">
              <button
                onClick={resetCreation}
                className="px-6 py-3 border border-[var(--luxury-midnight)] text-[var(--luxury-midnight)] rounded-lg hover:bg-[var(--luxury-midnight)] hover:text-[var(--luxury-pearl)] transition-all"
              >
                다시 만들기
              </button>
              <button className="px-6 py-3 bg-[var(--luxury-gold)] text-[var(--luxury-midnight)] rounded-lg hover:bg-[var(--luxury-gold)]/90 transition-all">
                사전 예약하기
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}