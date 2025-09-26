'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { AuthService } from 'lib/auth';
import { FragranceAI, type FragranceGenerationRequest, type FragranceGenerationResponse } from 'lib/fragrance-ai';
import LoginModal from 'components/auth/login-modal';

type CreationMode = 'story' | 'cards';
type CardStep = 'family' | 'mood' | 'intensity' | 'season' | 'region';

interface CardSelection {
  family?: string;
  mood?: string;
  intensity?: string;
  season?: string;
  region?: string;
  traditional?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export default function CreateFragrancePage() {
  const [mode, setMode] = useState<CreationMode>('cards');
  const [currentStep, setCurrentStep] = useState<CardStep>('family');
  const [cardSelections, setCardSelections] = useState<CardSelection>({});
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: '안녕하세요! 들숨의 AI 향수 아티스트입니다. 어떤 향수를 만들어드릴까요? 자유롭게 말씀해주세요.'
    }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [result, setResult] = useState<FragranceGenerationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [showLoginModal, setShowLoginModal] = useState(false);

  useEffect(() => {
    const savedUser = AuthService.getUser();
    setUser(savedUser);
  }, []);

  // 카드 선택 옵션들
  const cardOptions = {
    family: [
      { id: 'floral', title: '플로럴', emoji: '🌸', desc: '꽃향기의 우아함' },
      { id: 'citrus', title: '시트러스', emoji: '🍊', desc: '상큼한 과일향' },
      { id: 'woody', title: '우디', emoji: '🌳', desc: '따뜻한 나무향' },
      { id: 'oriental', title: '오리엔탈', emoji: '🌙', desc: '신비로운 향신료' },
      { id: 'fresh', title: '프레시', emoji: '🌿', desc: '깔끔한 자연향' }
    ],
    mood: [
      { id: 'romantic', title: '로맨틱', emoji: '💝', desc: '사랑스러운 분위기' },
      { id: 'fresh', title: '상쾌함', emoji: '✨', desc: '활기찬 에너지' },
      { id: 'elegant', title: '우아함', emoji: '👑', desc: '세련된 품격' },
      { id: 'mysterious', title: '신비로움', emoji: '🔮', desc: '독특한 매력' },
      { id: 'energetic', title: '활기참', emoji: '⚡', desc: '역동적인 느낌' }
    ],
    intensity: [
      { id: 'light', title: '라이트', emoji: '🪶', desc: '은은하고 부드럽게' },
      { id: 'moderate', title: '모더레이트', emoji: '🌸', desc: '적당히 향기롭게' },
      { id: 'strong', title: '스트롱', emoji: '💪', desc: '진하고 강렬하게' }
    ],
    season: [
      { id: 'spring', title: '봄', emoji: '🌸', desc: '생동감 넘치는 계절' },
      { id: 'summer', title: '여름', emoji: '☀️', desc: '뜨겁고 활기찬 계절' },
      { id: 'autumn', title: '가을', emoji: '🍂', desc: '따뜻하고 차분한 계절' },
      { id: 'winter', title: '겨울', emoji: '❄️', desc: '차가우면서 깊은 계절' }
    ],
    region: [
      { id: '제주', title: '제주', emoji: '🌊', desc: '바다와 감귤의 섬' },
      { id: '강원', title: '강원', emoji: '🏔️', desc: '설악산의 맑은 공기' },
      { id: '경상', title: '경상', emoji: '🏛️', desc: '경주의 고즈넉함' },
      { id: '전라', title: '전라', emoji: '🌾', desc: '전주의 한옥 정취' },
      { id: '충청', title: '충청', emoji: '🌸', desc: '계룡산의 자연미' }
    ]
  };

  const stepTitles = {
    family: '어떤 향족을 선호하시나요?',
    mood: '어떤 분위기를 원하시나요?',
    intensity: '향의 강도는 어느 정도로?',
    season: '어떤 계절에 사용하실 건가요?',
    region: '한국의 어느 지역에서 영감을 받을까요?'
  };

  const stepOrder: CardStep[] = ['family', 'mood', 'intensity', 'season', 'region'];

  const handleCardSelect = (value: string) => {
    const newSelections = { ...cardSelections, [currentStep]: value };
    setCardSelections(newSelections);

    // 다음 단계로
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1]);
    }
  };

  const goToPreviousStep = () => {
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(stepOrder[currentIndex - 1]);
    }
  };

  const generateFromCards = async () => {
    if (!cardSelections.family) return;

    setIsLoading(true);

    // 계절 선택에 따른 제대로 된 매핑
    const seasonMapping: { [key: string]: string } = {
      'spring': '봄',
      'summer': '여름',
      'autumn': '가을',
      'winter': '겨울'
    };

    const request: FragranceGenerationRequest = {
      fragrance_family: cardSelections.family,
      mood: cardSelections.mood || 'fresh',
      intensity: cardSelections.intensity || 'moderate',
      season: cardSelections.season ? seasonMapping[cardSelections.season] || cardSelections.season : undefined,
      korean_region: cardSelections.region,
      traditional_element: cardSelections.traditional
    };

    try {
      const response = await FragranceAI.generateFragrance(request, user?.token);
      setResult(response);
    } catch (error) {
      console.error('향수 생성 오류:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isLoading) return;

    const userInput = chatInput.trim();

    // 사용자 메시지 추가
    const userMessage: ChatMessage = { role: 'user', content: userInput };
    setChatMessages(prev => [...prev, userMessage]);

    setIsLoading(true);
    setChatInput('');

    try {
      // Agentic 채팅 API 호출
      const agenticResponse = await handleAgenticChat(userInput);

      // AI 응답 메시지 추가
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: agenticResponse.response
      }]);

      // 향수 생성이 필요한지 판단 (3번째 메시지 이후 또는 특정 키워드)
      const shouldGenerate = chatMessages.length >= 4 ||
                            userInput.includes('만들어') ||
                            userInput.includes('생성') ||
                            userInput.includes('완성');

      if (shouldGenerate) {
        // 대화 내용을 바탕으로 향수 생성
        await generateFragranceFromAgenticChat(userInput, agenticResponse);
      }

    } catch (error) {
      console.error('Agentic chat failed:', error);

      // 폴백: 기존 로직 사용
      const detailedResponse = generateDetailedResponse(userInput, chatMessages.length);

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: detailedResponse.message
      }]);

      if (detailedResponse.shouldGenerate) {
        generateFragranceFromChat(userInput, detailedResponse.extractedPreferences);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Agentic 채팅 API 호출
  const handleAgenticChat = async (userInput: string) => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (user?.token) {
      headers['Authorization'] = `Bearer ${user.token}`;
    }

    // 대화 컨텍스트 구성
    const conversationContext = chatMessages
      .filter(m => m.role === 'user')
      .map(m => m.content)
      .join('\n');

    const response = await fetch('http://localhost:8000/api/v2/agentic/chat', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        message: userInput,
        session_id: `chat_${Date.now()}`,
        context: {
          conversation_history: conversationContext,
          interaction_type: 'fragrance_chat',
          message_count: chatMessages.length
        }
      }),
    });

    if (!response.ok) {
      throw new Error(`Agentic chat failed: ${response.status}`);
    }

    return await response.json();
  };

  // Agentic 채팅에서 향수 생성
  const generateFragranceFromAgenticChat = async (userInput: string, agenticResponse: any) => {
    try {
      // 대화 내용을 바탕으로 향수 생성 요청 구성
      const fragmentRequest = {
        fragrance_family: 'floral', // 기본값, AI가 분석해서 결정
        mood: 'romantic',
        intensity: 'moderate',
        unique_request: userInput,
        conversation_context: chatMessages.filter(m => m.role === 'user').map(m => m.content).join('\n')
      };

      const fragranceResponse = await FragranceAI.generateFragrance(fragmentRequest, user?.token);
      setResult(fragranceResponse);

      // 향수 생성 완료 메시지 추가
      const completionMessage: ChatMessage = {
        role: 'assistant',
        content: `🌸 **"${fragranceResponse.customer_info.name}"** 완성!\n\n${fragranceResponse.customer_info.description}\n\n"${fragranceResponse.customer_info.story}"\n\n이 향수가 마음에 드시나요? 혹시 조정하고 싶은 부분이 있다면 언제든 말씀해주세요!`
      };

      setChatMessages(prev => [...prev, completionMessage]);

    } catch (error) {
      console.error('Agentic fragrance generation failed:', error);

      // 폴백: 기존 방식으로 향수 생성
      const preferences = extractPreferencesFromConversation(chatMessages, userInput);
      await generateFragranceFromChat(userInput, preferences);
    }
  };

  const generateDetailedResponse = (userInput: string, messageCount: number) => {
    const responses = [
      // 첫 번째 메시지
      {
        condition: () => messageCount <= 2,
        message: `"${userInput}"... 정말 아름다운 표현이네요! 더 구체적으로 말씀해주시면 어떨까요? 예를 들어:\n\n• 어떤 계절에 주로 사용하고 싶으신가요?\n• 특별한 추억이나 장소가 있나요?\n• 선호하는 향의 강도는 어느 정도인가요?\n\n천천히 이야기해주세요.`,
        shouldGenerate: false,
        extractedPreferences: {}
      },
      // 두 번째 메시지
      {
        condition: () => messageCount <= 4,
        message: `네, 좋습니다! 말씀해주신 내용을 바탕으로 더 깊이 들어가볼까요?\n\n향수는 단순한 향기가 아니라 당신의 이야기를 담는 그릇이에요. 방금 말씀하신 "${userInput}"에서 특히 끌리는 부분이 있나요?\n\n예를 들어:\n• 꽃향기라면 어떤 꽃이 가장 좋으신가요?\n• 시간대는 언제를 생각하고 계신가요? (아침, 저녁 등)\n• 혼자 있을 때와 사람들과 함께 있을 때 중 언제 사용하고 싶으신가요?`,
        shouldGenerate: false,
        extractedPreferences: {}
      },
      // 세 번째 메시지 이후
      {
        condition: () => true,
        message: `완벽해요! 말씀해주신 모든 내용을 종합해서 당신만의 특별한 향수를 설계해보겠습니다.\n\n"${userInput}"... 이 표현에서 느껴지는 감성을 향으로 풀어내겠어요. 잠시만 기다려주세요. 당신의 이야기가 향기로 변화하는 마법 같은 순간이 곧 시작됩니다... ✨`,
        shouldGenerate: true,
        extractedPreferences: extractPreferencesFromConversation(chatMessages, userInput)
      }
    ];

    return responses.find(r => r.condition()) || responses[responses.length - 1];
  };

  const extractPreferencesFromConversation = (messages: ChatMessage[], latestInput: string) => {
    const allUserMessages = messages.filter(m => m.role === 'user').map(m => m.content).join(' ') + ' ' + latestInput;
    const lowerText = allUserMessages.toLowerCase();

    // 키워드 분석으로 선호도 추출
    const preferences: any = {
      unique_request: latestInput,
      conversation_context: allUserMessages
    };

    // 한국 전통 장소 분석
    if (/경복궁|궁궐|한옥|전통|고궁/.test(allUserMessages)) {
      preferences.fragrance_family = 'oriental';
      preferences.mood = 'elegant';
      preferences.korean_traditional = '경복궁';
      preferences.korean_region = '서울';
      preferences.traditional_elements = ['한옥의 나무향', '궁궐의 품격', '전통 정원'];
    }
    else if (/제주|바다|감귤|한라산/.test(allUserMessages)) {
      preferences.fragrance_family = 'citrus';
      preferences.korean_region = '제주';
    }
    else if (/부산|해운대|바다/.test(allUserMessages)) {
      preferences.fragrance_family = 'fresh';
      preferences.korean_region = '부산';
    }

    // 향족 분석 (더 정확하게)
    if (!preferences.fragrance_family) {
      if (/꽃|플로럴|장미|재스민|꽃향기|모란|국화/.test(allUserMessages)) preferences.fragrance_family = 'floral';
      else if (/시트러스|레몬|오렌지|상큼|새콤|유자|귤/.test(allUserMessages)) preferences.fragrance_family = 'citrus';
      else if (/나무|우디|샌달우드|따뜻|소나무|편백|삼나무/.test(allUserMessages)) preferences.fragrance_family = 'woody';
      else if (/오리엔탈|스파이시|신비|인센스|계피|정향/.test(allUserMessages)) preferences.fragrance_family = 'oriental';
      else preferences.fragrance_family = 'fresh';
    }

    // 분위기 분석
    if (/로맨틱|사랑|데이트/.test(allUserMessages)) preferences.mood = 'romantic';
    else if (/우아|품격|고급|궁궐|전통|클래식/.test(allUserMessages)) preferences.mood = 'elegant';
    else if (/신비|독특|특별|신성/.test(allUserMessages)) preferences.mood = 'mysterious';
    else if (/활기|에너지|활발/.test(allUserMessages)) preferences.mood = 'energetic';
    else if (!preferences.mood) preferences.mood = 'fresh';

    // 강도 분석
    if (/강한|진한|강렬|짙은/.test(allUserMessages)) preferences.intensity = 'strong';
    else if (/은은|부드러운|가벼운|연한/.test(allUserMessages)) preferences.intensity = 'light';
    else preferences.intensity = 'moderate';

    // 계절 분석 - 한글 계절명 직접 사용
    if (/봄|벚꽃|새싹|3월|4월|5월|spring/.test(allUserMessages)) preferences.season = '봄';
    else if (/여름|더위|시원|6월|7월|8월|summer/.test(allUserMessages)) preferences.season = '여름';
    else if (/가을|단풍|차분|9월|10월|11월|autumn|fall/.test(allUserMessages)) preferences.season = '가을';
    else if (/겨울|추위|따뜻|크리스마스|눈|12월|1월|2월|winter/.test(allUserMessages)) preferences.season = '겨울';

    return preferences;
  };

  const generateFragranceFromChat = async (userInput: string, preferences: any) => {
    const request: FragranceGenerationRequest = {
      fragrance_family: preferences.fragrance_family || 'floral',
      mood: preferences.mood || 'fresh',
      intensity: preferences.intensity || 'moderate',
      season: preferences.season,
      korean_region: preferences.korean_region,
      traditional_element: preferences.traditional_elements?.[0],
      unique_request: userInput,
      conversation_context: chatMessages.filter(m => m.role === 'user').map(m => m.content).join('\n')
    };

    try {
      const response = await FragranceAI.generateFragrance(request, user?.token);
      setResult(response);

      // 상세한 결과 메시지 추가
      const detailedResultMessage: ChatMessage = {
        role: 'assistant',
        content: `🌸 **"${response.customer_info.name}"** 완성!\n\n${response.customer_info.description}\n\n"${response.customer_info.story}"\n\n이 향수가 마음에 드시나요? 혹시 조정하고 싶은 부분이 있다면 언제든 말씀해주세요. 향의 강도를 조절하거나, 특정 노트를 더하거나 빼는 것도 가능해요!`
      };

      setChatMessages(prev => [...prev, detailedResultMessage]);
    } catch (error) {
      console.error('향수 생성 오류:', error);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: '죄송합니다. 향수 생성 중 오류가 발생했습니다. 다시 시도해주세요.'
      }]);
    }
  };

  const resetAll = () => {
    setResult(null);
    setCardSelections({});
    setCurrentStep('family');
    setChatMessages([{
      role: 'assistant',
      content: '안녕하세요! 들숨의 AI 향수 아티스트입니다. 어떤 향수를 만들어드릴까요? 자유롭게 말씀해주세요.'
    }]);
    setChatInput('');
  };

  const isCardStepComplete = () => {
    const requiredSteps = ['family', 'mood', 'intensity'];
    return requiredSteps.every(step => cardSelections[step as keyof CardSelection]);
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--ivory-light)' }}>
      {/* Header */}
      <header className="border-b border-neutral-200 bg-white">
        <div className="mx-auto max-w-screen-xl px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-2xl font-light">Deulsoom</Link>
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-light">AI 향수 제작소</h1>
            {user ? (
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded text-xs ${
                  user.role === 'admin' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'
                }`}>
                  {user.role === 'admin' ? '🔑 관리자' : '👤 고객'}
                </span>
                <button
                  onClick={() => {
                    AuthService.logout();
                    setUser(null);
                  }}
                  className="text-sm text-neutral-600 hover:text-neutral-900"
                >
                  로그아웃
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowLoginModal(true)}
                className="px-4 py-2 bg-neutral-900 text-white rounded-lg text-sm hover:bg-neutral-800"
              >
                로그인
              </button>
            )}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-screen-xl px-4 py-8">
        {!result && (
          <>
            {/* Mode Selection */}
            <div className="text-center mb-12">
              <h2 className="text-3xl font-light text-neutral-900 mb-4">
                당신만의 향기를 만들어보세요
              </h2>
              <p className="text-lg text-neutral-600 mb-8">
                두 가지 방법으로 맞춤형 향수를 제작할 수 있습니다
              </p>

              <div className="inline-flex rounded-lg border border-neutral-300 bg-white p-1">
                <button
                  onClick={() => {
                    setMode('cards');
                    resetAll();
                  }}
                  className={`px-8 py-3 rounded-md font-medium transition-colors ${
                    mode === 'cards'
                      ? 'bg-neutral-900 text-white'
                      : 'text-neutral-600 hover:text-neutral-900'
                  }`}
                >
                  나의 향기 시작하기
                </button>
                <button
                  onClick={() => {
                    setMode('story');
                    resetAll();
                  }}
                  className={`px-8 py-3 rounded-md font-medium transition-colors ${
                    mode === 'story'
                      ? 'bg-neutral-900 text-white'
                      : 'text-neutral-600 hover:text-neutral-900'
                  }`}
                >
                  이야기로 만들기
                </button>
              </div>
            </div>

            {mode === 'cards' ? (
              <div className="max-w-4xl mx-auto">
                {/* Progress Indicator */}
                <div className="flex justify-center mb-8">
                  <div className="flex space-x-2">
                    {stepOrder.map((step, index) => (
                      <div
                        key={step}
                        className={`w-3 h-3 rounded-full ${
                          stepOrder.indexOf(currentStep) >= index
                            ? 'bg-neutral-900'
                            : 'bg-neutral-300'
                        }`}
                      />
                    ))}
                  </div>
                </div>

                {/* Step Title */}
                <div className="text-center mb-8">
                  <h3 className="text-2xl font-light text-neutral-900 mb-2">
                    {stepTitles[currentStep]}
                  </h3>
                  <p className="text-neutral-600">
                    {Object.keys(cardSelections).length} / {stepOrder.length} 단계 완료
                  </p>
                </div>

                {/* Current Step Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                  {cardOptions[currentStep]?.map((option) => (
                    <button
                      key={option.id}
                      onClick={() => handleCardSelect(option.id)}
                      className={`p-8 rounded-xl border-2 transition-all hover:scale-105 ${
                        cardSelections[currentStep] === option.id
                          ? 'border-neutral-900 bg-neutral-900 text-white'
                          : 'border-neutral-200 bg-white hover:border-neutral-400'
                      }`}
                    >
                      <div className="text-4xl mb-4">{option.emoji}</div>
                      <div className="font-medium text-lg mb-2">{option.title}</div>
                      <div className={`text-sm ${
                        cardSelections[currentStep] === option.id
                          ? 'text-neutral-300'
                          : 'text-neutral-500'
                      }`}>
                        {option.desc}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Navigation Buttons */}
                <div className="flex justify-between items-center">
                  <button
                    onClick={goToPreviousStep}
                    disabled={currentStep === 'family'}
                    className="px-6 py-3 border border-neutral-300 text-neutral-700 rounded-lg hover:border-neutral-400 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ← 이전
                  </button>

                  {stepOrder.indexOf(currentStep) === stepOrder.length - 1 ? (
                    <button
                      onClick={generateFromCards}
                      disabled={!isCardStepComplete() || isLoading}
                      className="px-8 py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isLoading ? '제작 중...' : '향수 만들기'}
                    </button>
                  ) : (
                    <button
                      onClick={() => {
                        const currentIndex = stepOrder.indexOf(currentStep);
                        if (currentIndex < stepOrder.length - 1) {
                          setCurrentStep(stepOrder[currentIndex + 1]);
                        }
                      }}
                      disabled={!cardSelections[currentStep]}
                      className="px-6 py-3 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      다음 →
                    </button>
                  )}
                </div>

                {/* Selected Summary */}
                {Object.keys(cardSelections).length > 0 && (
                  <div className="mt-8 p-6 bg-white rounded-lg border border-neutral-200">
                    <h4 className="font-medium mb-4">선택한 옵션들:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(cardSelections).map(([key, value]) => (
                        <span
                          key={key}
                          className="px-3 py-1 bg-neutral-100 text-neutral-700 rounded-full text-sm"
                        >
                          {value}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* Chat Mode */
              <div className="max-w-4xl mx-auto">
                <div className="bg-white rounded-xl shadow-sm border border-neutral-200 h-96 flex flex-col">
                  {/* Chat Messages */}
                  <div className="flex-1 p-6 overflow-y-auto space-y-4">
                    {chatMessages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${
                          message.role === 'user' ? 'justify-end' : 'justify-start'
                        }`}
                      >
                        <div
                          className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                            message.role === 'user'
                              ? 'bg-neutral-900 text-white'
                              : 'bg-neutral-100 text-neutral-900'
                          }`}
                        >
                          <div className="text-sm mb-1">
                            {message.role === 'user' ? '나' : 'AI 아티스트'}
                          </div>
                          <div className="text-sm">{message.content}</div>
                        </div>
                      </div>
                    ))}

                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="bg-neutral-100 text-neutral-900 px-4 py-3 rounded-lg max-w-xs">
                          <div className="text-sm mb-1">AI 아티스트</div>
                          <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                            <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Chat Input */}
                  <form onSubmit={handleChatSubmit} className="p-6 border-t border-neutral-200">
                    <div className="flex space-x-4">
                      <input
                        type="text"
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        placeholder="예: 봄날 아침 정원의 꽃향기 같은 상쾌한 향수를 만들고 싶어요..."
                        className="flex-1 px-4 py-3 border border-neutral-300 rounded-lg focus:ring-2 focus:ring-neutral-500 focus:border-transparent bg-white text-neutral-900 placeholder-neutral-400"
                        disabled={isLoading}
                      />
                      <button
                        type="submit"
                        disabled={!chatInput.trim() || isLoading}
                        className="px-6 py-3 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        전송
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
          </>
        )}

        {/* Result Display */}
        {result && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-white rounded-xl shadow-lg p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-light text-neutral-900 mb-4">
                  🌸 {result.customer_info.name}
                </h2>
                <p className="text-lg text-neutral-600 mb-6">
                  {result.customer_info.description}
                </p>
                <div className="bg-neutral-50 rounded-lg p-6 mb-6">
                  <p className="text-neutral-700 italic">
                    "{result.customer_info.story}"
                  </p>
                </div>
              </div>

              {/* Detailed Customer Info */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                {/* 향수 기본 정보 */}
                <div className="bg-neutral-50 rounded-lg p-6">
                  <h4 className="font-medium mb-4 text-neutral-900 flex items-center">
                    🧪 향수 정보
                  </h4>
                  <div className="space-y-3 text-sm text-neutral-700">
                    <div className="flex justify-between border-b border-neutral-200 pb-2">
                      <span className="font-medium text-neutral-900">향족</span>
                      <span>{result.customer_info.fragrance_family}</span>
                    </div>
                    <div className="flex justify-between border-b border-neutral-200 pb-2">
                      <span className="font-medium text-neutral-900">분위기</span>
                      <span>{result.customer_info.mood}</span>
                    </div>
                    <div className="flex justify-between border-b border-neutral-200 pb-2">
                      <span className="font-medium text-neutral-900">강도</span>
                      <span>{result.customer_info.intensity}</span>
                    </div>
                    <div className="flex justify-between border-b border-neutral-200 pb-2">
                      <span className="font-medium text-neutral-900">지속성</span>
                      <span>{result.customer_info.longevity}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-medium text-neutral-900">확산성</span>
                      <span>{result.customer_info.sillage}</span>
                    </div>
                  </div>
                </div>

                {/* 향 노트 상세 */}
                <div className="bg-neutral-50 rounded-lg p-6">
                  <h4 className="font-medium mb-4 text-neutral-900 flex items-center">
                    🌸 향 노트 구성
                  </h4>
                  <div className="space-y-4 text-sm text-neutral-700">
                    <div>
                      <div className="font-medium text-neutral-900 mb-2 flex items-center">
                        <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                        탑 노트 (첫 인상)
                      </div>
                      <p className="ml-5 text-xs leading-relaxed">{result.customer_info.top_notes_description}</p>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 mb-2 flex items-center">
                        <div className="w-3 h-3 bg-pink-400 rounded-full mr-2"></div>
                        미들 노트 (핵심)
                      </div>
                      <p className="ml-5 text-xs leading-relaxed">{result.customer_info.middle_notes_description}</p>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 mb-2 flex items-center">
                        <div className="w-3 h-3 bg-amber-600 rounded-full mr-2"></div>
                        베이스 노트 (잔향)
                      </div>
                      <p className="ml-5 text-xs leading-relaxed">{result.customer_info.base_notes_description}</p>
                    </div>
                  </div>
                </div>

                {/* 추가 정보 */}
                <div className="bg-neutral-50 rounded-lg p-6">
                  <h4 className="font-medium mb-4 text-neutral-900 flex items-center">
                    ✨ 특별한 특징
                  </h4>
                  <div className="space-y-3 text-sm text-neutral-700">
                    <div>
                      <div className="font-medium text-neutral-900 mb-1">어울리는 시간</div>
                      <p className="text-xs">{result.customer_info.best_time || '오전 10시 - 오후 6시, 특별한 만남'}</p>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 mb-1">추천 계절</div>
                      <p className="text-xs">{result.customer_info.recommended_season || '사계절'}</p>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 mb-1">어울리는 스타일</div>
                      <p className="text-xs">{result.customer_info.style_match || '캐주얼 엘레강스, 로맨틱 페미닌'}</p>
                    </div>
                    <div>
                      <div className="font-medium text-neutral-900 mb-1">예상 가격대</div>
                      <p className="text-xs font-medium text-neutral-900">{result.customer_info.price_range || '150,000 - 200,000원 (50ml)'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* 향수 스토리 확장 */}
              <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 rounded-lg p-8 mb-8">
                <h4 className="font-medium mb-4 text-neutral-900 text-center">
                  📖 당신의 향기 이야기
                </h4>
                <div className="max-w-2xl mx-auto text-center">
                  <p className="text-neutral-700 italic text-lg leading-relaxed mb-4">
                    "{result.customer_info.story}"
                  </p>
                  <div className="text-sm text-neutral-600">
                    이 향수는 당신의 독특한 감성과 개성을 담아 특별히 제작되었습니다.
                    매일 새로운 기분으로 당신만의 향기를 경험해보세요.
                  </div>
                </div>
              </div>

              {/* Admin Recipe (관리자만) */}
              {result.admin_recipe && user?.role === 'admin' && (
                <div className="bg-red-50 rounded-lg p-6 mb-8 border-2 border-dashed border-red-300">
                  <h4 className="font-medium text-red-900 mb-4">🔑 관리자 전용 상세 레시피</h4>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-xl font-bold text-red-600">
                        {result.admin_recipe.total_cost.toLocaleString()}원
                      </div>
                      <div className="text-xs text-neutral-600">총 제조비용</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-xl font-bold text-red-600">
                        {result.admin_recipe.cost_per_ml.toLocaleString()}원
                      </div>
                      <div className="text-xs text-neutral-600">ml당 비용</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-xl font-bold text-red-600">
                        {result.admin_recipe.suggested_price.toLocaleString()}원
                      </div>
                      <div className="text-xs text-neutral-600">권장 판매가</div>
                    </div>
                  </div>
                  <div className="text-sm text-neutral-700">
                    <p><strong>레시피 ID:</strong> {result.admin_recipe.recipe_id}</p>
                    <p><strong>숙성 기간:</strong> {result.admin_recipe.maturation_time}일</p>
                  </div>
                </div>
              )}

              {/* 계속 대화하기 (이야기 모드인 경우) */}
              {mode === 'story' && (
                <div className="bg-blue-50 rounded-lg p-6 mb-8 border border-blue-200">
                  <h4 className="font-medium mb-4 text-blue-900 text-center">
                    💬 향수가 마음에 드시나요?
                  </h4>
                  <p className="text-sm text-blue-700 text-center mb-4">
                    조정하고 싶은 부분이 있다면 아래에서 계속 대화해보세요!<br/>
                    예: "조금 더 달콤하게 해주세요", "향이 너무 강해요", "장미 향을 더 넣어주세요"
                  </p>
                  <form onSubmit={handleChatSubmit} className="flex space-x-4">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="향수에 대한 피드백이나 조정 요청을 말씀해주세요..."
                      className="flex-1 px-4 py-3 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white text-neutral-900 placeholder-neutral-400"
                      disabled={isLoading}
                    />
                    <button
                      type="submit"
                      disabled={!chatInput.trim() || isLoading}
                      className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isLoading ? '처리 중...' : '조정 요청'}
                    </button>
                  </form>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-center space-x-4">
                <button
                  onClick={resetAll}
                  className="px-8 py-3 border border-neutral-300 text-neutral-700 font-medium rounded-lg hover:border-neutral-400"
                >
                  다시 만들기
                </button>
                <button className="px-8 py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800">
                  사전 예약하기
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(`${result.customer_info.name}\n\n${result.customer_info.description}\n\n${result.customer_info.story}`);
                    alert('향수 정보가 클립보드에 복사되었습니다!');
                  }}
                  className="px-8 py-3 border border-neutral-300 text-neutral-700 font-medium rounded-lg hover:border-neutral-400"
                >
                  정보 복사
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Login Modal */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onLoginSuccess={(user) => {
          setUser(user);
          setShowLoginModal(false);
        }}
      />
    </div>
  );
}