'use client';

import { useState } from 'react';
import Link from 'next/link';

type CreationMode = 'chat' | 'cards';

interface FragranceProfile {
  season?: string;
  time?: string;
  mood?: string;
  intensity?: string;
  family?: string;
}

export default function CreateFragrancePage() {
  const [mode, setMode] = useState<CreationMode>('cards');
  const [profile, setProfile] = useState<FragranceProfile>({});
  const [chatInput, setChatInput] = useState('');
  const [result, setResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleCardSelection = (category: keyof FragranceProfile, value: string) => {
    setProfile(prev => ({
      ...prev,
      [category]: prev[category] === value ? undefined : value
    }));
  };

  const generateFromProfile = async () => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));

    // 선택한 프로필에 따른 향수 생성
    const topNotes = [];
    const middleNotes = [];
    const baseNotes = [];

    // 계절별 노트
    if (profile.season === '봄') {
      topNotes.push('체리 블로섬', '프리지아');
      middleNotes.push('피오니', '라일락');
    } else if (profile.season === '여름') {
      topNotes.push('베르가못', '레몬');
      middleNotes.push('자스민', '네롤리');
    } else if (profile.season === '가을') {
      topNotes.push('사과', '배');
      middleNotes.push('계피', '정향');
    } else if (profile.season === '겨울') {
      topNotes.push('오렌지', '카다몬');
      middleNotes.push('장미', '제라늄');
    }

    // 시간대별 노트
    if (profile.time === '아침') {
      topNotes.push('그레이프프루트');
      baseNotes.push('화이트머스크');
    } else if (profile.time === '오후') {
      middleNotes.push('일랑일랑');
      baseNotes.push('샌달우드');
    } else if (profile.time === '저녁') {
      middleNotes.push('투베로즈');
      baseNotes.push('앰버', '바닐라');
    }

    // 무드별 조정
    if (profile.mood === '로맨틱') {
      middleNotes.push('장미', '바이올렛');
      baseNotes.push('머스크');
    } else if (profile.mood === '활기찬') {
      topNotes.push('민트', '유칼립투스');
    } else if (profile.mood === '차분한') {
      middleNotes.push('라벤더');
      baseNotes.push('시더우드');
    } else if (profile.mood === '신비로운') {
      baseNotes.push('파출리', '인센스');
    }

    // 강도별 조정
    const intensity = profile.intensity || '중간';

    // 향 계열별 추가
    if (profile.family === '플로럴') {
      middleNotes.push('작약');
    } else if (profile.family === '우디') {
      baseNotes.push('베티버');
    } else if (profile.family === '프레시') {
      topNotes.push('그린티');
    } else if (profile.family === '오리엔탈') {
      baseNotes.push('톤카빈');
    }

    setResult({
      name: `${profile.season || '사계절'} ${profile.mood || '특별한'} 향수`,
      description: `당신이 선택한 특성을 완벽하게 구현한 맞춤 향수`,
      notes: {
        top: [...new Set(topNotes.slice(0, 3))],
        middle: [...new Set(middleNotes.slice(0, 3))],
        base: [...new Set(baseNotes.slice(0, 3))]
      },
      intensity,
      profile
    });

    setIsLoading(false);
  };

  const generateFromChat = async () => {
    if (!chatInput.trim()) return;

    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));

    // 간단한 키워드 분석
    const text = chatInput.toLowerCase();
    const topNotes = [];
    const middleNotes = [];
    const baseNotes = [];

    if (text.includes('상쾌') || text.includes('fresh')) {
      topNotes.push('베르가못', '레몬', '민트');
      middleNotes.push('네롤리', '그린티');
    } else if (text.includes('달콤')) {
      topNotes.push('복숭아', '배');
      middleNotes.push('자스민', '일랑일랑');
      baseNotes.push('바닐라', '톤카빈');
    } else {
      topNotes.push('베르가못', '라벤더');
      middleNotes.push('장미', '자스민');
      baseNotes.push('샌달우드', '머스크');
    }

    setResult({
      name: '맞춤 제작 향수',
      description: chatInput,
      notes: {
        top: topNotes.slice(0, 3),
        middle: middleNotes.slice(0, 3),
        base: baseNotes.slice(0, 3)
      },
      intensity: '중간'
    });

    setIsLoading(false);
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--ivory-light)' }}>
      {/* Header */}
      <header className="border-b border-neutral-200 bg-white">
        <div className="mx-auto max-w-screen-xl px-4 py-4 flex items-center justify-between">
          <Link href="/" className="text-2xl font-light">ZEPHYRUS</Link>
          <h1 className="text-xl font-light">향수 제작소</h1>
        </div>
      </header>

      <div className="mx-auto max-w-screen-xl px-4 py-8">
        {/* Mode Selection */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-lg border border-neutral-300 bg-white p-1">
            <button
              onClick={() => setMode('cards')}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                mode === 'cards'
                  ? 'bg-neutral-900 text-white'
                  : 'text-neutral-600 hover:text-neutral-900'
              }`}
            >
              가이드 모드
            </button>
            <button
              onClick={() => setMode('chat')}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                mode === 'chat'
                  ? 'bg-neutral-900 text-white'
                  : 'text-neutral-600 hover:text-neutral-900'
              }`}
            >
              자유 입력 모드
            </button>
          </div>
        </div>

        {mode === 'cards' ? (
          <div className="space-y-8">
            {/* Season Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">계절을 선택하세요</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['봄', '여름', '가을', '겨울'].map(season => (
                  <button
                    key={season}
                    onClick={() => handleCardSelection('season', season)}
                    className={`p-6 rounded-lg border-2 transition-all ${
                      profile.season === season
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white hover:border-neutral-400'
                    }`}
                  >
                    <div className="text-2xl mb-2">
                      {season === '봄' && '🌸'}
                      {season === '여름' && '☀️'}
                      {season === '가을' && '🍂'}
                      {season === '겨울' && '❄️'}
                    </div>
                    <div className="font-medium">{season}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Time Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">시간대를 선택하세요</h3>
              <div className="grid grid-cols-3 gap-4">
                {['아침', '오후', '저녁'].map(time => (
                  <button
                    key={time}
                    onClick={() => handleCardSelection('time', time)}
                    className={`p-6 rounded-lg border-2 transition-all ${
                      profile.time === time
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white hover:border-neutral-400'
                    }`}
                  >
                    <div className="text-2xl mb-2">
                      {time === '아침' && '🌅'}
                      {time === '오후' && '☀️'}
                      {time === '저녁' && '🌙'}
                    </div>
                    <div className="font-medium">{time}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Mood Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">분위기를 선택하세요</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['로맨틱', '활기찬', '차분한', '신비로운'].map(mood => (
                  <button
                    key={mood}
                    onClick={() => handleCardSelection('mood', mood)}
                    className={`p-6 rounded-lg border-2 transition-all ${
                      profile.mood === mood
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white hover:border-neutral-400'
                    }`}
                  >
                    <div className="text-2xl mb-2">
                      {mood === '로맨틱' && '💝'}
                      {mood === '활기찬' && '✨'}
                      {mood === '차분한' && '🕊️'}
                      {mood === '신비로운' && '🔮'}
                    </div>
                    <div className="font-medium">{mood}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Intensity Selection */}
            <div>
              <h3 className="text-lg font-medium mb-4">향의 강도</h3>
              <div className="grid grid-cols-3 gap-4">
                {['가벼움', '중간', '진함'].map(intensity => (
                  <button
                    key={intensity}
                    onClick={() => handleCardSelection('intensity', intensity)}
                    className={`p-6 rounded-lg border-2 transition-all ${
                      profile.intensity === intensity
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white hover:border-neutral-400'
                    }`}
                  >
                    <div className="font-medium">{intensity}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Fragrance Family */}
            <div>
              <h3 className="text-lg font-medium mb-4">향 계열</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['플로럴', '우디', '프레시', '오리엔탈'].map(family => (
                  <button
                    key={family}
                    onClick={() => handleCardSelection('family', family)}
                    className={`p-6 rounded-lg border-2 transition-all ${
                      profile.family === family
                        ? 'border-neutral-900 bg-neutral-900 text-white'
                        : 'border-neutral-200 bg-white hover:border-neutral-400'
                    }`}
                  >
                    <div className="font-medium">{family}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <div className="flex justify-center">
              <button
                onClick={generateFromProfile}
                disabled={Object.keys(profile).length === 0 || isLoading}
                className="px-12 py-4 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? '제작 중...' : '향수 만들기'}
              </button>
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto">
            <div className="bg-white rounded-lg p-8">
              <h3 className="text-xl font-medium mb-4">원하시는 향을 자유롭게 설명해주세요</h3>
              <textarea
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="예: 봄날 아침 정원을 거닐 때의 상쾌한 꽃향기, 따뜻한 햇살과 함께 느껴지는 부드러운 향..."
                className="w-full h-32 p-4 border border-neutral-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-neutral-900 text-neutral-900"
              />
              <button
                onClick={generateFromChat}
                disabled={!chatInput.trim() || isLoading}
                className="mt-4 w-full py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? '제작 중...' : '향수 만들기'}
              </button>
            </div>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div className="mt-12 max-w-4xl mx-auto">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-2xl font-medium text-center mb-6">{result.name}</h2>
              <p className="text-center text-neutral-600 mb-8">{result.description}</p>

              <div className="grid grid-cols-3 gap-8 mb-8">
                <div className="text-center">
                  <h4 className="font-medium mb-3">탑 노트</h4>
                  {result.notes.top.map((note: string, i: number) => (
                    <p key={i} className="text-neutral-600">{note}</p>
                  ))}
                </div>
                <div className="text-center">
                  <h4 className="font-medium mb-3">미들 노트</h4>
                  {result.notes.middle.map((note: string, i: number) => (
                    <p key={i} className="text-neutral-600">{note}</p>
                  ))}
                </div>
                <div className="text-center">
                  <h4 className="font-medium mb-3">베이스 노트</h4>
                  {result.notes.base.map((note: string, i: number) => (
                    <p key={i} className="text-neutral-600">{note}</p>
                  ))}
                </div>
              </div>

              <div className="flex justify-center space-x-4">
                <button
                  className="px-8 py-3 bg-neutral-900 text-white font-medium rounded-lg hover:bg-neutral-800 transition-colors"
                >
                  이 향수 주문하기
                </button>
                <button
                  onClick={() => {
                    setResult(null);
                    setProfile({});
                    setChatInput('');
                  }}
                  className="px-8 py-3 border border-neutral-300 text-neutral-700 font-medium rounded-lg hover:border-neutral-400 transition-colors"
                >
                  다시 만들기
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}