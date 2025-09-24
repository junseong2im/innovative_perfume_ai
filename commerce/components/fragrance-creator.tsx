'use client';

import { useState } from 'react';

export default function FragranceCreator() {
  const [description, setDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim()) return;

    setIsLoading(true);
    setResult(null);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
      const response = await fetch(`${API_URL}/api/v1/generate/recipe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          description: description,
          fragrance_family: 'fresh',
          mood: 'romantic',
          intensity: 'moderate',
          gender: 'unisex'
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate fragrance recipe');
      }

      const data = await response.json();

      // Transform API response to match expected format
      setResult({
        name: data.name || '맞춤형 향수',
        description: data.description || `"${description}"에서 영감을 받은 개인 맞춤 향수`,
        notes: {
          top: data.top_notes || ['Bergamot', 'Citrus'],
          middle: data.heart_notes || ['Rose', 'Jasmine'],
          base: data.base_notes || ['Sandalwood', 'Musk']
        },
        price: data.price || '185,000 KRW'
      });
    } catch (error) {
      console.error('Error generating fragrance:', error);
      // Use basic fallback on error
      setResult({
        name: '맞춤형 향수',
        description: `"${description}"에서 영감을 받은 향수`,
        notes: {
          top: ['Citrus'],
          middle: ['Floral'],
          base: ['Woody']
        },
        price: '185,000 KRW'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const examplePrompts = [
    "봄꽃에 내린 상쾌한 아침 비",
    "시트러스와 나무향이 어우러진 따뜻한 여름 저녁",
    "향신료가 가드한 신비로운 가을 숲",
    "바닐라 향이 나는 아늠한 겨울 벽난로"
  ];

  return (
    <div className="mx-auto max-w-4xl">
      <form onSubmit={handleSubmit} className="mb-12">
        <div className="relative">
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="원하시는 향을 설명해주세요... (예: '비 갬 후 정원을 거닐 듯 상쾌한 시트러스와 꽃향기가 어우러진 아침 향기')"
            className="w-full h-32 px-6 py-4 text-lg border border-neutral-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:border-transparent placeholder-neutral-400"
            style={{backgroundColor: 'var(--ivory-light)'}}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!description.trim() || isLoading}
            className="absolute bottom-4 right-4 px-8 py-3 text-white text-sm font-medium rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            style={{backgroundColor: 'var(--light-brown)', ':hover': {backgroundColor: 'var(--light-brown-dark)'}}}
            onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--light-brown-dark)'}
            onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--light-brown)'}
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>생성 중...</span>
              </div>
            ) : (
              '향수 만들기'
            )}
          </button>
        </div>
      </form>

      {/* Example Prompts */}
      <div className="mb-12">
        <p className="mb-4 text-sm text-neutral-600 text-center">예시를 참고해보세요:</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examplePrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => setDescription(prompt)}
              className="p-4 text-sm text-left border border-neutral-200 rounded-lg hover:border-neutral-400 transition-colors"
              style={{backgroundColor: 'var(--ivory-light)'}}
              disabled={isLoading}
            >
              "{prompt}"
            </button>
          ))}
        </div>
      </div>

      {/* Result Display */}
      {result && (
        <div className="border border-neutral-200 rounded-lg p-8 shadow-lg" style={{backgroundColor: 'var(--ivory-light)'}}>
          <div className="text-center mb-8">
            <h3 className="text-2xl font-light text-neutral-900 mb-2">{result.name}</h3>
            <p className="text-neutral-600 mb-4">{result.description}</p>
            <p className="text-xl font-medium text-neutral-900">{result.price}</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <h4 className="font-medium text-neutral-900 mb-3">탑 노트</h4>
              <div className="space-y-1">
                {result.notes.top.map((note: string, index: number) => (
                  <p key={index} className="text-sm text-neutral-600">{note}</p>
                ))}
              </div>
            </div>

            <div className="text-center">
              <h4 className="font-medium text-neutral-900 mb-3">미들 노트</h4>
              <div className="space-y-1">
                {result.notes.middle.map((note: string, index: number) => (
                  <p key={index} className="text-sm text-neutral-600">{note}</p>
                ))}
              </div>
            </div>

            <div className="text-center">
              <h4 className="font-medium text-neutral-900 mb-3">베이스 노트</h4>
              <div className="space-y-1">
                {result.notes.base.map((note: string, index: number) => (
                  <p key={index} className="text-sm text-neutral-600">{note}</p>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-8 text-center">
            <button
              className="px-8 py-3 text-white font-medium rounded-md transition-colors mr-4"
              style={{backgroundColor: 'var(--light-brown)'}}
              onMouseEnter={(e) => e.target.style.backgroundColor = 'var(--light-brown-dark)'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'var(--light-brown)'}
            >
              맞춤 향수 주문하기
            </button>
            <button
              onClick={() => setResult(null)}
              className="px-8 py-3 border border-neutral-300 text-neutral-700 font-medium rounded-md hover:border-neutral-400 transition-colors"
            >
              다시 만들기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}