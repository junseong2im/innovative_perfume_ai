import { NextRequest, NextResponse } from 'next/server';
import { AIPerfumerEnhanced } from '../../../lib/ai-perfumer-enhanced';

const aiPerfumer = new AIPerfumerEnhanced();

export async function POST(request: NextRequest) {
  try {
    const { message, context = [] } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // 대화 맥락 결합
    const fullContext = [...context, message].join(' ');

    // AI 향수 시스템으로 응답 생성
    const response = aiPerfumer.generateResponse(message, fullContext);

    // 향수 생성이 필요한지 확인 (대화가 충분히 진행된 경우)
    let fragrance = null;
    if (context.length >= 2) {
      fragrance = aiPerfumer.executeCreativeProcess(fullContext);
    }

    return NextResponse.json({
      response,
      fragrance,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('AI Perfumer API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}