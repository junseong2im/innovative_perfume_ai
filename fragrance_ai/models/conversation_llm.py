"""
진짜 대화형 LLM 시스템
향수 레시피 생성과는 별도로 자연스러운 대화를 위한 LLM
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    pipeline
)
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ConversationalLLM:
    """진짜 대화형 LLM - 자연스러운 대화 처리"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 대화 히스토리 저장
        self.conversation_history = []

        # 모델 및 토크나이저 로드
        self._load_model()

        # 향수 전문 프롬프트 설정
        self._setup_perfumer_persona()

        logger.info(f"대화형 LLM 초기화 완료 - 모델: {model_name}, 디바이스: {self.device}")

    def _load_model(self):
        """실제 LLM 모델 로드"""
        try:
            # 양자화 설정 (GPU 메모리 절약)
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                # CPU에서는 작은 모델 사용
                self.model_name = "microsoft/DialoGPT-small"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model.to(self.device)

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 대화형 파이프라인 생성
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

        except Exception as e:
            logger.error(f"LLM 모델 로드 실패: {e}")
            raise

    def _setup_perfumer_persona(self):
        """향수 전문가 페르소나 설정"""
        self.system_persona = """당신은 세계적으로 유명한 마스터 조향사 'Artisan'입니다.

전문 배경:
- 50년 경력의 향수 제작 마스터
- 파리 그라스에서 수학한 정통 조향사
- 3000가지 이상의 향료에 대한 깊은 지식
- 고객 맞춤형 향수 제작 전문가

대화 스타일:
- 따뜻하고 친근하면서도 전문적
- 고객의 감정과 취향을 세심하게 파악
- 향수에 대한 시적이고 감성적인 표현 사용
- 질문을 통해 고객을 깊이 이해하려 노력

주요 역할:
- 고객과 자연스러운 대화를 통한 상담
- 향수 취향 및 라이프스타일 파악
- 향수에 대한 교육 및 조언 제공
- 향수 제작 의뢰 전 충분한 상담"""

    async def chat(self, user_message: str, context: Optional[Dict] = None) -> str:
        """실제 LLM을 사용한 대화 처리"""
        try:
            # 대화 맥락 구성
            conversation_context = self._build_conversation_context(user_message)

            # LLM 추론 실행
            response = await self._generate_llm_response(conversation_context)

            # 대화 히스토리에 저장
            self.conversation_history.append({
                "user": user_message,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })

            # 히스토리 길이 제한 (메모리 관리)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-8:]

            return response

        except Exception as e:
            logger.error(f"대화 처리 실패: {e}")
            raise Exception("대화 시스템에 일시적인 문제가 발생했습니다")

    def _build_conversation_context(self, current_message: str) -> str:
        """대화 맥락 구성"""
        context = f"{self.system_persona}\n\n"

        # 최근 대화 히스토리 포함
        if self.conversation_history:
            context += "최근 대화:\n"
            for chat in self.conversation_history[-3:]:  # 최근 3개만
                context += f"고객: {chat['user']}\n"
                context += f"조향사: {chat['assistant']}\n\n"

        context += f"고객: {current_message}\n조향사:"

        return context

    async def _generate_llm_response(self, prompt: str) -> str:
        """실제 LLM 추론으로 응답 생성"""
        try:
            # 토큰화
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            inputs = inputs.to(self.device)

            # LLM 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # 디코딩 및 후처리
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 프롬프트 부분 제거하고 응답만 추출
            response = generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

            # 응답 품질 개선
            response = self._improve_response_quality(response)

            return response

        except Exception as e:
            logger.error(f"LLM 생성 실패: {e}")
            raise

    def _improve_response_quality(self, response: str) -> str:
        """LLM 응답 품질 개선"""
        if not response or len(response.strip()) < 10:
            return "죄송합니다. 조금 더 구체적으로 말씀해 주시겠어요? 어떤 향수를 찾고 계신지 더 자세히 알려주세요."

        # 불완전한 문장 정리
        if not response.endswith(('.', '!', '?', '요', '다', '까요', '습니다')):
            response += "."

        # 향수 전문가 톤 추가
        if not any(word in response for word in ['향', '향수', '조향', '노트', '향료']):
            response += " 향수에 대해 더 궁금한 점이 있으시면 언제든 물어보세요."

        return response

    def reset_conversation(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        logger.info("대화 히스토리가 초기화되었습니다")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """현재 대화 요약 정보"""
        return {
            "total_exchanges": len(self.conversation_history),
            "model_name": self.model_name,
            "device": str(self.device),
            "last_interaction": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }

# 전역 인스턴스
conversation_llm = None

def get_conversation_llm():
    """대화형 LLM 인스턴스 반환"""
    global conversation_llm
    if conversation_llm is None:
        conversation_llm = ConversationalLLM()
    return conversation_llm