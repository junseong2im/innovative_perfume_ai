"""
Transformers 모델 로더
- Hugging Face 모델 로드 및 관리
- 4-bit/8-bit 양자화 지원
- 한국어 특화 모델 지원
"""

import torch
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import os
import json

logger = logging.getLogger(__name__)


@dataclass
class TransformersConfig:
    """Transformers 설정"""
    model_name: str = "microsoft/DialoGPT-medium"  # 기본 모델
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit: bool = True
    use_8bit: bool = False
    max_length: int = 1024
    temperature: float = 0.8
    top_p: float = 0.9
    cache_dir: str = "./models/cache"


class TransformersGenerator:
    """Transformers 기반 생성 모델"""

    def __init__(self, config: Optional[TransformersConfig] = None):
        """생성기 초기화"""
        self.config = config or TransformersConfig()
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def initialize(self):
        """모델 초기화 (지연 로딩)"""
        if self._initialized:
            return

        try:
            # 필요한 라이브러리 임포트
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig
            )

            logger.info(f"Loading model: {self.config.model_name}")

            # 양자화 설정
            quantization_config = None
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif self.config.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )

            # 모델 로드
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                ).to(self.config.device)

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )

            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._initialized = True
            logger.info(f"Model loaded successfully on {self.config.device}")

        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            logger.info("Install with: pip install transformers accelerate bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> str:
        """텍스트 생성"""
        if not self._initialized:
            self.initialize()

        try:
            # 입력 토큰화
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            )

            if self.config.device != "cpu" and not self.config.use_4bit and not self.config.use_8bit:
                inputs = inputs.to(self.config.device)

            # 생성 파라미터
            gen_params = {
                "max_new_tokens": max_new_tokens or 256,
                "temperature": temperature or self.config.temperature,
                "top_p": top_p or self.config.top_p,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }

            # 텍스트 생성
            with torch.no_grad():
                outputs = self.model.generate(inputs, **gen_params)

            # 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.shape[-1]:],  # 입력 부분 제외
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """채팅 형식 생성"""
        # 메시지를 프롬프트로 변환
        prompt = self._format_chat_prompt(messages)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """채팅 메시지를 프롬프트로 변환"""
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # 마지막에 Assistant: 추가
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)


class KoreanPerfumeGenerator(TransformersGenerator):
    """한국어 향수 전문 생성 모델"""

    def __init__(self):
        """한국어 모델로 초기화"""
        config = TransformersConfig(
            model_name="beomi/KoAlpaca-Polyglot-5.8B",  # 한국어 특화 모델
            use_4bit=True,  # 메모리 절약을 위한 4-bit 양자화
            temperature=0.8,
            top_p=0.95
        )
        super().__init__(config)

        self.perfume_prompt_template = """당신은 전문 조향사입니다. 사용자의 요청에 따라 창의적이고 조화로운 향수 레시피를 만들어주세요.

요청: {request}

향수 레시피:
이름:
컨셉:
계절:
분위기:

노트 구성:
- 탑 노트 (0-30분):
  *
  *

- 하트 노트 (30분-4시간):
  *
  *

- 베이스 노트 (4시간 이상):
  *
  *

조향 포인트:
"""

    def generate_perfume_recipe(
        self,
        request: str,
        style: str = "modern",
        intensity: str = "moderate"
    ) -> Dict[str, Any]:
        """향수 레시피 생성"""
        # 프롬프트 생성
        prompt = self.perfume_prompt_template.format(request=request)

        if style:
            prompt += f"\n스타일: {style}"
        if intensity:
            prompt += f"\n강도: {intensity}"

        # 레시피 생성
        generated_text = self.generate(
            prompt,
            max_new_tokens=500,
            temperature=0.8
        )

        # 파싱 (실제로는 더 정교한 파싱 필요)
        return self._parse_recipe(generated_text)

    def _parse_recipe(self, text: str) -> Dict[str, Any]:
        """생성된 텍스트를 레시피로 파싱"""
        lines = text.strip().split('\n')
        recipe = {
            "name": "Artisan Creation",
            "description": "",
            "top_notes": [],
            "heart_notes": [],
            "base_notes": [],
            "style": "",
            "season": "",
            "mood": ""
        }

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "이름:" in line:
                recipe["name"] = line.split("이름:")[-1].strip()
            elif "컨셉:" in line:
                recipe["description"] = line.split("컨셉:")[-1].strip()
            elif "계절:" in line:
                recipe["season"] = line.split("계절:")[-1].strip()
            elif "분위기:" in line:
                recipe["mood"] = line.split("분위기:")[-1].strip()
            elif "탑 노트" in line or "톱 노트" in line:
                current_section = "top"
            elif "하트 노트" in line or "미들 노트" in line:
                current_section = "heart"
            elif "베이스 노트" in line or "베이스노트" in line:
                current_section = "base"
            elif line.startswith("*") or line.startswith("-"):
                # 노트 항목
                note = line.lstrip("*- ").strip()
                if note and current_section:
                    if current_section == "top":
                        recipe["top_notes"].append(note)
                    elif current_section == "heart":
                        recipe["heart_notes"].append(note)
                    elif current_section == "base":
                        recipe["base_notes"].append(note)

        return recipe


# 전역 생성기 인스턴스
_generator_instance = None


def get_perfume_generator() -> KoreanPerfumeGenerator:
    """향수 생성기 싱글톤"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = KoreanPerfumeGenerator()
    return _generator_instance


def check_transformers_availability() -> bool:
    """Transformers 라이브러리 사용 가능 여부 확인"""
    try:
        import transformers
        import accelerate
        import bitsandbytes
        return True
    except ImportError:
        return False