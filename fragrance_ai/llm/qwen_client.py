# fragrance_ai/llm/qwen_client.py
"""
Qwen Client
Main LLM for Korean brief interpretation and JSON generation
"""

import json
import re
import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from .schemas import CreativeBrief, DEFAULT_BRIEF

logger = logging.getLogger(__name__)


# System prompt for Qwen
QWEN_SYSTEM_PROMPT = """당신은 한국어 향수 조향 보조 LLM입니다.
출력은 반드시 아래 JSON 스키마로만 반환하세요. 불필요한 텍스트는 금지됩니다.

JSON 스키마 키:
- language: "ko" 또는 "en"
- mood: 분위기/감정 키워드 배열 (예: ["fresh", "romantic"])
- season: 계절 배열 (spring, summer, autumn, winter)
- notes_preference: 노트 선호도 (0~1 범위, 합 <= 1)
  예: {"citrus": 0.3, "floral": 0.4, "woody": 0.3}
- forbidden_ingredients: 금지 성분 배열
- budget_tier: "low", "mid", "high" 중 하나
- target_profile: "daily_fresh", "evening", "luxury", "sport", "signature" 중 하나
- constraints: 제약 조건 딕셔너리 (예: {"max_allergens_ppm": 500})
- product_category: "EDP", "EDT", "PARFUM" 중 하나
- creative_hints: 빈 배열 [] (Llama가 채움)

값 범위와 enum을 반드시 지키고, 누락 없이 채우세요.
한국어 입력을 우선 처리하세요."""


class QwenClient:
    """
    Qwen LLM client for brief generation
    Supports local inference with quantization
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        load_in_4bit: bool = False
    ):
        """
        Initialize Qwen client

        Args:
            model_name: HuggingFace model name
            dtype: Data type (float16, float32)
            device_map: Device mapping strategy
            max_new_tokens: Max tokens to generate
            load_in_4bit: Use 4-bit quantization
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading Qwen model: {model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model with optional quantization
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            else:
                torch_dtype = torch.float16 if dtype == "float16" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )

            self.model.eval()
            logger.info(f"Qwen model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise

    def infer_brief(self, user_text: str, temperature: float = 0.7, top_p: float = 0.9) -> Optional[CreativeBrief]:
        """
        Infer CreativeBrief from user text

        Args:
            user_text: User input text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            CreativeBrief or None if failed
        """
        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_prompt(user_text)

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Qwen generation completed in {elapsed_ms:.2f}ms")

            # Parse JSON
            brief = self._parse_json_response(generated_text)

            if brief:
                logger.info(f"Successfully parsed CreativeBrief from Qwen")
                return brief
            else:
                logger.warning(f"Failed to parse JSON from Qwen response")
                return None

        except Exception as e:
            logger.error(f"Qwen inference failed: {e}")
            return None

    def _build_prompt(self, user_text: str) -> str:
        """Build prompt for Qwen"""
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": f"사용자 입력:\n{user_text}\n\nJSON 응답:"}
        ]

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = f"{QWEN_SYSTEM_PROMPT}\n\n사용자 입력:\n{user_text}\n\nJSON 응답:"

        return prompt

    def _parse_json_response(self, response_text: str) -> Optional[CreativeBrief]:
        """
        Parse JSON from model response

        Args:
            response_text: Raw model output

        Returns:
            CreativeBrief or None if failed
        """
        # Try to extract JSON from response
        # Pattern 1: Find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)

                # Validate with Pydantic
                brief = CreativeBrief(**data)
                return brief

            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return None
            except Exception as e:
                logger.warning(f"Pydantic validation error: {e}")
                return None

        # Pattern 2: Try entire response as JSON
        try:
            data = json.loads(response_text)
            brief = CreativeBrief(**data)
            return brief
        except:
            pass

        logger.warning(f"Could not extract valid JSON from response")
        return None


# Singleton instance
_qwen_client: Optional[QwenClient] = None


def get_qwen_client(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    **kwargs
) -> QwenClient:
    """
    Get or create Qwen client singleton

    Args:
        model_name: Model name
        **kwargs: Additional arguments for QwenClient

    Returns:
        QwenClient instance
    """
    global _qwen_client
    if _qwen_client is None:
        _qwen_client = QwenClient(model_name=model_name, **kwargs)
    return _qwen_client


def infer_brief_qwen(user_text: str, **kwargs) -> Optional[CreativeBrief]:
    """
    Convenience function to infer brief using Qwen

    Args:
        user_text: User input text
        **kwargs: Additional arguments for infer_brief

    Returns:
        CreativeBrief or None
    """
    client = get_qwen_client()
    return client.infer_brief(user_text, **kwargs)


__all__ = [
    "QwenClient",
    "get_qwen_client",
    "infer_brief_qwen",
    "QWEN_SYSTEM_PROMPT"
]
