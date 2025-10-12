# fragrance_ai/llm/llama_hints.py
"""
Llama Hints Generator
Generates creative hints for MOGA exploration (creative mode only)
"""

import json
import re
import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

logger = logging.getLogger(__name__)


# System prompt for Llama
LLAMA_SYSTEM_PROMPT = """You are a creative hint generator for perfume creation.
Given user input, generate 3-8 creative words or short phrases that inspire fragrance design.

Output ONLY a JSON list like: ["word1", "phrase 2", "concept3"]
NO explanations. NO additional text."""


class LlamaHintsGenerator:
    """
    Llama LLM client for creative hint generation
    Only used in "creative" mode
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        dtype: str = "float16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        hints_limit: int = 8
    ):
        """
        Initialize Llama hints generator

        Args:
            model_name: HuggingFace model name
            dtype: Data type
            device_map: Device mapping
            max_new_tokens: Max tokens (hints are short)
            load_in_4bit: Use 4-bit quantization
            hints_limit: Maximum number of hints
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.hints_limit = hints_limit

        logger.info(f"Loading Llama model: {model_name}")

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
            logger.info(f"Llama model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise

    def generate_hints(self, user_text: str, temperature: float = 0.9, top_p: float = 0.95) -> List[str]:
        """
        Generate creative hints from user text

        Args:
            user_text: User input text
            temperature: Higher temperature for creativity
            top_p: Nucleus sampling

        Returns:
            List of creative hints (max 8, length 2-48)
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
                max_length=1024
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
            logger.info(f"Llama generation completed in {elapsed_ms:.2f}ms")

            # Parse hints
            hints = self._parse_hints(generated_text)

            if hints:
                logger.info(f"Generated {len(hints)} creative hints")
                return hints
            else:
                logger.warning(f"No hints generated, using fallback")
                return self._fallback_hints(user_text)

        except Exception as e:
            logger.error(f"Llama generation failed: {e}, using fallback")
            return self._fallback_hints(user_text)

    def _build_prompt(self, user_text: str) -> str:
        """Build prompt for Llama"""
        messages = [
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": f"Input:\n{user_text}\n\nCreative hints (JSON list):"}
        ]

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback
            prompt = f"{LLAMA_SYSTEM_PROMPT}\n\nInput:\n{user_text}\n\nCreative hints (JSON list):"

        return prompt

    def _parse_hints(self, response_text: str) -> List[str]:
        """Parse hints from model response"""
        # Try to extract JSON list
        # Pattern: ["...", "...", ...]
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                hints = json.loads(json_str)

                # Validate
                if isinstance(hints, list):
                    # Filter valid hints
                    valid_hints = []
                    for hint in hints[:self.hints_limit]:
                        if isinstance(hint, str):
                            hint = hint.strip()
                            if 2 <= len(hint) <= 48:
                                valid_hints.append(hint)

                    return valid_hints

            except json.JSONDecodeError:
                pass

        # Try parsing line by line if JSON fails
        lines = response_text.split('\n')
        hints = []
        for line in lines:
            line = line.strip('- *"\'[]').strip()
            if 2 <= len(line) <= 48:
                hints.append(line)
                if len(hints) >= self.hints_limit:
                    break

        return hints

    def _fallback_hints(self, user_text: str) -> List[str]:
        """Generate fallback hints from keywords in text"""
        # Simple keyword extraction as fallback
        keywords = []

        # Extract Korean nouns (simple heuristic)
        korean_words = re.findall(r'[가-힣]{2,}', user_text)
        for word in korean_words[:4]:
            if len(word) <= 48:
                keywords.append(word)

        # Extract English words
        english_words = re.findall(r'[a-zA-Z]{3,}', user_text)
        for word in english_words[:4]:
            if len(word) <= 48 and word.lower() not in ['the', 'and', 'for']:
                keywords.append(word)

        # Return up to hints_limit
        return keywords[:self.hints_limit] if keywords else ["fresh", "clean", "elegant"]


# Singleton instance
_llama_generator: Optional[LlamaHintsGenerator] = None


def get_llama_generator(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    **kwargs
) -> LlamaHintsGenerator:
    """
    Get or create Llama generator singleton

    Args:
        model_name: Model name
        **kwargs: Additional arguments

    Returns:
        LlamaHintsGenerator instance
    """
    global _llama_generator
    if _llama_generator is None:
        _llama_generator = LlamaHintsGenerator(model_name=model_name, **kwargs)
    return _llama_generator


def generate_creative_hints(user_text: str, **kwargs) -> List[str]:
    """
    Convenience function to generate creative hints

    Args:
        user_text: User input text
        **kwargs: Additional arguments

    Returns:
        List of creative hints (max 8)
    """
    generator = get_llama_generator()
    return generator.generate_hints(user_text, **kwargs)


__all__ = [
    "LlamaHintsGenerator",
    "get_llama_generator",
    "generate_creative_hints",
    "LLAMA_SYSTEM_PROMPT"
]
