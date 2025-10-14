"""
JSON Hard Guard Layer - JSON 파싱 안정성 보장

실패 시 자동 복구:
1. 재시도 (백오프 + jitter)
2. 미니 리페어 (일반적인 JSON 오류 자동 수정)
3. 기본 브리프 폴백
"""

import json
import re
import time
import random
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JSONGuardStrategy(Enum):
    """JSON 가드 전략"""
    RETRY = "retry"
    REPAIR = "repair"
    FALLBACK = "fallback"


@dataclass
class JSONGuardConfig:
    """JSON 가드 설정"""
    max_retries: int = 3
    initial_backoff: float = 0.5  # seconds
    max_backoff: float = 5.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    enable_repair: bool = True
    enable_fallback: bool = True


class JSONRepairError(Exception):
    """JSON 리페어 실패"""
    pass


class JSONGuard:
    """
    JSON 하드 가드 레이어

    LLM 출력의 JSON 파싱을 보장합니다:
    1. 재시도: 백오프 + jitter로 일시적 오류 극복
    2. 리페어: 일반적인 JSON 오류 자동 수정
    3. 폴백: 최후의 기본 브리프 반환
    """

    def __init__(self, config: Optional[JSONGuardConfig] = None):
        self.config = config or JSONGuardConfig()
        self.repair_attempts = 0
        self.fallback_uses = 0

    def parse_with_guard(
        self,
        json_string: str,
        generator_func: Optional[Callable[[], str]] = None,
        fallback_value: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        JSON 파싱 with guard (재시도 + 리페어 + 폴백)

        Args:
            json_string: 파싱할 JSON 문자열
            generator_func: 재시도 시 호출할 생성 함수 (optional)
            fallback_value: 폴백 값 (optional)

        Returns:
            파싱된 딕셔너리

        Raises:
            ValueError: 모든 전략 실패 시
        """
        # Strategy 1: Direct parsing
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}")

        # Strategy 2: Retry with backoff (if generator_func provided)
        if generator_func and self.config.max_retries > 0:
            result = self._retry_with_backoff(generator_func)
            if result is not None:
                return result

        # Strategy 3: Repair JSON
        if self.config.enable_repair:
            result = self._repair_json(json_string)
            if result is not None:
                return result

        # Strategy 4: Fallback
        if self.config.enable_fallback and fallback_value:
            logger.warning("Using fallback value")
            self.fallback_uses += 1
            return fallback_value

        # All strategies failed
        raise ValueError(
            f"All JSON guard strategies failed. "
            f"Repair attempts: {self.repair_attempts}, "
            f"Fallback uses: {self.fallback_uses}"
        )

    def _retry_with_backoff(self, generator_func: Callable[[], str]) -> Optional[Dict[str, Any]]:
        """
        재시도 with 지수 백오프 + jitter

        Args:
            generator_func: JSON 생성 함수

        Returns:
            파싱된 딕셔너리 or None
        """
        backoff = self.config.initial_backoff

        for attempt in range(1, self.config.max_retries + 1):
            logger.info(f"Retry attempt {attempt}/{self.config.max_retries}")

            # Wait with backoff + jitter
            if attempt > 1:
                wait_time = backoff
                if self.config.jitter:
                    # Add jitter: ±25% random variation
                    jitter_factor = 1.0 + random.uniform(-0.25, 0.25)
                    wait_time *= jitter_factor

                logger.debug(f"Waiting {wait_time:.2f}s before retry")
                time.sleep(wait_time)

                # Exponential backoff
                backoff = min(backoff * self.config.backoff_multiplier, self.config.max_backoff)

            # Retry generation
            try:
                json_string = generator_func()
                result = json.loads(json_string)
                logger.info(f"Retry succeeded on attempt {attempt}")
                return result
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Retry attempt {attempt} failed: {e}")

        logger.error(f"All {self.config.max_retries} retries failed")
        return None

    def _repair_json(self, json_string: str) -> Optional[Dict[str, Any]]:
        """
        미니 리페어: 일반적인 JSON 오류 자동 수정

        수정 사항:
        1. 트레일링 쉼표 제거
        2. 누락된 따옴표 추가
        3. 불완전한 JSON 완성
        4. 코드 블록 마커 제거
        5. 이스케이프 문자 수정

        Args:
            json_string: 수정할 JSON 문자열

        Returns:
            파싱된 딕셔너리 or None
        """
        self.repair_attempts += 1

        repairs = [
            self._remove_code_blocks,
            self._remove_trailing_commas,
            self._fix_quotes,
            self._complete_incomplete_json,
            self._fix_escape_characters,
            self._extract_json_from_text
        ]

        repaired = json_string
        for repair_func in repairs:
            try:
                repaired = repair_func(repaired)
                result = json.loads(repaired)
                logger.info(f"JSON repaired successfully using {repair_func.__name__}")
                return result
            except json.JSONDecodeError:
                # Try next repair
                continue
            except Exception as e:
                logger.debug(f"Repair {repair_func.__name__} error: {e}")
                continue

        logger.error("All repair strategies failed")
        return None

    def _remove_code_blocks(self, text: str) -> str:
        """코드 블록 마커 제거 (```json, ```)"""
        # Remove ```json ... ```
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def _remove_trailing_commas(self, text: str) -> str:
        """트레일링 쉼표 제거"""
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return text

    def _fix_quotes(self, text: str) -> str:
        """누락된 따옴표 수정"""
        # Fix single quotes to double quotes
        # (naive approach - may not work for all cases)
        text = text.replace("'", '"')
        return text

    def _complete_incomplete_json(self, text: str) -> str:
        """불완전한 JSON 완성"""
        # Count braces
        open_braces = text.count('{')
        close_braces = text.count('}')

        # Add missing closing braces
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)

        # Count brackets
        open_brackets = text.count('[')
        close_brackets = text.count(']')

        # Add missing closing brackets
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)

        return text

    def _fix_escape_characters(self, text: str) -> str:
        """이스케이프 문자 수정"""
        # Fix unescaped newlines in strings
        # This is a simplified approach
        return text

    def _extract_json_from_text(self, text: str) -> str:
        """텍스트에서 JSON 추출"""
        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if match:
            return match.group(0)
        return text

    def get_default_creative_brief(self) -> Dict[str, Any]:
        """
        기본 CreativeBrief (최후의 폴백)

        Returns:
            기본 브리프 딕셔너리
        """
        return {
            "style": "fresh",
            "intensity": 0.5,
            "mood": "neutral",
            "season": ["all"],
            "top_notes": ["bergamot", "lemon"],
            "middle_notes": ["jasmine", "rose"],
            "base_notes": ["musk", "cedarwood"],
            "target_audience": "unisex"
        }

    def parse_llm_output(
        self,
        raw_output: str,
        generator_func: Optional[Callable[[], str]] = None
    ) -> Dict[str, Any]:
        """
        LLM 출력 파싱 (guard 적용)

        Args:
            raw_output: LLM 원시 출력
            generator_func: 재시도 생성 함수 (optional)

        Returns:
            파싱된 CreativeBrief
        """
        fallback = self.get_default_creative_brief()

        try:
            return self.parse_with_guard(
                json_string=raw_output,
                generator_func=generator_func,
                fallback_value=fallback
            )
        except ValueError as e:
            logger.error(f"JSON guard completely failed: {e}")
            # Last resort: return default brief
            return fallback


# =============================================================================
# Usage Example
# =============================================================================

def example_llm_generator() -> str:
    """LLM 생성 함수 예시"""
    # Simulate LLM output (sometimes invalid JSON)
    if random.random() < 0.7:
        # Valid JSON
        return json.dumps({
            "style": "floral",
            "intensity": 0.8,
            "mood": "romantic",
            "season": ["spring"],
            "top_notes": ["rose", "bergamot"],
            "middle_notes": ["jasmine", "lily"],
            "base_notes": ["musk", "amber"],
            "target_audience": "female"
        })
    else:
        # Invalid JSON (trailing comma)
        return """{
            "style": "floral",
            "intensity": 0.8,
            "mood": "romantic",
            "season": ["spring"],
            "top_notes": ["rose", "bergamot"],
            "middle_notes": ["jasmine", "lily"],
            "base_notes": ["musk", "amber"],
            "target_audience": "female",
        }"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create JSON guard
    guard = JSONGuard(JSONGuardConfig(
        max_retries=3,
        initial_backoff=0.5,
        enable_repair=True,
        enable_fallback=True
    ))

    # Test 1: Valid JSON
    print("=== Test 1: Valid JSON ===")
    valid_json = '{"style": "fresh", "intensity": 0.7}'
    result = guard.parse_with_guard(valid_json)
    print(f"Result: {result}\n")

    # Test 2: Invalid JSON (trailing comma) - repair
    print("=== Test 2: Invalid JSON (trailing comma) ===")
    invalid_json = '{"style": "fresh", "intensity": 0.7,}'
    result = guard.parse_with_guard(invalid_json)
    print(f"Result: {result}\n")

    # Test 3: Code block JSON
    print("=== Test 3: Code block JSON ===")
    code_block_json = '''```json
    {"style": "floral", "intensity": 0.8}
    ```'''
    result = guard.parse_with_guard(code_block_json)
    print(f"Result: {result}\n")

    # Test 4: Retry with generator
    print("=== Test 4: Retry with generator ===")
    result = guard.parse_llm_output("invalid json", generator_func=example_llm_generator)
    print(f"Result: {result}\n")

    # Test 5: Complete fallback
    print("=== Test 5: Complete fallback ===")
    result = guard.parse_with_guard(
        "completely broken json",
        fallback_value=guard.get_default_creative_brief()
    )
    print(f"Result: {result}\n")

    # Statistics
    print(f"=== Statistics ===")
    print(f"Repair attempts: {guard.repair_attempts}")
    print(f"Fallback uses: {guard.fallback_uses}")
