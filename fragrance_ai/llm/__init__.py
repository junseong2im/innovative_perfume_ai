# fragrance_ai/llm/__init__.py
"""
LLM Ensemble Module
Integrates Qwen, Mistral, and Llama for CreativeBrief generation
"""

import json
import logging
import time
import hashlib
from typing import Literal, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta

from .schemas import CreativeBrief, LLMResponse, DEFAULT_BRIEF
from .llm_router import route_mode, detect_language
from .qwen_client import infer_brief_qwen, get_qwen_client
from .mistral_validator import validate_and_patch
from .llama_hints import generate_creative_hints, get_llama_generator

logger = logging.getLogger(__name__)


# Thread pool for timeout enforcement
_executor = ThreadPoolExecutor(max_workers=3)


# Cache with TTL support
_brief_cache: Dict[str, Dict[str, Any]] = {}
MAX_CACHE_SIZE = 100
CACHE_TTL_SECONDS = 3600  # 1 hour


def _get_cache_key(user_text: str, mode: str) -> str:
    """Generate cache key from user text and mode"""
    combined = f"{user_text}_{mode}"
    return hashlib.md5(combined.encode()).hexdigest()


def _get_from_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get from cache if exists and not expired"""
    if cache_key in _brief_cache:
        cached_data = _brief_cache[cache_key]
        cached_time = cached_data.get("cached_at")

        # Check TTL expiration
        if cached_time:
            age_seconds = (datetime.now() - cached_time).total_seconds()
            if age_seconds > CACHE_TTL_SECONDS:
                logger.info(f"Cache EXPIRED for key {cache_key[:8]}... (age: {age_seconds:.1f}s)")
                del _brief_cache[cache_key]
                return None

        logger.info(f"Cache HIT for key {cache_key[:8]}...")
        return cached_data
    return None


def _save_to_cache(cache_key: str, data: Dict[str, Any]):
    """Save to cache with LRU eviction and TTL"""
    global _brief_cache

    # LRU eviction if cache is full
    if len(_brief_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_brief_cache))
        del _brief_cache[oldest_key]
        logger.debug(f"Cache evicted key {oldest_key[:8]}...")

    # Add timestamp for TTL
    data["cached_at"] = datetime.now()

    _brief_cache[cache_key] = data
    logger.info(f"Cached brief for key {cache_key[:8]}... (TTL: {CACHE_TTL_SECONDS}s)")


def build_brief(
    user_text: str,
    mode: Optional[Literal["fast", "balanced", "creative"]] = None,
    timeout_s: int = 12,
    retry: int = 1,
    use_cache: bool = True
) -> CreativeBrief:
    """
    Build CreativeBrief using LLM ensemble

    Args:
        user_text: User input text
        mode: LLM mode (fast/balanced/creative), auto-detect if None
        timeout_s: Timeout in seconds per LLM call
        retry: Number of retries on failure
        use_cache: Use cache for repeated requests

    Returns:
        CreativeBrief

    Pipeline:
        - fast: Qwen only
        - balanced: Qwen → Mistral validator
        - creative: Qwen → Mistral validator → Llama hints

    Fallback:
        - Qwen fails → Use DEFAULT_BRIEF
        - Mistral fails → Use Qwen result
        - Llama fails → Empty hints
    """
    start_time = time.time()

    # Auto-detect mode if not specified
    if mode is None:
        mode = route_mode(user_text)
        logger.info(f"Auto-detected mode: {mode}")

    # Check cache
    if use_cache:
        cache_key = _get_cache_key(user_text, mode)
        cached = _get_from_cache(cache_key)
        if cached:
            cached["cached"] = True
            cached["processing_time_ms"] = (time.time() - start_time) * 1000
            return CreativeBrief(**cached["brief"])

    models_used = []
    brief_result = None

    # Step 1: Qwen (always)
    logger.info(f"[1/3] Qwen inference...")
    for attempt in range(retry + 1):
        try:
            brief_result = _call_qwen_with_timeout(user_text, timeout_s)
            if brief_result:
                models_used.append("Qwen2.5-7B")
                break
        except Exception as e:
            logger.warning(f"Qwen attempt {attempt + 1} failed: {e}")

    # Fallback if Qwen failed
    if brief_result is None:
        logger.error(f"Qwen failed after {retry + 1} attempts, using DEFAULT_BRIEF")
        brief_result = DEFAULT_BRIEF
        models_used.append("DEFAULT")

    # Step 2: Mistral validator (balanced/creative modes)
    if mode in ["balanced", "creative"]:
        logger.info(f"[2/3] Mistral validation...")
        try:
            brief_result = validate_and_patch(brief_result)
            models_used.append("Mistral-7B-Validator")
        except Exception as e:
            logger.warning(f"Mistral validation failed: {e}, using Qwen result")

    # Step 3: Llama hints (creative mode only)
    if mode == "creative":
        logger.info(f"[3/3] Llama creative hints...")
        try:
            hints = _call_llama_with_timeout(user_text, timeout_s)
            if hints:
                # Merge hints (max 8)
                existing_hints = brief_result.creative_hints or []
                combined = existing_hints + hints
                brief_result.creative_hints = combined[:8]
                models_used.append("Llama-3-8B")
        except Exception as e:
            logger.warning(f"Llama hints generation failed: {e}, skipping hints")

    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000

    # Log result
    _log_brief_generation(brief_result, mode, models_used, processing_time_ms)

    # Save to cache
    if use_cache:
        cache_data = {
            "brief": brief_result.model_dump(),
            "mode_used": mode,
            "processing_time_ms": processing_time_ms,
            "models_used": models_used,
            "cached": False
        }
        _save_to_cache(cache_key, cache_data)

    return brief_result


def _call_qwen_with_timeout(user_text: str, timeout_s: int) -> Optional[CreativeBrief]:
    """Call Qwen with enforced timeout using ThreadPoolExecutor"""
    try:
        # Submit to thread pool with timeout
        future = _executor.submit(infer_brief_qwen, user_text)
        brief = future.result(timeout=timeout_s)
        return brief
    except FuturesTimeoutError:
        logger.error(f"Qwen inference TIMEOUT after {timeout_s}s")
        return None
    except Exception as e:
        logger.error(f"Qwen inference error: {e}")
        return None


def _call_llama_with_timeout(user_text: str, timeout_s: int) -> list:
    """Call Llama with enforced timeout using ThreadPoolExecutor"""
    try:
        # Submit to thread pool with timeout
        future = _executor.submit(generate_creative_hints, user_text)
        hints = future.result(timeout=timeout_s)
        return hints
    except FuturesTimeoutError:
        logger.error(f"Llama hints generation TIMEOUT after {timeout_s}s")
        return []
    except Exception as e:
        logger.error(f"Llama hints error: {e}")
        return []


def _log_brief_generation(
    brief: CreativeBrief,
    mode: str,
    models_used: list,
    processing_time_ms: float
):
    """Log brief generation event"""
    # Calculate sum of notes_preference for validation
    notes_sum = sum(brief.notes_preference.values()) if brief.notes_preference else 0.0

    log_data = {
        "event": "llm_brief",
        "mode": mode,
        "models_used": models_used,
        "processing_time_ms": round(processing_time_ms, 2),
        "language": brief.language,
        "mood_count": len(brief.mood),
        "season_count": len(brief.season),
        "notes_preference_sum": round(notes_sum, 3),
        "creative_hints_count": len(brief.creative_hints),
        "constraints_count": len(brief.constraints)
    }

    logger.info(json.dumps(log_data))


# Initialize models lazily (on first use)
def initialize_models(
    qwen_model: str = "Qwen/Qwen2.5-7B-Instruct",
    llama_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit: bool = False
):
    """
    Initialize LLM models

    Args:
        qwen_model: Qwen model name
        llama_model: Llama model name
        load_in_4bit: Use 4-bit quantization

    Call this function once at startup to load models into memory
    """
    logger.info(f"Initializing LLM ensemble...")

    # Load Qwen
    logger.info(f"Loading Qwen: {qwen_model}")
    get_qwen_client(model_name=qwen_model, load_in_4bit=load_in_4bit)

    # Load Llama (optional, only for creative mode)
    logger.info(f"Loading Llama: {llama_model}")
    get_llama_generator(model_name=llama_model, load_in_4bit=load_in_4bit)

    logger.info(f"LLM ensemble initialized successfully")


__all__ = [
    "build_brief",
    "initialize_models",
    "CreativeBrief",
    "LLMResponse",
    "DEFAULT_BRIEF"
]
