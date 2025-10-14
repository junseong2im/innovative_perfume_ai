#!/usr/bin/env python3
"""
Runbook: Pin Update
모델 버전 핀 업데이트 자동화
"""

import argparse
import json
import sys
from pathlib import Path
from loguru import logger


# 모델 버전 핀 파일
PIN_FILE = Path(__file__).parent.parent / "configs" / "model_pins.json"


def load_pins() -> dict:
    """
    모델 핀 파일 로드

    Returns:
        dict with model pins
    """
    if not PIN_FILE.exists():
        return {}

    with open(PIN_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pins(pins: dict):
    """
    모델 핀 파일 저장

    Args:
        pins: 모델 핀 딕셔너리
    """
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(PIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(pins, f, indent=2, ensure_ascii=False)

    logger.info(f"Model pins saved to {PIN_FILE}")


def update_pin(model: str, version: str):
    """
    모델 버전 핀 업데이트

    Args:
        model: 모델 이름
        version: 버전 번호
    """
    logger.info(f"Updating {model} pin to version {version}")

    pins = load_pins()

    # 이전 버전 백업
    if model in pins:
        logger.info(f"Previous version: {pins[model]}")

    # 버전 업데이트
    pins[model] = version

    save_pins(pins)

    logger.info(f"[OK] {model} pinned to {version}")


def list_pins():
    """
    현재 모델 핀 목록 출력
    """
    pins = load_pins()

    if not pins:
        logger.warning("No model pins found")
        return

    logger.info("Current model pins:")
    for model, version in pins.items():
        logger.info(f"  {model}: {version}")


def main():
    parser = argparse.ArgumentParser(description="Update model version pins")
    parser.add_argument("--model", help="Model name (qwen, mistral, llama)")
    parser.add_argument("--version", help="Version number")
    parser.add_argument("--list", action="store_true", help="List current pins")

    args = parser.parse_args()

    if args.list:
        list_pins()
    elif args.model and args.version:
        update_pin(args.model, args.version)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
