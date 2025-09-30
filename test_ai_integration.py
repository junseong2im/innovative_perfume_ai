"""
AI 시스템 통합 테스트
DEAP MOGA + RLHF 시스템 검증
"""

import asyncio
import aiohttp
import json
import numpy as np
from typing import Dict, Any, List

BASE_URL = "http://localhost:8001"


async def test_moga_generation():
    """MOGA 향수 생성 테스트"""
    print("\n=== MOGA Generation Test ===")

    request_data = {
        "description": "로맨틱하고 신비로운 가을 저녁의 향기",
        "emotional_keywords": ["romantic", "mysterious", "warm"],
        "fragrance_family": "oriental",
        "season": "fall",
        "occasion": "evening",
        "intensity": 0.7,
        "avoid_notes": ["mint", "eucalyptus"]
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/ai/generate/moga",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ MOGA Generation Success!")
                    print(f"  - Generation ID: {result['generation_id']}")
                    print(f"  - Processing Time: {result['processing_time']:.2f}s")
                    print(f"  - Recipe Name: {result['recipe']['name']}")
                    print(f"  - Top Notes: {len(result['recipe']['composition']['top_notes'])} notes")
                    print(f"  - DNA Length: {len(result['dna'])}")
                    print(f"  - Evaluation: {result['evaluation']}")
                    return result
                else:
                    print(f"❌ MOGA Generation Failed: {response.status}")
                    error = await response.text()
                    print(f"  Error: {error}")
                    return None
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None


async def test_rlhf_evolution(dna: List[float] = None):
    """RLHF 향수 진화 테스트"""
    print("\n=== RLHF Evolution Test ===")

    # 기본 DNA (테스트용)
    if dna is None:
        dna = np.random.random(30).tolist()

    request_data = {
        "current_dna": dna,
        "user_rating": 7.5,
        "feedback_text": "Love the floral notes but needs more depth and longevity",
        "metrics": {
            "harmony": 0.8,
            "longevity": 0.5,
            "sillage": 0.7,
            "creativity": 0.9
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/ai/evolve/rlhf",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ RLHF Evolution Success!")
                    print(f"  - Evolution ID: {result['evolution_id']}")
                    print(f"  - Processing Time: {result['processing_time']:.2f}s")
                    print(f"  - Variations Generated: {len(result['variations'])}")
                    for i, var in enumerate(result['variations']):
                        print(f"    Variation {i+1}: {var['description']}")
                    return result
                else:
                    print(f"❌ RLHF Evolution Failed: {response.status}")
                    error = await response.text()
                    print(f"  Error: {error}")
                    return None
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None


async def test_hybrid_generation():
    """하이브리드 (MOGA+RLHF) 생성 테스트"""
    print("\n=== Hybrid Generation Test ===")

    request_data = {
        "description": "상쾌하면서도 섹시한 여름밤의 향기",
        "emotional_keywords": ["fresh", "sexy", "sophisticated", "clean"],
        "fragrance_family": "citrus",
        "season": "summer",
        "occasion": "evening",
        "intensity": 0.6,
        "avoid_notes": ["patchouli", "musk"]
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{BASE_URL}/api/v1/ai/generate/hybrid",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Hybrid Generation Success!")
                    print(f"  - Generation ID: {result['generation_id']}")
                    print(f"  - Processing Time: {result['processing_time']:.2f}s")
                    print(f"  - AI Method: {result['ai_method']}")
                    print(f"  - Recipe Name: {result['recipe']['name']}")
                    print(f"  - Optimization Steps:")
                    print(f"    MOGA Generations: {result['optimization_steps']['moga_generations']}")
                    print(f"    RLHF Variations: {result['optimization_steps']['rlhf_variations']}")
                    return result
                else:
                    print(f"❌ Hybrid Generation Failed: {response.status}")
                    error = await response.text()
                    print(f"  Error: {error}")
                    return None
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None


async def test_ai_status():
    """AI 시스템 상태 확인"""
    print("\n=== AI System Status ===")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/api/v1/ai/status") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ AI System Status: {result['status']}")
                    print(f"  - Version: {result['version']}")
                    print(f"  - Engines:")
                    for engine, desc in result['engines'].items():
                        print(f"    • {engine}: {desc}")
                    print(f"  - Capabilities:")
                    for cap in result['capabilities']:
                        print(f"    • {cap}")
                    return result
                else:
                    print(f"❌ Status Check Failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            return None


async def test_full_workflow():
    """전체 워크플로우 테스트"""
    print("\n" + "="*60)
    print("     DEAP MOGA + RLHF AI SYSTEM INTEGRATION TEST")
    print("="*60)

    # 1. AI 시스템 상태 확인
    status = await test_ai_status()
    if not status:
        print("\n⚠️  AI System not available. Please start the server first.")
        return

    # 2. MOGA로 초기 향수 생성
    moga_result = await test_moga_generation()

    # 3. RLHF로 진화 (MOGA 결과 DNA 사용)
    if moga_result and 'dna' in moga_result:
        await test_rlhf_evolution(moga_result['dna'])
    else:
        await test_rlhf_evolution()  # 기본 DNA로 테스트

    # 4. 하이브리드 생성 테스트
    await test_hybrid_generation()

    print("\n" + "="*60)
    print("     TEST COMPLETED")
    print("="*60)


async def test_performance():
    """성능 테스트"""
    print("\n=== Performance Test ===")

    # 동시 요청 테스트
    tasks = []

    # 10개의 동시 MOGA 생성
    for i in range(3):
        request_data = {
            "description": f"Test fragrance {i}",
            "emotional_keywords": ["test", f"keyword_{i}"],
            "fragrance_family": "floral",
            "season": "spring",
            "occasion": "daily",
            "intensity": 0.5
        }

        async def make_request(data):
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{BASE_URL}/api/v1/ai/generate/moga",
                        json=data
                    ) as response:
                        return response.status == 200
                except:
                    return False

        tasks.append(make_request(request_data))

    import time
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    success_count = sum(results)
    print(f"✅ Concurrent Requests: {success_count}/{len(tasks)} successful")
    print(f"  - Total Time: {elapsed:.2f}s")
    print(f"  - Average Time per Request: {elapsed/len(tasks):.2f}s")


if __name__ == "__main__":
    # 전체 워크플로우 테스트
    asyncio.run(test_full_workflow())

    # 성능 테스트 (선택적)
    # asyncio.run(test_performance())