"""
Living Scent Orchestrator 직접 테스트
의존성 문제를 피해서 직접 import
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("Direct Orchestrator Import Test")
print("="*60)

try:
    # 필요한 모듈들만 직접 import
    import json
    import logging
    from typing import Dict, Any, Optional, List
    from dataclasses import asdict
    from datetime import datetime

    # Mock Session for testing
    Session = None

    print("\n1. Testing MOGA import...")
    try:
        from fragrance_ai.training.moga_optimizer import UnifiedProductionMOGA
        print("[OK] MOGA optimizer imported")

        # MOGA 초기화 테스트
        moga = UnifiedProductionMOGA()
        print("[OK] MOGA optimizer initialized")

    except Exception as e:
        print(f"[ERROR] MOGA import failed: {e}")

    print("\n2. Testing PPO import...")
    try:
        from fragrance_ai.training.ppo_engine import PPOTrainer, FragranceEnvironment
        print("[OK] PPO engine imported")

        # PPO 초기화 테스트
        env = FragranceEnvironment()
        ppo = PPOTrainer(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )
        print("[OK] PPO trainer initialized")

    except Exception as e:
        print(f"[ERROR] PPO import failed: {e}")

    print("\n3. Testing Living Scent models...")
    try:
        from fragrance_ai.models.living_scent.linguistic_receptor import get_linguistic_receptor
        from fragrance_ai.models.living_scent.cognitive_core import get_cognitive_core
        from fragrance_ai.models.living_scent.olfactory_recombinator import get_olfactory_recombinator
        from fragrance_ai.models.living_scent.epigenetic_variation import get_epigenetic_variation

        print("[OK] All Living Scent models imported")

        # 초기화 테스트
        receptor = get_linguistic_receptor()
        core = get_cognitive_core()
        recombinator = get_olfactory_recombinator()
        epigenetic = get_epigenetic_variation()

        print("[OK] All Living Scent models initialized")

    except Exception as e:
        print(f"[ERROR] Living Scent models failed: {e}")

    print("\n4. Integration Summary:")
    print("-" * 40)

    # 통합 상태 체크
    components = {
        "MOGA Engine": 'moga' in locals(),
        "PPO Engine": 'ppo' in locals(),
        "Linguistic Receptor": 'receptor' in locals(),
        "Cognitive Core": 'core' in locals(),
        "Olfactory Recombinator": 'recombinator' in locals(),
        "Epigenetic Variation": 'epigenetic' in locals()
    }

    working = sum(1 for v in components.values() if v)
    total = len(components)

    for name, status in components.items():
        status_text = "[OK]" if status else "[FAIL]"
        print(f"  {status_text} {name}")

    print(f"\nIntegration Score: {working}/{total}")

    if working == total:
        print("\nPERFECT! All components ready for integration!")
    elif working >= 4:
        print("\nGood! Most components working.")
    else:
        print("\nWarning: Some components need attention.")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

except Exception as e:
    print(f"\n[FATAL] Test failed: {e}")
    import traceback
    traceback.print_exc()