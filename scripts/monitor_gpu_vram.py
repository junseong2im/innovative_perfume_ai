"""
GPU/VRAM Monitoring Script
세 모델 동시 로딩 시 여유 VRAM 20% 이상 유지
"""

import subprocess
import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

VRAM_HEADROOM_PERCENT = 20  # 20% minimum free VRAM
VRAM_WARNING_THRESHOLD = 25  # Warning at 25% free
CHECK_INTERVAL_SECONDS = 10

# Expected VRAM usage per model (MB)
MODEL_VRAM_USAGE = {
    "qwen_32b": 16000,    # ~16GB for Qwen 32B (4-bit quantized)
    "mistral_7b": 4000,   # ~4GB for Mistral 7B (4-bit quantized)
    "llama3_8b": 5000     # ~5GB for Llama 3 8B (4-bit quantized)
}


# =============================================================================
# GPU Query Functions
# =============================================================================

def get_gpu_info() -> List[Dict]:
    """
    Query GPU information using nvidia-smi

    Returns:
        List of GPU info dicts with memory and utilization
    """
    try:
        # Query nvidia-smi in JSON format
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 7:
                continue

            gpu_info = {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
                "utilization_percent": int(parts[5]),
                "temperature_c": int(parts[6]),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Calculate percentages
            total_mb = gpu_info["memory_total_mb"]
            used_mb = gpu_info["memory_used_mb"]
            free_mb = gpu_info["memory_free_mb"]

            gpu_info["memory_used_percent"] = (used_mb / total_mb) * 100
            gpu_info["memory_free_percent"] = (free_mb / total_mb) * 100

            gpus.append(gpu_info)

        return gpus

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to query nvidia-smi: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing GPU info: {e}")
        return []


def check_vram_headroom(gpu_info: Dict) -> Dict:
    """
    Check if GPU has sufficient VRAM headroom

    Args:
        gpu_info: GPU info dict from get_gpu_info()

    Returns:
        Dict with status and recommendation
    """
    free_percent = gpu_info["memory_free_percent"]
    free_mb = gpu_info["memory_free_mb"]

    status = {
        "gpu_index": gpu_info["index"],
        "free_percent": round(free_percent, 2),
        "free_mb": free_mb,
        "threshold_percent": VRAM_HEADROOM_PERCENT,
        "sufficient": free_percent >= VRAM_HEADROOM_PERCENT,
        "warning": free_percent < VRAM_WARNING_THRESHOLD,
        "critical": free_percent < VRAM_HEADROOM_PERCENT,
        "recommendation": None
    }

    # Generate recommendations
    if status["critical"]:
        status["recommendation"] = "CRITICAL: Unload one model or reduce batch size"
    elif status["warning"]:
        status["recommendation"] = "WARNING: Monitor closely, consider reducing load"
    else:
        status["recommendation"] = "OK: Sufficient VRAM headroom"

    return status


def can_load_models(gpu_info: Dict, models: List[str]) -> Dict:
    """
    Check if GPU can load specified models with 20% headroom

    Args:
        gpu_info: GPU info dict
        models: List of model names (e.g., ["qwen_32b", "mistral_7b"])

    Returns:
        Dict with load feasibility analysis
    """
    free_mb = gpu_info["memory_free_mb"]
    total_mb = gpu_info["memory_total_mb"]

    # Calculate required VRAM for models
    required_mb = sum(MODEL_VRAM_USAGE.get(model, 0) for model in models)

    # Calculate headroom requirement (20% of total)
    headroom_mb = total_mb * (VRAM_HEADROOM_PERCENT / 100)

    # Total needed = models + headroom
    total_needed_mb = required_mb + headroom_mb

    can_load = free_mb >= total_needed_mb

    return {
        "gpu_index": gpu_info["index"],
        "can_load": can_load,
        "free_mb": free_mb,
        "required_mb": required_mb,
        "headroom_mb": round(headroom_mb, 2),
        "total_needed_mb": round(total_needed_mb, 2),
        "remaining_mb": round(free_mb - total_needed_mb, 2) if can_load else 0,
        "models": models,
        "recommendation": "OK to load all models" if can_load else f"Insufficient VRAM: need {total_needed_mb - free_mb:.0f} MB more"
    }


# =============================================================================
# Model Load Planning
# =============================================================================

def suggest_model_placement(gpus: List[Dict]) -> Dict:
    """
    Suggest which models to load on which GPUs

    Args:
        gpus: List of GPU info dicts

    Returns:
        Placement strategy
    """
    if not gpus:
        return {"error": "No GPUs available"}

    # Strategy: Load all three models on GPU 0 if possible
    gpu0 = gpus[0]
    all_models = ["qwen_32b", "mistral_7b", "llama3_8b"]

    feasibility = can_load_models(gpu0, all_models)

    if feasibility["can_load"]:
        return {
            "strategy": "single_gpu",
            "gpu_0": {
                "models": all_models,
                "vram_used_mb": feasibility["required_mb"],
                "vram_free_mb": feasibility["remaining_mb"],
                "status": "OK"
            }
        }

    # Fallback: Split models across GPUs
    if len(gpus) >= 2:
        # GPU 0: Qwen (largest)
        # GPU 1: Mistral + Llama
        gpu1 = gpus[1]

        qwen_feasibility = can_load_models(gpu0, ["qwen_32b"])
        others_feasibility = can_load_models(gpu1, ["mistral_7b", "llama3_8b"])

        if qwen_feasibility["can_load"] and others_feasibility["can_load"]:
            return {
                "strategy": "multi_gpu",
                "gpu_0": {
                    "models": ["qwen_32b"],
                    "vram_used_mb": qwen_feasibility["required_mb"],
                    "vram_free_mb": qwen_feasibility["remaining_mb"],
                    "status": "OK"
                },
                "gpu_1": {
                    "models": ["mistral_7b", "llama3_8b"],
                    "vram_used_mb": others_feasibility["required_mb"],
                    "vram_free_mb": others_feasibility["remaining_mb"],
                    "status": "OK"
                }
            }

    # Cannot fit all models
    return {
        "strategy": "insufficient_vram",
        "recommendation": "Use model offloading or increase GPU memory",
        "alternatives": [
            "Load only 2 models (Qwen + Mistral) for Creative + Balanced modes",
            "Use CPU offloading for least-used model",
            "Upgrade GPU VRAM"
        ]
    }


# =============================================================================
# Monitoring Loop
# =============================================================================

def monitor_loop(duration_seconds: Optional[int] = None):
    """
    Continuous monitoring loop

    Args:
        duration_seconds: How long to monitor (None = forever)
    """
    start_time = time.time()
    iteration = 0

    print("=" * 80)
    print("GPU/VRAM MONITORING")
    print(f"Headroom requirement: {VRAM_HEADROOM_PERCENT}% minimum")
    print(f"Warning threshold: {VRAM_WARNING_THRESHOLD}%")
    print("=" * 80)
    print()

    try:
        while True:
            iteration += 1

            # Check if duration exceeded
            if duration_seconds and (time.time() - start_time) > duration_seconds:
                logger.info("Monitoring duration completed")
                break

            # Query GPUs
            gpus = get_gpu_info()

            if not gpus:
                logger.warning("No GPU information available")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # Print status
            print(f"[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 80)

            for gpu in gpus:
                headroom = check_vram_headroom(gpu)

                status_symbol = "[OK]" if headroom["sufficient"] else "[CRITICAL]"
                if headroom["warning"] and not headroom["critical"]:
                    status_symbol = "[WARNING]"

                print(f"GPU {gpu['index']}: {gpu['name']}")
                print(f"  Memory: {gpu['memory_used_mb']} / {gpu['memory_total_mb']} MB "
                      f"({gpu['memory_used_percent']:.1f}% used, {headroom['free_percent']:.1f}% free)")
                print(f"  Utilization: {gpu['utilization_percent']}%")
                print(f"  Temperature: {gpu['temperature_c']}°C")
                print(f"  Status: {status_symbol} {headroom['recommendation']}")
                print()

            # Check if all three models can be loaded
            if gpus:
                placement = suggest_model_placement(gpus)

                if placement.get("strategy") == "single_gpu":
                    print("[PLACEMENT] All three models can fit on GPU 0 with 20% headroom")
                elif placement.get("strategy") == "multi_gpu":
                    print("[PLACEMENT] Models should be split across GPUs:")
                    print(f"  GPU 0: {', '.join(placement['gpu_0']['models'])}")
                    if "gpu_1" in placement:
                        print(f"  GPU 1: {', '.join(placement['gpu_1']['models'])}")
                elif placement.get("strategy") == "insufficient_vram":
                    print("[PLACEMENT] WARNING: Insufficient VRAM for all models")
                    print(f"  Recommendation: {placement.get('recommendation')}")

                print()

            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print()
        logger.info("Monitoring stopped by user")


# =============================================================================
# One-time Check
# =============================================================================

def check_once():
    """Single GPU check and report"""
    print("=" * 80)
    print("GPU/VRAM STATUS CHECK")
    print("=" * 80)
    print()

    gpus = get_gpu_info()

    if not gpus:
        print("[ERROR] No GPUs detected")
        print()
        print("Please ensure:")
        print("  - NVIDIA drivers are installed")
        print("  - nvidia-smi is in PATH")
        print("  - GPU is accessible")
        return

    # Display GPU info
    for gpu in gpus:
        headroom = check_vram_headroom(gpu)

        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  Total VRAM: {gpu['memory_total_mb']} MB")
        print(f"  Used: {gpu['memory_used_mb']} MB ({gpu['memory_used_percent']:.1f}%)")
        print(f"  Free: {gpu['memory_free_mb']} MB ({headroom['free_percent']:.1f}%)")
        print(f"  Utilization: {gpu['utilization_percent']}%")
        print(f"  Temperature: {gpu['temperature_c']}°C")
        print()

        # Headroom check
        if headroom["sufficient"]:
            print(f"  [PASS] Sufficient VRAM headroom ({headroom['free_percent']:.1f}% >= {VRAM_HEADROOM_PERCENT}%)")
        else:
            print(f"  [FAIL] Insufficient VRAM headroom ({headroom['free_percent']:.1f}% < {VRAM_HEADROOM_PERCENT}%)")

        print(f"  Recommendation: {headroom['recommendation']}")
        print()

    # Model placement suggestion
    print("=" * 80)
    print("MODEL PLACEMENT ANALYSIS")
    print("=" * 80)
    print()

    placement = suggest_model_placement(gpus)

    if placement.get("strategy") == "single_gpu":
        print("[STRATEGY] Single GPU Deployment")
        print()
        print(f"GPU 0 can accommodate all three models:")
        for model in placement["gpu_0"]["models"]:
            vram_mb = MODEL_VRAM_USAGE.get(model, 0)
            print(f"  - {model}: ~{vram_mb} MB")
        print()
        print(f"Total VRAM required: {placement['gpu_0']['vram_used_mb']} MB")
        print(f"Free after loading: {placement['gpu_0']['vram_free_mb']:.0f} MB")
        print(f"Status: {placement['gpu_0']['status']}")

    elif placement.get("strategy") == "multi_gpu":
        print("[STRATEGY] Multi-GPU Deployment")
        print()
        print(f"GPU 0: {', '.join(placement['gpu_0']['models'])}")
        print(f"  VRAM required: {placement['gpu_0']['vram_used_mb']} MB")
        print(f"  Free after loading: {placement['gpu_0']['vram_free_mb']:.0f} MB")
        print()
        print(f"GPU 1: {', '.join(placement['gpu_1']['models'])}")
        print(f"  VRAM required: {placement['gpu_1']['vram_used_mb']} MB")
        print(f"  Free after loading: {placement['gpu_1']['vram_free_mb']:.0f} MB")

    else:
        print("[STRATEGY] Insufficient VRAM")
        print()
        print(f"Recommendation: {placement.get('recommendation')}")
        print()
        print("Alternatives:")
        for alt in placement.get("alternatives", []):
            print(f"  - {alt}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU/VRAM monitoring for LLM deployment")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds")

    args = parser.parse_args()

    if args.once:
        check_once()
    else:
        monitor_loop(duration_seconds=args.duration)
