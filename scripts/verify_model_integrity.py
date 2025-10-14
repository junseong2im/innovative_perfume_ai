"""
Model Weight Integrity Check
가중치 무결성: 해시 검증 로그 1회 저장
"""

import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIRECTORIES = [
    "./models",
    "./checkpoints",
    "./checkpoints_dev",
    "./checkpoints_stg",
    "./checkpoints_prod",
    "/data/models",
    "/data/checkpoints_prod"
]

CHECKPOINT_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.bin"
]

INTEGRITY_LOG_PATH = "./logs/model_integrity_check.json"


# =============================================================================
# Hash Calculation
# =============================================================================

def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """파일 해시 계산 (SHA256)"""
    hash_obj = hashlib.new(algorithm)

    try:
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b''):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash {file_path}: {e}")
        return None


def calculate_directory_hash(directory: Path) -> Dict[str, Any]:
    """디렉토리 내 모든 모델 파일 해시 계산"""
    hashes = {}
    total_size = 0
    file_count = 0

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return {"error": "Directory not found"}

    # Find all checkpoint files
    for pattern in CHECKPOINT_PATTERNS:
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    relative_path = str(file_path.relative_to(directory))
                    file_size = file_path.stat().st_size

                    hashes[relative_path] = {
                        "hash": file_hash,
                        "algorithm": "sha256",
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }

                    total_size += file_size
                    file_count += 1

                    logger.info(f"Hashed: {relative_path} ({hashes[relative_path]['size_mb']} MB)")

    return {
        "directory": str(directory),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "files": hashes
    }


# =============================================================================
# Integrity Verification
# =============================================================================

def verify_against_baseline(current_hash: str, baseline_hash: str) -> bool:
    """베이스라인 해시와 비교"""
    return current_hash == baseline_hash


def load_baseline(baseline_path: Path = None) -> Optional[Dict[str, Any]]:
    """베이스라인 해시 로드"""
    if baseline_path is None:
        baseline_path = Path(INTEGRITY_LOG_PATH)

    if not baseline_path.exists():
        logger.warning(f"No baseline found at {baseline_path}")
        return None

    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load baseline: {e}")
        return None


def save_baseline(integrity_data: Dict[str, Any], output_path: Path = None):
    """베이스라인 해시 저장"""
    if output_path is None:
        output_path = Path(INTEGRITY_LOG_PATH)

    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(integrity_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[SAVED] Baseline saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save baseline: {e}")


def compare_with_baseline(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """현재 해시와 베이스라인 비교"""
    differences = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": []
    }

    current_files = set(current.get("files", {}).keys())
    baseline_files = set(baseline.get("files", {}).keys())

    # Find added files
    for file in current_files - baseline_files:
        differences["added"].append({
            "file": file,
            "current_hash": current["files"][file]["hash"]
        })

    # Find removed files
    for file in baseline_files - current_files:
        differences["removed"].append({
            "file": file,
            "baseline_hash": baseline["files"][file]["hash"]
        })

    # Find modified/unchanged files
    for file in current_files & baseline_files:
        current_hash = current["files"][file]["hash"]
        baseline_hash = baseline["files"][file]["hash"]

        if current_hash != baseline_hash:
            differences["modified"].append({
                "file": file,
                "current_hash": current_hash,
                "baseline_hash": baseline_hash
            })
        else:
            differences["unchanged"].append(file)

    return differences


# =============================================================================
# Integrity Check Report
# =============================================================================

def generate_integrity_report() -> Dict[str, Any]:
    """무결성 검증 보고서 생성"""
    print("=" * 80)
    print("MODEL WEIGHT INTEGRITY CHECK")
    print("=" * 80)
    print()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "directories": [],
        "summary": {
            "total_directories": 0,
            "total_files": 0,
            "total_size_mb": 0.0
        }
    }

    # Check each directory
    for dir_path in MODEL_DIRECTORIES:
        directory = Path(dir_path)

        print(f"Checking directory: {dir_path}")
        print("-" * 80)

        if not directory.exists():
            print(f"[SKIP] Directory not found, skipping")
            print()
            continue

        # Calculate hashes
        dir_result = calculate_directory_hash(directory)

        if "error" in dir_result:
            print(f"[ERROR] {dir_result['error']}")
            print()
            continue

        report["directories"].append(dir_result)
        report["summary"]["total_directories"] += 1
        report["summary"]["total_files"] += dir_result["file_count"]
        report["summary"]["total_size_mb"] += dir_result["total_size_mb"]

        # Print summary
        print(f"Files found: {dir_result['file_count']}")
        print(f"Total size: {dir_result['total_size_mb']:.2f} MB")
        print()

        # Show some files
        if dir_result["files"]:
            print("Sample files:")
            for i, (file, info) in enumerate(list(dir_result["files"].items())[:5]):
                print(f"  {i+1}. {file}")
                print(f"     Hash: {info['hash'][:16]}...")
                print(f"     Size: {info['size_mb']} MB")
                print(f"     Modified: {info['modified']}")

            if len(dir_result["files"]) > 5:
                print(f"  ... and {len(dir_result['files']) - 5} more files")

        print()

    return report


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """메인 실행"""
    print("\n" + "=" * 80)
    print("MODEL WEIGHT INTEGRITY VERIFICATION")
    print("=" * 80)
    print()

    # Step 1: Generate current integrity report
    print("Step 1: Calculating hashes for all model files...")
    print()

    report = generate_integrity_report()

    # Step 2: Load baseline (if exists)
    print("=" * 80)
    print("Step 2: Comparing with baseline")
    print("=" * 80)
    print()

    baseline = load_baseline()

    if baseline is None:
        print("No baseline found. This will be the first baseline.")
        print()

        # Save as baseline
        save_baseline(report)

        print("[SUCCESS] Baseline created successfully")
        print()
    else:
        print(f"Baseline loaded from: {INTEGRITY_LOG_PATH}")
        print(f"Baseline timestamp: {baseline.get('timestamp', 'Unknown')}")
        print()

        # Compare with baseline
        print("Comparing current state with baseline...")
        print()

        for dir_report in report["directories"]:
            dir_path = dir_report["directory"]

            # Find matching baseline directory
            baseline_dir = None
            for bd in baseline.get("directories", []):
                if bd["directory"] == dir_path:
                    baseline_dir = bd
                    break

            if baseline_dir is None:
                print(f"Directory: {dir_path}")
                print("  [NEW] Not in baseline (newly added)")
                print()
                continue

            # Compare
            differences = compare_with_baseline(dir_report, baseline_dir)

            print(f"Directory: {dir_path}")
            print(f"  Added: {len(differences['added'])} files")
            print(f"  Removed: {len(differences['removed'])} files")
            print(f"  Modified: {len(differences['modified'])} files")
            print(f"  Unchanged: {len(differences['unchanged'])} files")

            # Show details
            if differences["added"]:
                print()
                print("  Added files:")
                for item in differences["added"][:3]:
                    print(f"    + {item['file']}")

            if differences["removed"]:
                print()
                print("  Removed files:")
                for item in differences["removed"][:3]:
                    print(f"    - {item['file']}")

            if differences["modified"]:
                print()
                print("  [WARNING] Modified files (INTEGRITY VIOLATION):")
                for item in differences["modified"][:3]:
                    print(f"    ~ {item['file']}")
                    print(f"      Baseline: {item['baseline_hash'][:16]}...")
                    print(f"      Current:  {item['current_hash'][:16]}...")

            print()

        # Option to update baseline
        update_baseline = input("Update baseline with current state? (y/N): ")
        if update_baseline.lower() == 'y':
            save_baseline(report)
            print("[SUCCESS] Baseline updated")
        else:
            print("Baseline not updated")

    # Step 3: Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Directories scanned: {report['summary']['total_directories']}")
    print(f"Model files found: {report['summary']['total_files']}")
    print(f"Total size: {report['summary']['total_size_mb']:.2f} MB")
    print()

    if report['summary']['total_files'] > 0:
        print("[SUCCESS] INTEGRITY CHECK COMPLETED")
        print()
        print("Integrity log saved to:")
        print(f"  {INTEGRITY_LOG_PATH}")
    else:
        print("[WARNING] No model files found")
        print()
        print("Searched in:")
        for dir_path in MODEL_DIRECTORIES:
            print(f"  - {dir_path}")

    print()


if __name__ == "__main__":
    main()
