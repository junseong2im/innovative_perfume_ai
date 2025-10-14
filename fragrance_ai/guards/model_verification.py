"""
Model Hash Verification - 모델 무결성 검증

SHA256 해시를 사용한 모델 파일 무결성 검증
"""

import hashlib
import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelHashInfo:
    """모델 해시 정보"""
    model_name: str
    model_path: str
    sha256_hash: str
    file_size_bytes: int
    verified: bool
    verified_at: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)


class ModelVerifier:
    """
    모델 검증기

    SHA256 해시를 사용하여 모델 파일의 무결성을 검증합니다.
    - 모델 파일 해시 계산
    - 신뢰할 수 있는 해시와 비교
    - 검증 실패 시 경고
    """

    def __init__(self, trusted_hashes_path: Optional[str] = None):
        """
        Args:
            trusted_hashes_path: 신뢰할 수 있는 해시 목록 파일 경로
        """
        self.trusted_hashes: Dict[str, str] = {}
        self.verification_history: Dict[str, ModelHashInfo] = {}

        if trusted_hashes_path:
            self.load_trusted_hashes(trusted_hashes_path)

    def load_trusted_hashes(self, file_path: str):
        """
        신뢰할 수 있는 해시 목록 로드

        Args:
            file_path: JSON 파일 경로

        Format:
            {
                "qwen-2.5-7b": "abc123...",
                "mistral-7b": "def456...",
                "llama-3-8b": "ghi789..."
            }
        """
        try:
            with open(file_path, 'r') as f:
                self.trusted_hashes = json.load(f)
            logger.info(f"Loaded {len(self.trusted_hashes)} trusted hashes from {file_path}")
        except FileNotFoundError:
            logger.warning(f"Trusted hashes file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse trusted hashes file: {e}")

    def save_trusted_hashes(self, file_path: str):
        """
        신뢰할 수 있는 해시 목록 저장

        Args:
            file_path: JSON 파일 경로
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.trusted_hashes, f, indent=2)
            logger.info(f"Saved {len(self.trusted_hashes)} trusted hashes to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save trusted hashes: {e}")

    def compute_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """
        파일 SHA256 해시 계산

        Args:
            file_path: 파일 경로
            chunk_size: 청크 크기 (bytes)

        Returns:
            SHA256 해시 (hex string)
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    sha256.update(chunk)

            return sha256.hexdigest()

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to compute hash for {file_path}: {e}")

    def verify_model(
        self,
        model_name: str,
        model_path: str,
        expected_hash: Optional[str] = None
    ) -> ModelHashInfo:
        """
        모델 파일 검증

        Args:
            model_name: 모델 이름 (e.g., "qwen", "mistral", "llama")
            model_path: 모델 파일 경로
            expected_hash: 기대 해시 (optional, trusted_hashes에서 조회 가능)

        Returns:
            ModelHashInfo
        """
        # Get expected hash
        if expected_hash is None:
            expected_hash = self.trusted_hashes.get(model_name)

        if expected_hash is None:
            logger.warning(f"No trusted hash found for model: {model_name}")

        # Check file exists
        if not os.path.exists(model_path):
            return ModelHashInfo(
                model_name=model_name,
                model_path=model_path,
                sha256_hash="",
                file_size_bytes=0,
                verified=False,
                verified_at=datetime.now().isoformat(),
                error_message=f"File not found: {model_path}"
            )

        # Get file size
        file_size = os.path.getsize(model_path)

        try:
            # Compute hash
            logger.info(f"Computing hash for {model_name} ({file_size / (1024**3):.2f} GB)...")
            computed_hash = self.compute_file_hash(model_path)

            # Verify
            verified = (expected_hash == computed_hash) if expected_hash else True

            if not verified:
                error_msg = (
                    f"Hash mismatch for {model_name}:\n"
                    f"  Expected: {expected_hash}\n"
                    f"  Computed: {computed_hash}"
                )
                logger.error(error_msg)
            else:
                logger.info(f"Model {model_name} verification: OK")

            hash_info = ModelHashInfo(
                model_name=model_name,
                model_path=model_path,
                sha256_hash=computed_hash,
                file_size_bytes=file_size,
                verified=verified,
                verified_at=datetime.now().isoformat(),
                error_message=error_msg if not verified else None
            )

            # Cache result
            self.verification_history[model_name] = hash_info

            return hash_info

        except Exception as e:
            error_msg = f"Verification failed: {e}"
            logger.error(error_msg)

            return ModelHashInfo(
                model_name=model_name,
                model_path=model_path,
                sha256_hash="",
                file_size_bytes=file_size,
                verified=False,
                verified_at=datetime.now().isoformat(),
                error_message=error_msg
            )

    def verify_all_models(self, model_paths: Dict[str, str]) -> Dict[str, ModelHashInfo]:
        """
        모든 모델 검증

        Args:
            model_paths: {model_name: model_path}

        Returns:
            {model_name: ModelHashInfo}
        """
        results = {}

        for model_name, model_path in model_paths.items():
            logger.info(f"Verifying {model_name}...")
            results[model_name] = self.verify_model(model_name, model_path)

        return results

    def add_trusted_hash(self, model_name: str, sha256_hash: str):
        """
        신뢰할 수 있는 해시 추가

        Args:
            model_name: 모델 이름
            sha256_hash: SHA256 해시
        """
        self.trusted_hashes[model_name] = sha256_hash
        logger.info(f"Added trusted hash for {model_name}")

    def get_verification_history(self, model_name: Optional[str] = None) -> Dict:
        """
        검증 히스토리 조회

        Args:
            model_name: 특정 모델 (optional)

        Returns:
            검증 히스토리
        """
        if model_name:
            return self.verification_history.get(model_name, {})
        return self.verification_history

    def generate_hash_report(self) -> str:
        """
        해시 검증 리포트 생성

        Returns:
            리포트 문자열
        """
        lines = [
            "=" * 80,
            "Model Hash Verification Report",
            "=" * 80,
            ""
        ]

        if not self.verification_history:
            lines.append("No verification history available.")
            return "\n".join(lines)

        for model_name, info in self.verification_history.items():
            status = "[OK]" if info.verified else "[FAIL]"
            file_size_gb = info.file_size_bytes / (1024**3)

            lines.extend([
                f"{status} {model_name}",
                f"  Path:      {info.model_path}",
                f"  Size:      {file_size_gb:.2f} GB",
                f"  Hash:      {info.sha256_hash[:16]}...{info.sha256_hash[-16:]}",
                f"  Verified:  {info.verified_at}",
            ])

            if info.error_message:
                lines.append(f"  Error:     {info.error_message}")

            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


# =============================================================================
# Quick Hash Tool
# =============================================================================

def quick_hash(file_path: str) -> str:
    """
    파일 SHA256 해시 빠르게 계산 (standalone 함수)

    Args:
        file_path: 파일 경로

    Returns:
        SHA256 해시
    """
    verifier = ModelVerifier()
    return verifier.compute_file_hash(file_path)


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create verifier
    verifier = ModelVerifier()

    # Example: Create trusted hashes file
    print("=== Creating Trusted Hashes ===")

    # Note: In production, these would be actual model file hashes
    verifier.add_trusted_hash(
        "qwen-2.5-7b",
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2"
    )
    verifier.add_trusted_hash(
        "mistral-7b",
        "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2a3"
    )
    verifier.add_trusted_hash(
        "llama-3-8b",
        "c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2a3b4"
    )

    # Save trusted hashes
    trusted_hashes_path = "model_hashes.json"
    verifier.save_trusted_hashes(trusted_hashes_path)
    print()

    # Example: Verify models (using dummy paths for demonstration)
    print("=== Verifying Models ===")

    # Create dummy model files for testing
    dummy_model_paths = {}
    for model_name in ["qwen-2.5-7b", "mistral-7b", "llama-3-8b"]:
        dummy_path = f"models/{model_name}.bin"
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)

        # Create dummy file with some content
        with open(dummy_path, 'wb') as f:
            f.write(f"Dummy model content for {model_name}".encode())

        dummy_model_paths[model_name] = dummy_path

    # Verify all models
    results = verifier.verify_all_models(dummy_model_paths)

    print()

    # Generate report
    print(verifier.generate_hash_report())

    # Cleanup
    import shutil
    if os.path.exists("models"):
        shutil.rmtree("models")
    if os.path.exists(trusted_hashes_path):
        os.remove(trusted_hashes_path)

    print("\n=== Quick Hash Tool ===")
    print("Usage: quick_hash('/path/to/model.bin')")
    print("This standalone function can be used to compute SHA256 hash of any file.")
