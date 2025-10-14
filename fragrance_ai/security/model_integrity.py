# fragrance_ai/security/model_integrity.py
"""
Model File Integrity Verification
모델 파일/가중치 무결성 - SHA256 체크 + 로딩 시 검증
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Model Checksum Database
# ============================================================================

@dataclass
class ModelChecksum:
    """Model file checksum information"""
    model_name: str
    file_path: str
    sha256: str
    file_size_bytes: int
    last_verified: str
    version: Optional[str] = None
    source: Optional[str] = None  # HuggingFace, local, etc.


class ChecksumDatabase:
    """Database of trusted model checksums"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize checksum database

        Args:
            db_path: Path to checksum JSON database
        """
        if db_path is None:
            # Default to project root
            self.db_path = Path(__file__).parent.parent.parent / "model_checksums.json"
        else:
            self.db_path = Path(db_path)

        self.checksums: Dict[str, ModelChecksum] = {}
        self._load_database()

    def _load_database(self):
        """Load checksums from JSON database"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for key, value in data.items():
                    self.checksums[key] = ModelChecksum(**value)

                logger.info(f"Loaded {len(self.checksums)} model checksums from {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to load checksum database: {e}")
                self.checksums = {}
        else:
            logger.warning(f"Checksum database not found at {self.db_path}, creating new one")
            self.checksums = {}

    def save_database(self):
        """Save checksums to JSON database"""
        try:
            # Create directory if not exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict
            data = {
                key: asdict(checksum)
                for key, checksum in self.checksums.items()
            }

            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.checksums)} model checksums to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save checksum database: {e}")

    def add_checksum(self, checksum: ModelChecksum):
        """Add or update checksum"""
        key = f"{checksum.model_name}:{checksum.file_path}"
        self.checksums[key] = checksum
        logger.info(f"Added checksum for {checksum.model_name} ({checksum.file_path})")

    def get_checksum(self, model_name: str, file_path: str) -> Optional[ModelChecksum]:
        """Get checksum for model file"""
        key = f"{model_name}:{file_path}"
        return self.checksums.get(key)

    def remove_checksum(self, model_name: str, file_path: str):
        """Remove checksum"""
        key = f"{model_name}:{file_path}"
        if key in self.checksums:
            del self.checksums[key]
            logger.info(f"Removed checksum for {model_name} ({file_path})")


# Global checksum database
_checksum_db: Optional[ChecksumDatabase] = None


def get_checksum_database(db_path: Optional[str] = None) -> ChecksumDatabase:
    """Get global checksum database instance"""
    global _checksum_db
    if _checksum_db is None:
        _checksum_db = ChecksumDatabase(db_path)
    return _checksum_db


# ============================================================================
# Hash Calculation
# ============================================================================

def calculate_file_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """
    Calculate SHA256 hash of file

    Args:
        file_path: Path to file
        chunk_size: Bytes to read per chunk

    Returns:
        SHA256 hex digest
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def calculate_directory_checksums(
    directory: str,
    extensions: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Calculate checksums for all files in directory

    Args:
        directory: Directory path
        extensions: File extensions to include (e.g., ['.bin', '.safetensors'])

    Returns:
        Dictionary of {file_path: sha256_hash}
    """
    if extensions is None:
        extensions = ['.bin', '.safetensors', '.pt', '.pth', '.ckpt']

    checksums = {}
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.error(f"Directory not found: {directory}")
        return checksums

    # Walk directory
    for file_path in dir_path.rglob('*'):
        if file_path.is_file() and file_path.suffix in extensions:
            try:
                logger.info(f"Calculating checksum for {file_path.name}...")
                checksum = calculate_file_sha256(str(file_path))
                checksums[str(file_path)] = checksum
                logger.info(f"  {file_path.name}: {checksum[:16]}...")
            except Exception as e:
                logger.error(f"Failed to calculate checksum for {file_path}: {e}")

    return checksums


# ============================================================================
# Verification
# ============================================================================

@dataclass
class VerificationResult:
    """Result of integrity verification"""
    verified: bool
    model_name: str
    file_path: str
    expected_sha256: Optional[str]
    actual_sha256: str
    file_size_bytes: int
    error_message: Optional[str] = None


def verify_model_integrity(
    model_name: str,
    file_path: str,
    expected_checksum: Optional[str] = None,
    auto_register: bool = False
) -> VerificationResult:
    """
    Verify model file integrity

    Args:
        model_name: Name of model
        file_path: Path to model file
        expected_checksum: Expected SHA256 (if None, look up in database)
        auto_register: Automatically register new checksums

    Returns:
        VerificationResult
    """
    file_path_obj = Path(file_path)

    # Check file exists
    if not file_path_obj.exists():
        return VerificationResult(
            verified=False,
            model_name=model_name,
            file_path=file_path,
            expected_sha256=expected_checksum,
            actual_sha256="",
            file_size_bytes=0,
            error_message=f"File not found: {file_path}"
        )

    # Get file size
    file_size = file_path_obj.stat().st_size

    # Calculate actual checksum
    logger.info(f"Verifying {model_name} integrity ({file_size / 1024 / 1024:.1f} MB)...")
    try:
        actual_checksum = calculate_file_sha256(file_path)
    except Exception as e:
        return VerificationResult(
            verified=False,
            model_name=model_name,
            file_path=file_path,
            expected_sha256=expected_checksum,
            actual_sha256="",
            file_size_bytes=file_size,
            error_message=f"Failed to calculate checksum: {e}"
        )

    # Get expected checksum from database if not provided
    if expected_checksum is None:
        db = get_checksum_database()
        checksum_entry = db.get_checksum(model_name, file_path)

        if checksum_entry:
            expected_checksum = checksum_entry.sha256
        elif auto_register:
            # Auto-register new model
            logger.warning(f"Model {model_name} not in database, auto-registering...")
            new_entry = ModelChecksum(
                model_name=model_name,
                file_path=file_path,
                sha256=actual_checksum,
                file_size_bytes=file_size,
                last_verified=datetime.utcnow().isoformat(),
                source="auto_registered"
            )
            db.add_checksum(new_entry)
            db.save_database()

            return VerificationResult(
                verified=True,
                model_name=model_name,
                file_path=file_path,
                expected_sha256=actual_checksum,
                actual_sha256=actual_checksum,
                file_size_bytes=file_size,
                error_message="Auto-registered (no previous checksum)"
            )
        else:
            return VerificationResult(
                verified=False,
                model_name=model_name,
                file_path=file_path,
                expected_sha256=None,
                actual_sha256=actual_checksum,
                file_size_bytes=file_size,
                error_message="No expected checksum in database (set auto_register=True to register)"
            )

    # Verify checksum
    verified = actual_checksum == expected_checksum

    if verified:
        logger.info(f"✓ {model_name} integrity verified: {actual_checksum[:16]}...")

        # Update last_verified timestamp
        db = get_checksum_database()
        checksum_entry = db.get_checksum(model_name, file_path)
        if checksum_entry:
            checksum_entry.last_verified = datetime.utcnow().isoformat()
            db.save_database()
    else:
        logger.error(
            f"✗ {model_name} integrity verification FAILED!\n"
            f"  Expected: {expected_checksum}\n"
            f"  Actual:   {actual_checksum}"
        )

    return VerificationResult(
        verified=verified,
        model_name=model_name,
        file_path=file_path,
        expected_sha256=expected_checksum,
        actual_sha256=actual_checksum,
        file_size_bytes=file_size,
        error_message=None if verified else "Checksum mismatch"
    )


def verify_model_directory(
    model_name: str,
    directory: str,
    extensions: Optional[List[str]] = None,
    auto_register: bool = False
) -> Dict[str, VerificationResult]:
    """
    Verify all model files in directory

    Args:
        model_name: Name of model
        directory: Directory containing model files
        extensions: File extensions to verify
        auto_register: Auto-register new checksums

    Returns:
        Dictionary of {file_path: VerificationResult}
    """
    results = {}

    # Calculate checksums for all files
    checksums = calculate_directory_checksums(directory, extensions)

    # Verify each file
    for file_path, actual_checksum in checksums.items():
        result = verify_model_integrity(
            model_name=model_name,
            file_path=file_path,
            auto_register=auto_register
        )
        results[file_path] = result

    # Summary
    total = len(results)
    verified = sum(1 for r in results.values() if r.verified)
    logger.info(f"Verification summary: {verified}/{total} files verified")

    return results


# ============================================================================
# CLI Tools
# ============================================================================

def register_model_checksum(
    model_name: str,
    file_path: str,
    version: Optional[str] = None,
    source: Optional[str] = None
):
    """
    Register model checksum in database

    Args:
        model_name: Name of model
        file_path: Path to model file
        version: Model version
        source: Source (HuggingFace, local, etc.)
    """
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        logger.error(f"File not found: {file_path}")
        return

    # Calculate checksum
    logger.info(f"Calculating checksum for {model_name}...")
    checksum = calculate_file_sha256(file_path)
    file_size = file_path_obj.stat().st_size

    # Create entry
    entry = ModelChecksum(
        model_name=model_name,
        file_path=file_path,
        sha256=checksum,
        file_size_bytes=file_size,
        last_verified=datetime.utcnow().isoformat(),
        version=version,
        source=source
    )

    # Add to database
    db = get_checksum_database()
    db.add_checksum(entry)
    db.save_database()

    logger.info(f"Registered {model_name}: {checksum[:16]}... ({file_size / 1024 / 1024:.1f} MB)")


def list_registered_models() -> List[ModelChecksum]:
    """
    List all registered models

    Returns:
        List of ModelChecksum entries
    """
    db = get_checksum_database()
    return list(db.checksums.values())


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ModelChecksum',
    'ChecksumDatabase',
    'get_checksum_database',
    'calculate_file_sha256',
    'calculate_directory_checksums',
    'VerificationResult',
    'verify_model_integrity',
    'verify_model_directory',
    'register_model_checksum',
    'list_registered_models'
]
