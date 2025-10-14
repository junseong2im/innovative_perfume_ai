#!/usr/bin/env python3
"""
Release Tagging Script
SemVer + 모델(정책/가치) 스냅샷 버전 태그
"""

import os
import sys
import json
import argparse
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


class ReleaseTag:
    """릴리스 태그 정보"""

    def __init__(
        self,
        version: str,
        model_snapshot: str,
        checkpoint_hash: str,
        release_notes: str = "",
        timestamp: Optional[str] = None
    ):
        self.version = version  # e.g., "v2.1.0"
        self.model_snapshot = model_snapshot  # e.g., "model-20251014-abc123"
        self.checkpoint_hash = checkpoint_hash  # SHA256 hash
        self.release_notes = release_notes
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "model_snapshot": self.model_snapshot,
            "checkpoint_hash": self.checkpoint_hash,
            "release_notes": self.release_notes,
            "timestamp": self.timestamp
        }

    def to_tag_message(self) -> str:
        """Git tag 메시지 생성"""
        return f"""Release {self.version}

Model Snapshot: {self.model_snapshot}
Checkpoint Hash: {self.checkpoint_hash}
Timestamp: {self.timestamp}

{self.release_notes}
"""


class ModelSnapshotManager:
    """모델 체크포인트 스냅샷 관리"""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)

    def find_latest_checkpoint(self) -> Optional[Path]:
        """최신 체크포인트 파일 찾기"""
        if not self.checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory not found: {self.checkpoint_dir}")
            return None

        # Find all checkpoint files
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints += list(self.checkpoint_dir.glob("*.pth"))
        checkpoints += list(self.checkpoint_dir.glob("*.ckpt"))

        if not checkpoints:
            print("Warning: No checkpoint files found")
            return None

        # Sort by modification time (latest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def calculate_checkpoint_hash(self, checkpoint_path: Path) -> str:
        """체크포인트 파일의 SHA256 해시 계산"""
        sha256_hash = hashlib.sha256()

        with open(checkpoint_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def create_snapshot(self, version: str) -> Tuple[str, str]:
        """
        모델 스냅샷 생성

        Returns:
            (snapshot_id, checkpoint_hash)
        """
        latest_checkpoint = self.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint found - use dummy hash
            snapshot_id = f"model-{datetime.now().strftime('%Y%m%d')}-no-checkpoint"
            checkpoint_hash = "0" * 64
            print(f"Warning: No checkpoint found, using placeholder")
            return snapshot_id, checkpoint_hash

        # Calculate hash
        checkpoint_hash = self.calculate_checkpoint_hash(latest_checkpoint)
        short_hash = checkpoint_hash[:8]

        # Create snapshot ID
        timestamp = datetime.now().strftime("%Y%m%d")
        snapshot_id = f"model-{timestamp}-{short_hash}"

        # Copy checkpoint to snapshots directory
        snapshot_dir = self.checkpoint_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)

        snapshot_path = snapshot_dir / f"{snapshot_id}.pt"

        # Copy file
        import shutil
        shutil.copy2(latest_checkpoint, snapshot_path)

        print(f"Created model snapshot: {snapshot_id}")
        print(f"  Source: {latest_checkpoint}")
        print(f"  Snapshot: {snapshot_path}")
        print(f"  Hash: {checkpoint_hash[:16]}...")

        return snapshot_id, checkpoint_hash


class ReleaseTagManager:
    """릴리스 태그 관리"""

    def __init__(self, release_file: str = "RELEASES.json"):
        self.release_file = Path(release_file)
        self.releases = self._load_releases()

    def _load_releases(self) -> Dict[str, Dict]:
        """릴리스 기록 로드"""
        if not self.release_file.exists():
            return {}

        with open(self.release_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_releases(self):
        """릴리스 기록 저장"""
        with open(self.release_file, 'w', encoding='utf-8') as f:
            json.dump(self.releases, f, indent=2, ensure_ascii=False)

    def get_latest_version(self) -> Optional[str]:
        """최신 버전 조회"""
        if not self.releases:
            return None

        versions = list(self.releases.keys())
        versions.sort(reverse=True)
        return versions[0]

    def suggest_next_version(self, bump_type: str = "minor") -> str:
        """다음 버전 제안 (SemVer)"""
        latest = self.get_latest_version()

        if latest is None:
            return "v1.0.0"

        # Parse version (e.g., "v2.1.3" -> [2, 1, 3])
        version_str = latest.lstrip("v")
        parts = [int(x) for x in version_str.split(".")]

        if bump_type == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif bump_type == "minor":
            parts[1] += 1
            parts[2] = 0
        elif bump_type == "patch":
            parts[2] += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")

        return f"v{parts[0]}.{parts[1]}.{parts[2]}"

    def create_release(
        self,
        version: str,
        model_snapshot: str,
        checkpoint_hash: str,
        release_notes: str = ""
    ) -> ReleaseTag:
        """릴리스 생성"""
        release = ReleaseTag(
            version=version,
            model_snapshot=model_snapshot,
            checkpoint_hash=checkpoint_hash,
            release_notes=release_notes
        )

        # Save to releases file
        self.releases[version] = release.to_dict()
        self._save_releases()

        print(f"Release record created: {version}")

        return release

    def create_git_tag(self, release: ReleaseTag, push: bool = False):
        """Git 태그 생성"""
        tag_name = release.version
        tag_message = release.to_tag_message()

        # Create annotated tag
        try:
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", tag_message],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Git tag created: {tag_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating git tag: {e.stderr}")
            sys.exit(1)

        # Push tag if requested
        if push:
            try:
                subprocess.run(
                    ["git", "push", "origin", tag_name],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Git tag pushed to remote: {tag_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error pushing git tag: {e.stderr}")
                sys.exit(1)

    def list_releases(self):
        """릴리스 목록 출력"""
        print("=" * 80)
        print("Release History")
        print("=" * 80)
        print()

        if not self.releases:
            print("No releases found")
            return

        versions = sorted(self.releases.keys(), reverse=True)

        for version in versions:
            release_data = self.releases[version]
            print(f"Version: {version}")
            print(f"  Model Snapshot: {release_data['model_snapshot']}")
            print(f"  Checkpoint Hash: {release_data['checkpoint_hash'][:16]}...")
            print(f"  Timestamp: {release_data['timestamp']}")
            if release_data.get('release_notes'):
                print(f"  Notes: {release_data['release_notes'][:50]}...")
            print()


def main():
    parser = argparse.ArgumentParser(description="Release Tagging with Model Snapshots")
    parser.add_argument(
        "--version",
        help="Release version (e.g., v2.1.0). If not provided, auto-increment."
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default="minor",
        help="Version bump type (default: minor)"
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Release notes"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push git tag to remote"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all releases and exit"
    )

    args = parser.parse_args()

    # Initialize managers
    model_manager = ModelSnapshotManager(checkpoint_dir=args.checkpoint_dir)
    release_manager = ReleaseTagManager()

    # List releases if requested
    if args.list:
        release_manager.list_releases()
        sys.exit(0)

    # Determine version
    if args.version:
        version = args.version
        if not version.startswith("v"):
            version = f"v{version}"
    else:
        version = release_manager.suggest_next_version(bump_type=args.bump)
        print(f"Auto-generated version: {version}")

    # Check if version already exists
    if version in release_manager.releases:
        print(f"Error: Version {version} already exists")
        sys.exit(1)

    print()
    print("=" * 80)
    print(f"Creating Release: {version}")
    print("=" * 80)
    print()

    # Create model snapshot
    print("Step 1: Creating model snapshot...")
    snapshot_id, checkpoint_hash = model_manager.create_snapshot(version)
    print()

    # Create release record
    print("Step 2: Creating release record...")
    release = release_manager.create_release(
        version=version,
        model_snapshot=snapshot_id,
        checkpoint_hash=checkpoint_hash,
        release_notes=args.notes
    )
    print()

    # Create git tag
    print("Step 3: Creating git tag...")
    release_manager.create_git_tag(release, push=args.push)
    print()

    # Summary
    print("=" * 80)
    print("Release Created Successfully")
    print("=" * 80)
    print(f"Version:         {release.version}")
    print(f"Model Snapshot:  {release.model_snapshot}")
    print(f"Checkpoint Hash: {release.checkpoint_hash[:16]}...")
    print(f"Timestamp:       {release.timestamp}")
    print()

    if not args.push:
        print("To push the tag to remote:")
        print(f"  git push origin {version}")
        print()


if __name__ == "__main__":
    main()
