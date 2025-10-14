#!/usr/bin/env python3
"""
Runbook: Downshift
트래픽을 점진적으로 감소시켜 불안정한 서비스의 부하 감소
"""

import argparse
import subprocess
import sys
from loguru import logger


def downshift_traffic(target: str, reduce_to: int):
    """
    트래픽 감소

    Args:
        target: 타겟 서비스
        reduce_to: 감소할 트래픽 비율 (%)
    """
    logger.info(f"Downshifting {target} traffic to {reduce_to}%")

    # Nginx 가중치 업데이트
    try:
        subprocess.run([
            "bash", "scripts/update_nginx_weights.sh",
            "--stable", str(100 - reduce_to),
            "--canary", str(reduce_to)
        ], check=True)

        logger.info(f"[OK] Traffic reduced to {reduce_to}%")

    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to downshift: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Downshift traffic to service")
    parser.add_argument("--target", required=True, help="Target service")
    parser.add_argument("--reduce-to", type=int, default=50, help="Reduce to percentage")

    args = parser.parse_args()

    downshift_traffic(args.target, args.reduce_to)


if __name__ == "__main__":
    main()
