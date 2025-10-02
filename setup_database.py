"""
데이터베이스 설정 및 초기화 스크립트
PostgreSQL + pgvector 설정
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.database.init_db import DatabaseInitializer


def check_postgresql():
    """PostgreSQL 설치 확인"""
    try:
        result = subprocess.run(
            ["psql", "--version"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            print(f"✓ PostgreSQL 설치 확인: {result.stdout.strip()}")
            return True
        else:
            print("✗ PostgreSQL이 설치되지 않음")
            return False
    except Exception as e:
        print(f"✗ PostgreSQL 확인 실패: {e}")
        return False


def setup_database(reset=False):
    """데이터베이스 설정"""
    print("\n" + "="*60)
    print("데이터베이스 설정 시작")
    print("="*60)

    # 1. PostgreSQL 확인
    if not check_postgresql():
        print("\nPostgreSQL 설치가 필요합니다:")
        print("- Windows: https://www.postgresql.org/download/windows/")
        print("- Mac: brew install postgresql")
        print("- Linux: sudo apt-get install postgresql")
        return False

    # 2. 데이터베이스 URL 설정
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/fragrance_ai"
    )
    print(f"\n데이터베이스 URL: {db_url}")

    # 3. 데이터베이스 초기화
    initializer = DatabaseInitializer(db_url)

    # 4. 초기화 실행
    print("\n데이터베이스 초기화 시작...")
    initializer.initialize()
    print("✓ 데이터베이스 초기화 완료")

    print("\n" + "="*60)
    print("데이터베이스 설정 완료!")
    print("="*60)

    return True


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="데이터베이스 설정")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터 삭제 후 재생성"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="데이터베이스 URL 지정"
    )

    args = parser.parse_args()

    if args.url:
        os.environ["DATABASE_URL"] = args.url

    success = setup_database(reset=args.reset)

    if success:
        print("\n✅ 데이터베이스 설정 성공!")
        print("\n다음 단계:")
        print("1. API 서버 시작: uvicorn fragrance_ai.api.main:app --reload")
        print("2. 테스트 실행: python test_real_ai_final.py")
    else:
        print("\n❌ 데이터베이스 설정 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()