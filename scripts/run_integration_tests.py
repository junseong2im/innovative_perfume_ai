#!/usr/bin/env python3
"""
Fragrance AI 시스템 통합 테스트 실행 스크립트
전체 시스템의 기능과 성능을 종합적으로 검증
"""

import asyncio
import sys
import argparse
import logging
import subprocess
import time
import signal
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

# 프로젝트 루트 디렉토리 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.api.main import app
from fragrance_ai.core.config import settings
from tests.integration.test_complete_system import run_integration_tests

logger = logging.getLogger(__name__)

class SystemTestRunner:
    """시스템 테스트 실행기"""

    def __init__(self):
        self.server_process: Optional[subprocess.Popen] = None
        self.server_started = False
        self.test_results: Dict[str, Any] = {}

    async def start_test_server(self, port: int = 8000, timeout: int = 30) -> bool:
        """테스트용 서버 시작"""
        print(f"🚀 테스트 서버 시작 중... (포트: {port})")

        try:
            # 기존 프로세스 종료
            await self._kill_existing_processes(port)

            # 서버 시작
            self.server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "fragrance_ai.api.main:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--log-level", "warning",
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 서버 시작 대기
            for i in range(timeout):
                if self.server_process.poll() is not None:
                    # 프로세스가 종료됨
                    stdout, stderr = self.server_process.communicate()
                    print(f"❌ 서버 시작 실패:")
                    print(f"STDOUT: {stdout.decode()}")
                    print(f"STDERR: {stderr.decode()}")
                    return False

                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"http://localhost:{port}/health")
                        if response.status_code == 200:
                            print(f"✅ 테스트 서버가 성공적으로 시작되었습니다 (포트: {port})")
                            self.server_started = True
                            return True
                except:
                    pass

                await asyncio.sleep(1)
                print(f"⏳ 서버 시작 대기 중... ({i+1}/{timeout})")

            print(f"❌ 서버 시작 시간 초과 ({timeout}초)")
            return False

        except Exception as e:
            print(f"❌ 서버 시작 중 오류 발생: {e}")
            return False

    async def _kill_existing_processes(self, port: int):
        """기존 프로세스 종료"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and any(str(port) in arg for arg in cmdline):
                    if 'uvicorn' in ' '.join(cmdline) or 'fragrance_ai' in ' '.join(cmdline):
                        print(f"🔄 기존 프로세스 종료 중 (PID: {proc.info['pid']})")
                        proc.terminate()
                        proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

    def stop_test_server(self):
        """테스트 서버 종료"""
        if self.server_process and self.server_started:
            print("🛑 테스트 서버 종료 중...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                print("✅ 테스트 서버가 정상적으로 종료되었습니다")
            except subprocess.TimeoutExpired:
                print("⚠️  서버가 정상적으로 종료되지 않아 강제 종료합니다")
                self.server_process.kill()
                self.server_process.wait()
            except Exception as e:
                print(f"❌ 서버 종료 중 오류: {e}")
            finally:
                self.server_started = False

    async def run_system_validation(self) -> Dict[str, Any]:
        """시스템 유효성 검사"""
        print("\n🔍 시스템 유효성 검사 중...")

        validation_results = {
            'environment': True,
            'dependencies': True,
            'configuration': True,
            'database': True,
            'models': True
        }

        try:
            # 1. 환경 검사
            print("  📋 환경 변수 확인...")
            required_vars = ['DATABASE_URL', 'REDIS_URL']
            for var in required_vars:
                if not hasattr(settings, var.lower()) or not getattr(settings, var.lower()):
                    print(f"    ⚠️  {var} 설정이 기본값입니다")

            # 2. 의존성 검사
            print("  📦 필수 의존성 확인...")
            required_packages = [
                'torch', 'transformers', 'sentence-transformers',
                'chromadb', 'fastapi', 'sqlalchemy', 'redis'
            ]

            for package in required_packages:
                try:
                    __import__(package)
                except ImportError as e:
                    print(f"    ❌ {package} 패키지가 없습니다: {e}")
                    validation_results['dependencies'] = False

            # 3. 설정 검사
            print("  ⚙️  설정 유효성 확인...")
            if not settings.secret_key or settings.secret_key == "your-super-secret-key-change-in-production":
                print("    ⚠️  보안키가 기본값입니다. 운영환경에서는 변경하세요")

            # 4. 모델 파일 검사
            print("  🤖 AI 모델 가용성 확인...")
            try:
                from fragrance_ai.models.embedding import AdvancedKoreanFragranceEmbedding
                embedding_model = AdvancedKoreanFragranceEmbedding()
                # 간단한 테스트
                await embedding_model.encode_async(["테스트"])
                print("    ✅ 임베딩 모델 정상")
            except Exception as e:
                print(f"    ❌ 임베딩 모델 오류: {e}")
                validation_results['models'] = False

        except Exception as e:
            print(f"  ❌ 시스템 검증 중 오류: {e}")
            validation_results['environment'] = False

        return validation_results

    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """종합 테스트 실행"""
        print("\n🧪 종합 시스템 테스트 실행 중...")

        start_time = time.time()
        results = {
            'start_time': start_time,
            'integration_tests': False,
            'performance_tests': False,
            'load_tests': False,
            'end_time': None,
            'total_duration': 0,
            'summary': {}
        }

        try:
            # 통합 테스트 실행
            print("\n📋 통합 테스트 실행...")
            integration_success = await run_integration_tests()
            results['integration_tests'] = integration_success

            if integration_success:
                print("✅ 통합 테스트 성공")
            else:
                print("❌ 통합 테스트 실패")

        except Exception as e:
            print(f"❌ 테스트 실행 중 오류: {e}")
            logger.error("Test execution failed", exc_info=True)

        finally:
            end_time = time.time()
            results['end_time'] = end_time
            results['total_duration'] = end_time - start_time

        return results

    def generate_test_report(self, validation_results: Dict[str, Any], test_results: Dict[str, Any]):
        """테스트 리포트 생성"""
        print("\n" + "="*80)
        print("📊 FRAGRANCE AI 시스템 테스트 리포트")
        print("="*80)

        # 시스템 정보
        print(f"테스트 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python 버전: {sys.version.split()[0]}")
        print(f"플랫폼: {sys.platform}")

        # 유효성 검사 결과
        print("\n🔍 시스템 유효성 검사:")
        for category, result in validation_results.items():
            status = "✅ 통과" if result else "❌ 실패"
            print(f"  {category}: {status}")

        # 테스트 결과
        print("\n🧪 테스트 결과:")
        total_tests = 0
        passed_tests = 0

        for test_name, result in test_results.items():
            if test_name in ['start_time', 'end_time', 'total_duration', 'summary']:
                continue

            total_tests += 1
            if result:
                passed_tests += 1
                print(f"  {test_name}: ✅ 성공")
            else:
                print(f"  {test_name}: ❌ 실패")

        # 전체 요약
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\n📈 전체 요약:")
        print(f"  총 테스트: {total_tests}")
        print(f"  성공: {passed_tests}")
        print(f"  실패: {total_tests - passed_tests}")
        print(f"  성공률: {success_rate:.1f}%")
        print(f"  총 소요시간: {test_results.get('total_duration', 0):.1f}초")

        # 최종 판정
        overall_success = all(validation_results.values()) and success_rate >= 90

        print(f"\n{'🎉 시스템이 정상적으로 작동하고 있습니다!' if overall_success else '⚠️  시스템에 문제가 있습니다.'}")

        if not overall_success:
            print("\n🔧 문제 해결 방법:")
            if not all(validation_results.values()):
                print("  - 시스템 유효성 검사 실패 항목을 확인하세요")
            if success_rate < 90:
                print("  - 실패한 테스트의 로그를 확인하세요")
                print("  - 필요한 서비스(PostgreSQL, Redis, ChromaDB)가 실행 중인지 확인하세요")

        print("="*80)

        return overall_success

    async def cleanup(self):
        """리소스 정리"""
        print("\n🧹 리소스 정리 중...")
        self.stop_test_server()


async def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="Fragrance AI 시스템 통합 테스트")
    parser.add_argument('--port', type=int, default=8000, help='테스트 서버 포트')
    parser.add_argument('--timeout', type=int, default=60, help='서버 시작 타임아웃 (초)')
    parser.add_argument('--skip-server', action='store_true', help='서버 시작 건너뛰기')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로그 출력')

    args = parser.parse_args()

    # 로깅 설정
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    runner = SystemTestRunner()

    try:
        print("🧪 Fragrance AI 시스템 테스트 시작")
        print("="*50)

        # 서버 시작 (건너뛰지 않는 경우)
        if not args.skip_server:
            server_started = await runner.start_test_server(
                port=args.port,
                timeout=args.timeout
            )
            if not server_started:
                print("❌ 서버 시작 실패로 테스트를 중단합니다")
                return 1

        # 시스템 유효성 검사
        validation_results = await runner.run_system_validation()

        # 종합 테스트 실행
        test_results = await runner.run_comprehensive_tests()

        # 리포트 생성
        overall_success = runner.generate_test_report(validation_results, test_results)

        return 0 if overall_success else 1

    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 테스트가 중단되었습니다")
        return 130

    except Exception as e:
        print(f"\n❌ 테스트 실행 중 예상치 못한 오류: {e}")
        logger.error("Unexpected error during test execution", exc_info=True)
        return 1

    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  테스트가 중단되었습니다")
        sys.exit(130)