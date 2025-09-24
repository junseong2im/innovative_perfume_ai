"""
LLM 설정 및 다운로드 스크립트
- 필요한 패키지 설치 확인
- 모델 다운로드
- Ollama 설정
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMSetup:
    """LLM 설정 도구"""

    def __init__(self):
        """초기화"""
        self.system = platform.system()
        self.config_path = Path("configs/local.json")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    def check_python_packages(self) -> bool:
        """Python 패키지 확인"""
        required_packages = {
            "transformers": "transformers>=4.36.0",
            "accelerate": "accelerate>=0.25.0",
            "bitsandbytes": "bitsandbytes>=0.41.0",
            "torch": "torch>=2.0.0",
            "sentencepiece": "sentencepiece",
            "protobuf": "protobuf",
            "aiohttp": "aiohttp"
        }

        missing_packages = []

        for package, install_name in required_packages.items():
            try:
                __import__(package)
                logger.info(f"✓ {package} installed")
            except ImportError:
                logger.warning(f"✗ {package} not installed")
                missing_packages.append(install_name)

        if missing_packages:
            logger.info("\n필요한 패키지를 설치합니다...")
            return self.install_packages(missing_packages)

        return True

    def install_packages(self, packages: list) -> bool:
        """패키지 설치"""
        try:
            # CUDA 사용 가능 여부 확인
            cuda_available = self.check_cuda()

            if cuda_available:
                logger.info("CUDA detected. Installing PyTorch with CUDA support...")
                # CUDA 버전용 PyTorch 설치
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ])
            else:
                logger.info("CUDA not detected. Installing CPU-only PyTorch...")
                # CPU 버전 PyTorch 설치
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio"
                ])

            # 나머지 패키지 설치
            for package in packages:
                if "torch" not in package:  # torch는 이미 설치함
                    logger.info(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

            logger.info("✓ All packages installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Package installation failed: {e}")
            return False

    def check_cuda(self) -> bool:
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            # nvidia-smi 명령으로 확인
            try:
                subprocess.check_output(['nvidia-smi'])
                return True
            except:
                return False

    def setup_ollama(self) -> bool:
        """Ollama 설정"""
        logger.info("\n=== Ollama 설정 ===")

        # Ollama 설치 확인
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            logger.info(f"✓ Ollama installed: {result.stdout.strip()}")

            # Ollama 서버 실행 확인
            import requests
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    logger.info("✓ Ollama server is running")

                    # 모델 리스트 확인
                    models = response.json().get('models', [])
                    if models:
                        logger.info(f"Available models: {[m['name'] for m in models]}")
                    else:
                        logger.info("No models installed yet")

                    # 필요한 모델들 다운로드
                    required_models = [
                        'llama3:8b-instruct-q4_K_M',  # Artisan orchestrator
                        'mistral:7b-instruct-q4_K_M',  # Customer service
                        'qwen:32b'  # Perfume description interpreter (or qwen:14b for smaller systems)
                    ]

                    for model in required_models:
                        model_base = model.split(':')[0]
                        if not any(model_base in m['name'] for m in models):
                            logger.info(f"Downloading {model}...")
                            subprocess.run(['ollama', 'pull', model])
                            logger.info(f"✓ {model} downloaded")
                        else:
                            logger.info(f"✓ {model_base} already available")

                    return True
            except:
                logger.warning("Ollama server not running. Please start it with: ollama serve")
                return False

        except FileNotFoundError:
            logger.warning("Ollama not installed. Please install from: https://ollama.ai")

            # 플랫폼별 설치 가이드
            if self.system == "Linux":
                logger.info("\nLinux installation:")
                logger.info("curl -fsSL https://ollama.ai/install.sh | sh")
            elif self.system == "Darwin":  # macOS
                logger.info("\nmacOS installation:")
                logger.info("brew install ollama")
            elif self.system == "Windows":
                logger.info("\nWindows installation:")
                logger.info("Download from: https://ollama.ai/download/windows")

            return False

    def download_korean_model(self) -> bool:
        """한국어 모델 다운로드"""
        logger.info("\n=== 한국어 모델 다운로드 ===")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "beomi/KoAlpaca-Polyglot-5.8B"
            cache_dir = "./models/cache"

            logger.info(f"Downloading {model_name}...")
            logger.info("This may take a while (model size: ~10GB)...")

            # 토크나이저 다운로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            logger.info("✓ Tokenizer downloaded")

            # 모델 다운로드 (4-bit 양자화)
            if self.check_cuda():
                logger.info("Downloading with 4-bit quantization for GPU...")
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            else:
                logger.info("Downloading for CPU (may be slow)...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )

            logger.info("✓ Korean model downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            logger.info("The model will be downloaded automatically when first used")
            return False

    def verify_setup(self) -> bool:
        """설정 검증"""
        logger.info("\n=== 설정 검증 ===")

        # 설정 파일 확인
        if self.config_path.exists():
            logger.info("✓ Configuration file exists")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Ollama 설정 확인
            ollama_config = config.get('llm_orchestrator', {})
            if ollama_config.get('provider') == 'ollama':
                logger.info(f"✓ Ollama configured: {ollama_config.get('model_name_or_path')}")

            # Transformers 설정 확인
            trans_config = config.get('llm_specialist_generator', {})
            if trans_config.get('provider') == 'transformers':
                logger.info(f"✓ Transformers configured: {trans_config.get('model_name_or_path')}")

        else:
            logger.warning("✗ Configuration file not found")
            return False

        # 간단한 테스트
        logger.info("\n=== Testing LLM Components ===")

        # Ollama 테스트
        try:
            from fragrance_ai.llm.ollama_client import OllamaClient
            client = OllamaClient()

            import asyncio
            is_available = asyncio.run(client.check_availability())

            if is_available:
                logger.info("✓ Ollama client working")
            else:
                logger.warning("✗ Ollama not available")

        except Exception as e:
            logger.error(f"Ollama test failed: {e}")

        # Transformers 테스트
        try:
            from fragrance_ai.llm.transformers_loader import check_transformers_availability

            if check_transformers_availability():
                logger.info("✓ Transformers available")
            else:
                logger.warning("✗ Transformers not available")

        except Exception as e:
            logger.error(f"Transformers test failed: {e}")

        return True

    def run(self):
        """전체 설정 실행"""
        logger.info("=== Artisan AI LLM Setup ===\n")

        steps = [
            ("Checking Python packages", self.check_python_packages),
            ("Setting up Ollama", self.setup_ollama),
            ("Downloading Korean model", self.download_korean_model),
            ("Verifying setup", self.verify_setup)
        ]

        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            success = step_func()

            if not success:
                logger.warning(f"{step_name} had issues, but continuing...")

        logger.info("\n=== Setup Complete ===")
        logger.info("\nTo start using the LLM system:")
        logger.info("1. Make sure Ollama is running: ollama serve")
        logger.info("2. Start the API server: uvicorn fragrance_ai.api.main:app --reload --port 8001")
        logger.info("3. Access the chat interface at: http://localhost:3000/artisan")


if __name__ == "__main__":
    setup = LLMSetup()
    setup.run()