"""
Ollama LLM 클라이언트
- 로컬 Ollama 서버와 통신
- Llama3, Mistral 등 오픈소스 모델 지원
"""

import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Ollama 설정"""
    base_url: str = "http://localhost:11434"
    model: str = "llama3:8b-instruct-q4_K_M"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: int = 60


class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """클라이언트 초기화"""
        self.config = config or OllamaConfig()
        self.session = None
        self._is_available = None

    async def __aenter__(self):
        """컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()

    async def check_availability(self) -> bool:
        """Ollama 서버 가용성 확인"""
        if self._is_available is not None:
            return self._is_available

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m['name'] for m in data.get('models', [])]

                        if self.config.model in models:
                            logger.info(f"Ollama server available with model {self.config.model}")
                            self._is_available = True
                            return True
                        else:
                            logger.warning(f"Model {self.config.model} not found. Available: {models}")
                            self._is_available = False
                            return False
        except Exception as e:
            logger.error(f"Ollama server not available: {e}")
            self._is_available = False
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """텍스트 생성"""
        if not await self.check_availability():
            raise RuntimeError("Ollama server not available")

        # 전체 프롬프트 구성
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.config.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens or self.config.max_tokens
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('response', '')
                    else:
                        error = await response.text()
                        raise Exception(f"Ollama API error: {error}")

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """채팅 형식 대화"""
        if not await self.check_availability():
            raise RuntimeError("Ollama server not available")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens or self.config.max_tokens
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('message', {}).get('content', '')
                    else:
                        error = await response.text()
                        raise Exception(f"Ollama API error: {error}")

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def embeddings(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        if not await self.check_availability():
            raise RuntimeError("Ollama server not available")

        payload = {
            "model": self.config.model,
            "prompt": text
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.base_url}/api/embeddings",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('embedding', [])
                    else:
                        error = await response.text()
                        raise Exception(f"Ollama API error: {error}")

        except Exception as e:
            logger.error(f"Ollama embeddings failed: {e}")
            raise


class OllamaOrchestrator:
    """Ollama 기반 오케스트레이터"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """오케스트레이터 초기화"""
        self.client = OllamaClient(config)
        self.system_prompt = """You are Artisan, an expert AI perfumer with deep knowledge of fragrance composition, history, and artistry.
        You help users create personalized perfume recipes by understanding their preferences, emotions, and desires.

        Your responses should be:
        1. Knowledgeable yet accessible
        2. Creative and inspiring
        3. Focused on perfumery and fragrance
        4. Personalized to the user's request

        When creating perfume recipes, consider:
        - Fragrance families and their characteristics
        - Note composition (top, heart, base)
        - Seasonal appropriateness
        - Emotional resonance
        - Cultural significance
        """

    async def analyze_intent(self, message: str) -> Dict[str, Any]:
        """사용자 의도 분석"""
        prompt = f"""Analyze the following message and identify the user's intent regarding perfume:

Message: "{message}"

Respond in JSON format with:
1. intent_type: (create_perfume/search_perfume/knowledge_query/validate_recipe/general_chat)
2. confidence: (0.0 to 1.0)
3. entities: extracted entities like fragrance_family, mood, season, etc.
4. summary: brief summary of what the user wants

Response:"""

        try:
            response = await self.client.generate(prompt, system_prompt=self.system_prompt)

            # JSON 파싱 시도
            try:
                # JSON 부분 추출
                if '{' in response and '}' in response:
                    json_start = response.index('{')
                    json_end = response.rindex('}') + 1
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
            except:
                pass

            # 파싱 실패 시 기본값
            return {
                "intent_type": "general_chat",
                "confidence": 0.5,
                "entities": {},
                "summary": message[:100]
            }

        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # 폴백 규칙 기반 분석
            return self._fallback_intent_analysis(message)

    def _fallback_intent_analysis(self, message: str) -> Dict[str, Any]:
        """폴백 의도 분석"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["만들", "create", "design", "제작"]):
            return {
                "intent_type": "create_perfume",
                "confidence": 0.8,
                "entities": {},
                "summary": "Create a new perfume"
            }
        elif any(word in message_lower for word in ["찾", "search", "find", "추천"]):
            return {
                "intent_type": "search_perfume",
                "confidence": 0.8,
                "entities": {},
                "summary": "Search for perfumes"
            }
        else:
            return {
                "intent_type": "general_chat",
                "confidence": 0.5,
                "entities": {},
                "summary": message[:100]
            }

    async def generate_response(
        self,
        message: str,
        intent: Dict[str, Any],
        context: List[Dict[str, str]] = None
    ) -> str:
        """응답 생성"""
        # 컨텍스트 포함한 메시지 구성
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": message})

        try:
            response = await self.client.chat(messages)
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해주세요."


# 전역 Ollama 클라이언트
_ollama_client = None


def get_ollama_client() -> OllamaClient:
    """Ollama 클라이언트 싱글톤"""
    global _ollama_client
    if _ollama_client is None:
        # 설정 로드
        try:
            with open("configs/local.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
                ollama_config = config.get('llm_orchestrator', {})

                _ollama_client = OllamaClient(OllamaConfig(
                    base_url=ollama_config.get('api_base', 'http://localhost:11434'),
                    model=ollama_config.get('model_name_or_path', 'llama3:8b-instruct-q4_K_M')
                ))
        except:
            _ollama_client = OllamaClient()

    return _ollama_client