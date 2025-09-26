"""
vLLM 클라이언트 - 고성능 LLM 추론 엔진 인터페이스
PagedAttention을 사용하여 최대 24배 높은 처리량 제공
"""

import asyncio
import aiohttp
import json
from typing import AsyncIterator, Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """vLLM 서버 설정"""
    base_url: str = "http://localhost:8100"
    model_name: str = "llama3-8b"
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3


class VLLMClient:
    """vLLM 서빙 엔진 클라이언트"""

    def __init__(self, config: Optional[VLLMConfig] = None):
        """
        vLLM 클라이언트 초기화

        Args:
            config: vLLM 설정 객체
        """
        self.config = config or VLLMConfig()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """세션이 없으면 생성"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> AsyncIterator[str] | str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: nucleus sampling 파라미터
            stream: 스트리밍 여부
            **kwargs: 추가 생성 파라미터

        Returns:
            생성된 텍스트 (스트리밍 또는 일반)
        """
        await self._ensure_session()

        # vLLM OpenAI 호환 API 엔드포인트
        url = urljoin(self.config.base_url, "/v1/completions")

        # 요청 페이로드
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "stop": kwargs.get("stop", None),
            "n": kwargs.get("n", 1),
            "best_of": kwargs.get("best_of", 1),
            "logprobs": kwargs.get("logprobs", None),
            "echo": kwargs.get("echo", False),
        }

        # None 값 제거
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if stream:
            return self._stream_generate(url, headers, payload)
        else:
            return await self._generate(url, headers, payload)

    async def _generate(self, url: str, headers: Dict, payload: Dict) -> str:
        """일반 생성 (비스트리밍)"""
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["text"]

            except aiohttp.ClientError as e:
                logger.warning(f"vLLM request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 지수 백오프

    async def _stream_generate(
        self,
        url: str,
        headers: Dict,
        payload: Dict
    ) -> AsyncIterator[str]:
        """스트리밍 생성"""
        headers["Accept"] = "text/event-stream"

        async with self.session.post(
            url,
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                if not line:
                    continue

                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    break

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        text = data["choices"][0]["text"]
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> AsyncIterator[str] | str:
        """
        채팅 완성 (ChatGPT 스타일)

        Args:
            messages: 대화 메시지 리스트
            max_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            stream: 스트리밍 여부

        Returns:
            생성된 응답
        """
        await self._ensure_session()

        url = urljoin(self.config.base_url, "/v1/chat/completions")

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if stream:
            return self._stream_chat(url, headers, payload)
        else:
            return await self._chat(url, headers, payload)

    async def _chat(self, url: str, headers: Dict, payload: Dict) -> str:
        """일반 채팅 (비스트리밍)"""
        async with self.session.post(
            url,
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]

    async def _stream_chat(
        self,
        url: str,
        headers: Dict,
        payload: Dict
    ) -> AsyncIterator[str]:
        """스트리밍 채팅"""
        headers["Accept"] = "text/event-stream"

        async with self.session.post(
            url,
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()

            async for line in response.content:
                if not line:
                    continue

                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    break

                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def embeddings(
        self,
        input_texts: List[str],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        텍스트 임베딩 생성

        Args:
            input_texts: 임베딩할 텍스트 리스트
            model: 임베딩 모델 (선택적)

        Returns:
            임베딩 벡터 리스트
        """
        await self._ensure_session()

        url = urljoin(self.config.base_url, "/v1/embeddings")

        payload = {
            "model": model or self.config.model_name,
            "input": input_texts
        }

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with self.session.post(
            url,
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return [item["embedding"] for item in data["data"]]

    async def health_check(self) -> bool:
        """
        서버 상태 확인

        Returns:
            서버가 정상이면 True
        """
        await self._ensure_session()

        try:
            url = urljoin(self.config.base_url, "/health")
            async with self.session.get(url) as response:
                return response.status == 200
        except:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 조회

        Returns:
            모델 메타데이터
        """
        await self._ensure_session()

        url = urljoin(self.config.base_url, "/v1/models")

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        async with self.session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


class VLLMBatchProcessor:
    """배치 처리 최적화"""

    def __init__(self, client: VLLMClient, batch_size: int = 32):
        """
        배치 프로세서 초기화

        Args:
            client: vLLM 클라이언트
            batch_size: 배치 크기
        """
        self.client = client
        self.batch_size = batch_size

    async def process_batch(
        self,
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """
        배치 처리

        Args:
            prompts: 프롬프트 리스트
            **generation_kwargs: 생성 파라미터

        Returns:
            생성된 텍스트 리스트
        """
        results = []

        # 배치로 나누기
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]

            # 병렬 처리
            tasks = [
                self.client.generate(prompt, **generation_kwargs)
                for prompt in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results


# 싱글톤 인스턴스
_vllm_client: Optional[VLLMClient] = None


def get_vllm_client(config: Optional[VLLMConfig] = None) -> VLLMClient:
    """
    vLLM 클라이언트 싱글톤 가져오기

    Args:
        config: vLLM 설정 (선택적)

    Returns:
        vLLM 클라이언트 인스턴스
    """
    global _vllm_client

    if _vllm_client is None:
        _vllm_client = VLLMClient(config)

    return _vllm_client


# 사용 예제
async def example_usage():
    """vLLM 사용 예제"""

    # 클라이언트 생성
    config = VLLMConfig(
        base_url="http://localhost:8100",
        model_name="llama3-8b"
    )

    async with VLLMClient(config) as client:
        # 1. 일반 생성
        response = await client.generate(
            "Create a summer citrus perfume recipe:",
            max_tokens=200,
            temperature=0.7
        )
        print(f"Generated: {response}")

        # 2. 스트리밍 생성
        print("\nStreaming generation:")
        async for chunk in client.generate(
            "Describe the scent of roses:",
            max_tokens=100,
            stream=True
        ):
            print(chunk, end="", flush=True)
        print()

        # 3. 채팅 완성
        messages = [
            {"role": "system", "content": "You are a perfume expert."},
            {"role": "user", "content": "What makes a good perfume?"}
        ]

        chat_response = await client.chat_completion(
            messages,
            max_tokens=150
        )
        print(f"\nChat response: {chat_response}")

        # 4. 배치 처리
        processor = VLLMBatchProcessor(client, batch_size=4)
        prompts = [
            "Create a floral perfume:",
            "Create a woody perfume:",
            "Create a fresh perfume:",
            "Create an oriental perfume:"
        ]

        batch_results = await processor.process_batch(
            prompts,
            max_tokens=50
        )

        for prompt, result in zip(prompts, batch_results):
            print(f"\n{prompt}\n{result}")


if __name__ == "__main__":
    asyncio.run(example_usage())