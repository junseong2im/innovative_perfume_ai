"""
AI 서버 분리 모듈

로컬에서 실행되는 AI 모델들을 독립적인 FastAPI 서버로 분리하여
원격에서 HTTP API로 접근할 수 있도록 합니다.

주요 기능:
- 임베딩 모델 API 엔드포인트
- 텍스트 생성 모델 API 엔드포인트
- ChromaDB 벡터 저장소 API 엔드포인트
- 하이브리드 배포 지원
"""

import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fragrance_ai.core.config import settings


class EmbeddingRequest(BaseModel):
    """임베딩 요청 모델"""
    texts: List[str]
    normalize: bool = True


class GenerationRequest(BaseModel):
    """텍스트 생성 요청 모델"""
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class SearchRequest(BaseModel):
    """벡터 검색 요청 모델"""
    query: str
    top_k: int = 10
    search_type: str = "similarity"


class AIServerClient:
    """
    원격 AI 서버와 통신하는 클라이언트 클래스

    로컬 AI 서버 또는 원격 AI 서버와 HTTP로 통신하여
    AI 기능을 사용할 수 있습니다.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.ai_server_url or "http://localhost:8001"
        self.session = None

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트의 임베딩 벡터를 생성합니다.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        url = f"{self.base_url}/embeddings"
        payload = EmbeddingRequest(texts=texts).dict()

        if self.session:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["embeddings"]
        else:
            # 동기 방식 폴백
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["embeddings"]

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        주어진 프롬프트로 텍스트를 생성합니다.

        Args:
            prompt: 생성할 텍스트의 프롬프트
            **kwargs: 생성 파라미터 (temperature, top_p 등)

        Returns:
            생성된 텍스트
        """
        url = f"{self.base_url}/generate"
        payload = GenerationRequest(prompt=prompt, **kwargs).dict()

        if self.session:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["generated_text"]
        else:
            # 동기 방식 폴백
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["generated_text"]

    async def search_similar(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        유사한 향수 레시피를 검색합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수

        Returns:
            검색 결과 리스트
        """
        url = f"{self.base_url}/search"
        payload = SearchRequest(query=query, top_k=top_k).dict()

        if self.session:
            async with self.session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["results"]
        else:
            # 동기 방식 폴백
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["results"]

    async def health_check(self) -> Dict[str, Any]:
        """
        AI 서버의 상태를 확인합니다.

        Returns:
            서버 상태 정보
        """
        url = f"{self.base_url}/health"

        if self.session:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        else:
            # 동기 방식 폴백
            response = requests.get(url)
            response.raise_for_status()
            return response.json()


def get_ai_client() -> AIServerClient:
    """
    AI 서버 클라이언트 인스턴스를 반환합니다.

    설정에 따라 로컬 또는 원격 AI 서버에 연결합니다.

    Returns:
        AI 서버 클라이언트 인스턴스
    """
    return AIServerClient()


# 동기 버전의 편의 함수들
def sync_get_embeddings(texts: List[str]) -> List[List[float]]:
    """동기 버전의 임베딩 생성"""
    client = get_ai_client()
    return asyncio.run(client.get_embeddings(texts))


def sync_generate_text(prompt: str, **kwargs) -> str:
    """동기 버전의 텍스트 생성"""
    client = get_ai_client()
    return asyncio.run(client.generate_text(prompt, **kwargs))


def sync_search_similar(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """동기 버전의 유사성 검색"""
    client = get_ai_client()
    return asyncio.run(client.search_similar(query, top_k))