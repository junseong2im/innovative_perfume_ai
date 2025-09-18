"""
RAG (Retrieval-Augmented Generation) 시스템 구현
최신 RAG 기법들을 적용한 향수 도메인 특화 시스템
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer
)
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
import time
from collections import defaultdict

from ..core.config import settings
from .embedding import AdvancedKoreanFragranceEmbedding, EmbeddingResult

logger = logging.getLogger(__name__)


class RAGMode(Enum):
    """RAG 동작 모드"""
    DENSE_RETRIEVAL = "dense_retrieval"
    SPARSE_RETRIEVAL = "sparse_retrieval" 
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    SELF_RAG = "self_rag"  # Self-reflective RAG
    ADAPTIVE_RAG = "adaptive_rag"  # Adaptive retrieval


@dataclass
class RetrievalContext:
    """검색 컨텍스트"""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    similarity_scores: List[float]
    retrieval_method: str
    retrieval_time: float
    relevance_threshold: float = 0.5


@dataclass
class GenerationResult:
    """생성 결과"""
    generated_text: str
    source_documents: List[str]
    confidence_score: float
    generation_time: float
    retrieval_context: Optional[RetrievalContext] = None
    reasoning_steps: Optional[List[str]] = None


class FragranceRAGSystem:
    """향수 도메인 특화 RAG 시스템"""
    
    def __init__(
        self,
        embedding_model: Optional[AdvancedKoreanFragranceEmbedding] = None,
        generator_model: str = "google/flan-t5-base",
        rag_mode: RAGMode = RAGMode.HYBRID_RETRIEVAL,
        max_retrieved_docs: int = 10,
        max_generation_length: int = 512
    ):
        # Enhanced GPU setup
        self._setup_device_optimization()
        self.rag_mode = rag_mode
        self.max_retrieved_docs = max_retrieved_docs
        self.max_generation_length = max_generation_length

        # Advanced RAG features
        self.use_reranking = True
        self.use_query_expansion = True
        self.use_contextual_filtering = True
        self.similarity_threshold = 0.7
        
        # Initialize embedding model
        self.embedding_model = embedding_model or AdvancedKoreanFragranceEmbedding()
        
        # Initialize generator
        self._load_generator(generator_model)
        
        # Initialize knowledge base
        self.knowledge_base = self._init_knowledge_base()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"RAG System initialized with mode: {rag_mode}")

    def _setup_device_optimization(self) -> None:
        """Enhanced GPU setup for RAG system"""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

            # Enable optimizations
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            # Mixed precision support
            self.use_mixed_precision = torch.cuda.get_device_capability()[0] >= 7
            if self.use_mixed_precision:
                self.scaler = torch.cuda.amp.GradScaler()

            logger.info(f"RAG GPU optimization enabled on {self.device}, mixed precision: {self.use_mixed_precision}")
        else:
            self.device = torch.device("cpu")
            self.use_mixed_precision = False
            logger.warning("RAG system running on CPU - consider using GPU for better performance")

    async def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = None,
        rerank_top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """고급 재순위화를 통한 검색"""
        try:
            top_k = top_k or self.max_retrieved_docs

            # 1단계: 초기 검색 (더 많은 문서 검색)
            initial_docs = await self.retrieve(query, top_k=top_k * 2)

            if not self.use_reranking or len(initial_docs) <= rerank_top_k:
                return initial_docs[:top_k]

            # 2단계: 크로스 인코더를 이용한 재순위화
            reranked_docs = await self._rerank_documents(query, initial_docs, rerank_top_k)

            return reranked_docs[:top_k]

        except Exception as e:
            logger.error(f"Failed to retrieve with reranking: {e}")
            # Fallback to basic retrieval
            return await self.retrieve(query, top_k)

    async def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """문서 재순위화"""
        try:
            # 향수 도메인 특화 재순위화 로직
            scores = []

            for doc in documents:
                content = doc.get('content', '')

                # 1. 의미적 유사도 (기존)
                semantic_score = doc.get('similarity_score', 0.0)

                # 2. 향수 관련 키워드 매칭
                fragrance_keywords = ['향료', '노트', '조향', '블렌딩', '향수', '퍼퓸', '오드투왈렛', 'EDT', 'EDP']
                keyword_score = sum(1 for keyword in fragrance_keywords if keyword in content) / len(fragrance_keywords)

                # 3. 문서 길이 정규화 (너무 짧거나 긴 문서 페널티)
                length_score = min(1.0, len(content) / 1000) if len(content) > 50 else 0.5

                # 4. 질의와 문서의 향수 카테고리 매칭
                category_score = self._calculate_category_match(query, doc)

                # 종합 점수 계산
                final_score = (
                    semantic_score * 0.4 +
                    keyword_score * 0.3 +
                    length_score * 0.2 +
                    category_score * 0.1
                )

                scores.append((doc, final_score))

            # 점수순으로 정렬
            scores.sort(key=lambda x: x[1], reverse=True)

            # 재순위화된 문서 반환
            reranked_docs = []
            for doc, score in scores[:top_k]:
                doc['rerank_score'] = score
                reranked_docs.append(doc)

            logger.info(f"Reranked {len(documents)} documents to top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            return documents[:top_k]

    def _calculate_category_match(self, query: str, document: Dict[str, Any]) -> float:
        """향수 카테고리 매칭 점수 계산"""
        try:
            # 향수 카테고리 매핑
            categories = {
                '플로럴': ['장미', '자스민', '라벤더', '피오니', '벚꽃'],
                '우디': ['샌달우드', '시더', '베티버', '패출리'],
                '오리엔탈': ['바닐라', '앰버', '머스크', '인센스'],
                '프레시': ['시트러스', '레몬', '베르가못', '민트', '유칼립투스'],
                '프루티': ['복숭아', '사과', '베리', '체리', '망고']
            }

            query_lower = query.lower()
            doc_content = document.get('content', '').lower()
            doc_metadata = document.get('metadata', {})

            # 질의에서 카테고리 추출
            query_categories = []
            for category, keywords in categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_categories.append(category)

            # 문서에서 카테고리 추출
            doc_categories = []
            for category, keywords in categories.items():
                if any(keyword in doc_content for keyword in keywords):
                    doc_categories.append(category)

            # 메타데이터에서도 카테고리 확인
            doc_category = doc_metadata.get('category', '')
            if doc_category and doc_category not in doc_categories:
                doc_categories.append(doc_category)

            # 매칭 점수 계산
            if not query_categories:
                return 0.5  # 중립 점수

            matches = len(set(query_categories) & set(doc_categories))
            return matches / len(query_categories) if query_categories else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate category match: {e}")
            return 0.0

    def _load_generator(self, model_name: str):
        """생성 모델 로드"""
        try:
            if "t5" in model_name.lower():
                self.generator = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
                self.generator_tokenizer = T5Tokenizer.from_pretrained(model_name)
            elif "bart" in model_name.lower():
                self.generator = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
                self.generator_tokenizer = BartTokenizer.from_pretrained(model_name)
            else:
                self.generator = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
                self.generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if self.generator_tokenizer.pad_token is None:
                    self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
            logger.info(f"Loaded generator model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load generator: {e}")
            raise
    
    def _init_knowledge_base(self) -> Dict[str, List[Dict[str, Any]]]:
        """향수 지식 베이스 초기화"""
        return {
            "fragrance_notes": [
                {
                    "id": "citrus_001",
                    "content": "시트러스 계열의 향수는 상쾌하고 깨끗한 느낌을 주며, 주로 탑 노트로 사용됩니다. 레몬, 베르가못, 자몽 등이 대표적입니다.",
                    "metadata": {"category": "notes", "type": "top", "family": "citrus"},
                    "embedding": None  # Will be populated
                },
                {
                    "id": "floral_001", 
                    "content": "플로럴 계열은 꽃의 향기를 표현하며, 로맨틱하고 우아한 분위기를 연출합니다. 장미, 자스민, 피오니가 인기가 높습니다.",
                    "metadata": {"category": "notes", "type": "heart", "family": "floral"},
                    "embedding": None
                },
                {
                    "id": "woody_001",
                    "content": "우디 계열은 나무의 따뜻하고 깊은 향을 표현합니다. 백단향, 삼나무, 우드가 베이스 노트로 많이 사용됩니다.",
                    "metadata": {"category": "notes", "type": "base", "family": "woody"},
                    "embedding": None
                }
            ],
            
            "fragrance_tips": [
                {
                    "id": "tip_001",
                    "content": "향수를 더 오래 지속시키려면 피부 보습이 중요합니다. 로션을 먼저 발라주세요.",
                    "metadata": {"category": "tips", "topic": "longevity"},
                    "embedding": None
                },
                {
                    "id": "tip_002", 
                    "content": "계절에 따라 향수를 선택하는 것이 좋습니다. 여름엔 가벼운 시트러스, 겨울엔 따뜻한 우디 계열이 어울립니다.",
                    "metadata": {"category": "tips", "topic": "seasonal"},
                    "embedding": None
                }
            ],
            
            "brand_info": [
                {
                    "id": "brand_001",
                    "content": "샤넬은 1921년 코코 샤넬이 창립한 프랑스 럭셔리 브랜드로, 샤넬 No.5가 대표작입니다.",
                    "metadata": {"category": "brand", "name": "chanel", "country": "france"},
                    "embedding": None
                }
            ]
        }
    
    async def generate_with_rag(
        self,
        query: str,
        context: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_reasoning: bool = True
    ) -> GenerationResult:
        """RAG를 사용한 텍스트 생성"""
        start_time = time.time()
        
        # 1. Retrieval phase
        retrieval_context = await self._retrieve_relevant_documents(
            query, 
            max_docs=self.max_retrieved_docs
        )
        
        # 2. Context preparation
        prepared_context = self._prepare_generation_context(
            query, 
            retrieval_context, 
            additional_context=context
        )
        
        # 3. Generation phase
        if self.rag_mode == RAGMode.SELF_RAG:
            result = await self._generate_with_self_rag(
                prepared_context, 
                temperature, 
                top_p,
                enable_reasoning
            )
        elif self.rag_mode == RAGMode.ADAPTIVE_RAG:
            result = await self._generate_with_adaptive_rag(
                query,
                prepared_context,
                temperature,
                top_p
            )
        else:
            result = await self._generate_standard(
                prepared_context,
                temperature,
                top_p
            )
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            generated_text=result,
            source_documents=[doc["content"] for doc in retrieval_context.retrieved_documents],
            confidence_score=self._calculate_confidence_score(result, retrieval_context),
            generation_time=generation_time,
            retrieval_context=retrieval_context,
            reasoning_steps=None  # Will be populated by self-RAG
        )
    
    async def _retrieve_relevant_documents(
        self,
        query: str,
        max_docs: int = 10,
        relevance_threshold: float = 0.5
    ) -> RetrievalContext:
        """관련 문서 검색"""
        start_time = time.time()
        
        if self.rag_mode == RAGMode.SPARSE_RETRIEVAL:
            docs, scores = await self._sparse_retrieval(query, max_docs)
        elif self.rag_mode == RAGMode.HYBRID_RETRIEVAL:
            docs, scores = await self._hybrid_retrieval(query, max_docs)
        else:  # Dense retrieval (default)
            docs, scores = await self._dense_retrieval(query, max_docs)
        
        # Filter by relevance threshold
        filtered_docs = []
        filtered_scores = []
        
        for doc, score in zip(docs, scores):
            if score >= relevance_threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        
        retrieval_time = time.time() - start_time
        
        return RetrievalContext(
            query=query,
            retrieved_documents=filtered_docs,
            similarity_scores=filtered_scores,
            retrieval_method=self.rag_mode.value,
            retrieval_time=retrieval_time,
            relevance_threshold=relevance_threshold
        )
    
    async def _dense_retrieval(
        self, 
        query: str, 
        max_docs: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Dense 검색 (임베딩 기반)"""
        query_embedding_result = await self.embedding_model.encode_async([query])
        query_embedding = query_embedding_result.embeddings[0]
        
        all_docs = []
        for doc_list in self.knowledge_base.values():
            all_docs.extend(doc_list)
        
        # Generate embeddings for documents if not already done
        for doc in all_docs:
            if doc["embedding"] is None:
                doc_embedding_result = await self.embedding_model.encode_async([doc["content"]])
                doc["embedding"] = doc_embedding_result.embeddings[0]
        
        # Calculate similarities
        similarities = []
        for doc in all_docs:
            similarity = self.embedding_model.compute_semantic_similarity(
                query, doc["content"]
            )
            similarities.append((doc, similarity))
        
        # Sort and select top documents
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_docs = similarities[:max_docs]
        
        docs = [item[0] for item in top_docs]
        scores = [item[1] for item in top_docs]
        
        return docs, scores
    
    async def _sparse_retrieval(
        self,
        query: str,
        max_docs: int
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Sparse 검색 (키워드 기반)"""
        query_terms = set(query.lower().split())
        
        all_docs = []
        for doc_list in self.knowledge_base.values():
            all_docs.extend(doc_list)
        
        # BM25-like scoring
        doc_scores = []
        for doc in all_docs:
            doc_terms = set(doc["content"].lower().split())
            overlap = len(query_terms.intersection(doc_terms))
            score = overlap / len(query_terms) if len(query_terms) > 0 else 0
            doc_scores.append((doc, score))
        
        # Sort and select top documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = doc_scores[:max_docs]
        
        docs = [item[0] for item in top_docs]
        scores = [item[1] for item in top_docs]
        
        return docs, scores
    
    async def _hybrid_retrieval(
        self,
        query: str,
        max_docs: int,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """하이브리드 검색 (Dense + Sparse)"""
        # Get both dense and sparse results
        dense_docs, dense_scores = await self._dense_retrieval(query, max_docs * 2)
        sparse_docs, sparse_scores = await self._sparse_retrieval(query, max_docs * 2)
        
        # Combine scores
        doc_score_map = {}
        
        # Add dense scores
        for doc, score in zip(dense_docs, dense_scores):
            doc_id = doc["id"]
            doc_score_map[doc_id] = {
                "doc": doc,
                "dense_score": score * dense_weight,
                "sparse_score": 0
            }
        
        # Add sparse scores
        for doc, score in zip(sparse_docs, sparse_scores):
            doc_id = doc["id"]
            if doc_id in doc_score_map:
                doc_score_map[doc_id]["sparse_score"] = score * sparse_weight
            else:
                doc_score_map[doc_id] = {
                    "doc": doc,
                    "dense_score": 0,
                    "sparse_score": score * sparse_weight
                }
        
        # Calculate final scores
        final_scores = []
        for doc_info in doc_score_map.values():
            final_score = doc_info["dense_score"] + doc_info["sparse_score"]
            final_scores.append((doc_info["doc"], final_score))
        
        # Sort and select top documents
        final_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = final_scores[:max_docs]
        
        docs = [item[0] for item in top_docs]
        scores = [item[1] for item in top_docs]
        
        return docs, scores
    
    def _prepare_generation_context(
        self,
        query: str,
        retrieval_context: RetrievalContext,
        additional_context: Optional[str] = None
    ) -> str:
        """생성을 위한 컨텍스트 준비"""
        context_parts = []
        
        # Add retrieved documents
        context_parts.append("관련 정보:")
        for i, doc in enumerate(retrieval_context.retrieved_documents[:5]):  # Top 5 only
            score = retrieval_context.similarity_scores[i]
            context_parts.append(f"{i+1}. {doc['content']} (관련도: {score:.2f})")
        
        # Add additional context if provided
        if additional_context:
            context_parts.append(f"추가 컨텍스트: {additional_context}")
        
        # Add the question
        context_parts.append(f"질문: {query}")
        context_parts.append("답변:")
        
        return "\n".join(context_parts)
    
    async def _generate_standard(
        self,
        context: str,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """표준 생성"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_text_sync,
            context,
            temperature,
            top_p
        )
    
    def _generate_text_sync(
        self,
        context: str,
        temperature: float,
        top_p: float
    ) -> str:
        """동기 텍스트 생성"""
        inputs = self.generator_tokenizer(
            context,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=self.max_generation_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.generator_tokenizer.pad_token_id,
                eos_token_id=self.generator_tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = self.generator_tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    async def _generate_with_self_rag(
        self,
        context: str,
        temperature: float,
        top_p: float,
        enable_reasoning: bool
    ) -> str:
        """Self-RAG 생성 (자기 반성적)"""
        # 1. Initial generation
        initial_response = await self._generate_standard(context, temperature, top_p)
        
        if not enable_reasoning:
            return initial_response
        
        # 2. Self-evaluation
        evaluation_prompt = f"""
        다음 답변을 평가해주세요:
        질문: {context.split('질문:')[-1].split('답변:')[0].strip()}
        답변: {initial_response}
        
        이 답변이 정확하고 완전한지 평가하고, 개선점이 있다면 더 나은 답변을 제공해주세요.
        """
        
        refined_response = await self._generate_standard(evaluation_prompt, temperature * 0.8, top_p)
        
        # 3. Choose better response
        if len(refined_response) > len(initial_response) * 0.8:
            return refined_response
        else:
            return initial_response
    
    async def _generate_with_adaptive_rag(
        self,
        query: str,
        context: str,
        temperature: float,
        top_p: float
    ) -> str:
        """적응적 RAG (쿼리 복잡도에 따른 적응)"""
        # Analyze query complexity
        complexity_score = self._analyze_query_complexity(query)
        
        if complexity_score > 0.7:
            # Complex query - use more documents and self-refinement
            return await self._generate_with_self_rag(context, temperature, top_p, True)
        elif complexity_score > 0.4:
            # Medium complexity - standard generation with lower temperature
            return await self._generate_standard(context, temperature * 0.8, top_p)
        else:
            # Simple query - faster generation
            return await self._generate_standard(context, temperature * 1.2, top_p)
    
    def _analyze_query_complexity(self, query: str) -> float:
        """쿼리 복잡도 분석"""
        complexity_indicators = [
            "어떻게", "왜", "설명", "비교", "차이", "방법", "과정", "이유",
            "how", "why", "explain", "compare", "difference", "method", "process"
        ]
        
        query_lower = query.lower()
        complexity_score = 0.0
        
        # Length factor
        complexity_score += min(len(query) / 100, 0.3)
        
        # Complexity keywords
        keyword_count = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        complexity_score += min(keyword_count * 0.2, 0.4)
        
        # Question marks and sentence complexity
        question_marks = query.count('?')
        complexity_score += min(question_marks * 0.1, 0.3)
        
        return min(complexity_score, 1.0)
    
    def _calculate_confidence_score(
        self,
        generated_text: str,
        retrieval_context: RetrievalContext
    ) -> float:
        """신뢰도 점수 계산"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Retrieval quality
        if retrieval_context.similarity_scores:
            avg_similarity = np.mean(retrieval_context.similarity_scores)
            confidence += avg_similarity * 0.3
        
        # Factor 2: Generated text length (reasonable length indicates good generation)
        text_length_factor = min(len(generated_text) / 200, 0.2)
        confidence += text_length_factor
        
        # Factor 3: Number of retrieved documents
        doc_count_factor = min(len(retrieval_context.retrieved_documents) / 10, 0.1)
        confidence += doc_count_factor
        
        return min(confidence, 1.0)
    
    def add_to_knowledge_base(
        self,
        category: str,
        documents: List[Dict[str, Any]]
    ):
        """지식 베이스에 문서 추가"""
        if category not in self.knowledge_base:
            self.knowledge_base[category] = []
        
        self.knowledge_base[category].extend(documents)
        logger.info(f"Added {len(documents)} documents to {category}")
    
    def get_knowledge_base_stats(self) -> Dict[str, int]:
        """지식 베이스 통계"""
        stats = {}
        for category, docs in self.knowledge_base.items():
            stats[category] = len(docs)
        return stats
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Backward compatibility alias
KoreanFragranceEmbedding = AdvancedKoreanFragranceEmbedding