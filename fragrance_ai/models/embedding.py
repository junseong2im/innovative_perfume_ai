from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, RobertaModel, DebertaV2Model
)
from sentence_transformers import SentenceTransformer, util
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    embeddings: np.ndarray
    tokens: Optional[List[str]] = None
    attention_weights: Optional[np.ndarray] = None
    pooling_strategy: str = "mean"
    processing_time: float = 0.0


class EmbeddingModel(Protocol):
    """임베딩 모델 프로토콜"""
    def encode(self, texts: List[str], **kwargs) -> np.ndarray: ...
    async def encode_async(self, texts: List[str], **kwargs) -> EmbeddingResult: ...


class AdvancedKoreanFragranceEmbedding:
    """한국어 특화 향수 임베딩 모델 - 최신 기법 적용"""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        use_adapter: bool = True,
        enable_multi_aspect: bool = True,
        cache_size: int = 10000
    ):
        self.model_name = model_name or settings.embedding_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_adapter = use_adapter
        self.enable_multi_aspect = enable_multi_aspect
        
        # Initialize caching
        self.embedding_cache = {}
        self.cache_size = cache_size
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load models and initialize components
        self._load_model()
        self._init_fragrance_adapter()
        self._init_multi_aspect_embeddings()
        self._init_fragrance_vocabulary()
        
        logger.info(f"Advanced Korean Fragrance Embedding initialized")
    
    def _load_model(self) -> None:
        """최신 임베딩 모델 로드 (Multi-model support)"""
        try:
            # Primary model - sentence-transformers for production
            self.primary_model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # Secondary model for Korean-specific tasks
            korean_models = [
                "jhgan/ko-sroberta-multitask",
                "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
                "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
            ]
            
            for korean_model in korean_models:
                try:
                    self.korean_model = SentenceTransformer(korean_model, device=self.device)
                    logger.info(f"Loaded Korean model: {korean_model}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {korean_model}: {e}")
                    continue
            else:
                self.korean_model = self.primary_model
            
            # Raw transformer for custom pooling
            self.raw_model = AutoModel.from_pretrained(
                "klue/bert-base", 
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            self.raw_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
            
            logger.info(f"Multi-model embedding system loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def _init_fragrance_adapter(self) -> None:
        """향수 도메인 특화 어댑터 초기화"""
        if not self.use_adapter:
            self.adapter = None
            return
            
        # Simple adapter network for domain adaptation
        embedding_dim = self.primary_model.get_sentence_embedding_dimension()
        
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        ).to(self.device)
        
        # Initialize with identity-like weights for stable training
        with torch.no_grad():
            self.adapter[0].weight.fill_(0.1)
            self.adapter[3].weight.fill_(0.1)
    
    def _init_multi_aspect_embeddings(self) -> None:
        """다면적 임베딩 시스템 초기화"""
        if not self.enable_multi_aspect:
            self.aspect_embedders = None
            return
            
        self.aspect_embedders = {
            "scent_profile": self._create_aspect_embedder("scent"),
            "mood_emotion": self._create_aspect_embedder("mood"), 
            "occasion_context": self._create_aspect_embedder("occasion"),
            "intensity_longevity": self._create_aspect_embedder("intensity")
        }
    
    def _create_aspect_embedder(self, aspect_type: str) -> nn.Module:
        """특정 측면을 위한 임베딩 생성기"""
        base_dim = self.primary_model.get_sentence_embedding_dimension()
        
        return nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, base_dim // 2),
            nn.Tanh()
        ).to(self.device)
    
    def _init_fragrance_vocabulary(self) -> Dict[str, List[str]]:
        """향수 특화 어휘 사전 초기화 (확장)"""
        self.fragrance_vocab = {
            "top_notes": [
                # 시트러스
                "시트러스", "레몬", "베르가못", "자몽", "오렌지", "라임", "유자", "한라봉",
                "citrus", "lemon", "bergamot", "grapefruit", "orange", "lime", "yuzu",
                
                # 프루티
                "사과", "배", "복숭아", "살구", "체리", "딸기", "라즈베리", "블랙커런트",
                "apple", "pear", "peach", "apricot", "cherry", "strawberry", "raspberry",
                
                # 허브/아로마틱
                "바질", "로즈마리", "라벤더", "민트", "유칼립투스", "타임", "세이지",
                "basil", "rosemary", "lavender", "mint", "eucalyptus", "thyme", "sage"
            ],
            
            "heart_notes": [
                # 플로럴
                "장미", "자스민", "백합", "피오니", "아이리스", "바이올렛", "프리지아", "일랑일랑",
                "rose", "jasmine", "lily", "peony", "iris", "violet", "freesia", "ylang-ylang",
                
                # 스파이시
                "계피", "정향", "육두구", "후추", "생강", "카다몬", "사프란", "코리앤더",
                "cinnamon", "clove", "nutmeg", "pepper", "ginger", "cardamom", "saffron",
                
                # 그린/허브
                "잔디", "이끼", "차잎", "대나무", "연꽃", "국화", "쑥", "창포",
                "grass", "moss", "tea leaf", "bamboo", "lotus", "chrysanthemum"
            ],
            
            "base_notes": [
                # 우디
                "백단향", "삼나무", "참나무", "소나무", "편백", "히노키", "티크", "로즈우드",
                "sandalwood", "cedarwood", "oakwood", "pine", "cypress", "hinoki", "teak",
                
                # 오리엔탈/발삼
                "바닐라", "앰버", "머스크", "우드", "인센스", "미르", "프랑킨센스", "벤조인",
                "vanilla", "amber", "musk", "oud", "incense", "myrrh", "frankincense", "benzoin",
                
                # 애니멀릭
                "사향", "용연향", "캐스토륨", "시벳", "ambergris", "castoreum", "civet"
            ],
            
            "korean_traditional": [
                "송화", "매화", "난초", "국화", "모란", "창포", "쑥", "솔잎", "대나무",
                "인삼", "당귀", "계피", "정향", "팔각", "회향", "생강", "마", "연근"
            ],
            
            "mood_descriptors": [
                "우아한", "시원한", "따뜻한", "신비로운", "로맨틱한", "프레시한", "세련된",
                "elegant", "cool", "warm", "mysterious", "romantic", "fresh", "sophisticated",
                "차분한", "활기찬", "관능적인", "깔끔한", "부드러운", "강렬한", "은은한"
            ]
        }
        
        # Reverse mapping for efficient lookup
        self.vocab_to_category = {}
        for category, terms in self.fragrance_vocab.items():
            for term in terms:
                self.vocab_to_category[term.lower()] = category
        
        return self.fragrance_vocab
    
    async def encode_async(
        self, 
        texts: Union[str, List[str]], 
        model_type: str = "primary",
        pooling_strategy: str = "mean",
        enable_caching: bool = True,
        return_attention: bool = False
    ) -> EmbeddingResult:
        """비동기 임베딩 생성 (최신 기법)"""
        import time
        start_time = time.time()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        if enable_caching:
            cache_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, model_type, pooling_strategy)
                if cache_key in self.embedding_cache:
                    cache_results.append((i, self.embedding_cache[cache_key]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cache_results = []
        
        # Encode uncached texts
        if uncached_texts:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor, 
                self._encode_batch, 
                uncached_texts, 
                model_type, 
                pooling_strategy,
                return_attention
            )
            
            # Cache results
            if enable_caching:
                for i, text in enumerate(uncached_texts):
                    cache_key = self._get_cache_key(text, model_type, pooling_strategy)
                    self.embedding_cache[cache_key] = embeddings[i]
                    self._manage_cache_size()
        else:
            embeddings = np.array([])
        
        # Combine cached and new results
        if cache_results and len(uncached_texts) > 0:
            final_embeddings = np.zeros((len(texts), embeddings.shape[1]))
            
            # Insert cached results
            for idx, cached_emb in cache_results:
                final_embeddings[idx] = cached_emb
            
            # Insert new results
            for i, idx in enumerate(uncached_indices):
                final_embeddings[idx] = embeddings[i]
            
            embeddings = final_embeddings
        elif cache_results and len(uncached_texts) == 0:
            embeddings = np.array([cached_emb for _, cached_emb in cache_results])
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            pooling_strategy=pooling_strategy,
            processing_time=processing_time
        )
    
    def _encode_batch(
        self, 
        texts: List[str], 
        model_type: str = "primary",
        pooling_strategy: str = "mean",
        return_attention: bool = False
    ) -> np.ndarray:
        """배치 임베딩 생성"""
        if model_type == "korean":
            embeddings = self.korean_model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        elif model_type == "custom":
            embeddings = self._encode_with_custom_pooling(texts, pooling_strategy)
        else:  # primary
            embeddings = self.primary_model.encode(
                texts,
                batch_size=32, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        # Apply domain adapter if enabled
        if self.adapter is not None:
            with torch.no_grad():
                embeddings_tensor = torch.from_numpy(embeddings).to(self.device)
                adapted_embeddings = self.adapter(embeddings_tensor)
                embeddings = adapted_embeddings.cpu().numpy()
        
        return embeddings
    
    def _encode_with_custom_pooling(
        self, 
        texts: List[str], 
        pooling_strategy: str = "mean"
    ) -> np.ndarray:
        """커스텀 풀링 전략을 사용한 임베딩"""
        encoded_inputs = self.raw_tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.raw_model(**encoded_inputs)
            
            if pooling_strategy == "mean":
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = encoded_inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            elif pooling_strategy == "cls":
                # CLS token
                embeddings = outputs.last_hidden_state[:, 0]
            
            elif pooling_strategy == "max":
                # Max pooling
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = encoded_inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[input_mask_expanded == 0] = -1e9
                embeddings = torch.max(token_embeddings, 1)[0]
            
            elif pooling_strategy == "attention_weighted":
                # Attention-weighted pooling
                attention_weights = outputs.attentions[-1].mean(dim=1)  # Average over heads
                token_embeddings = outputs.last_hidden_state
                attention_expanded = attention_weights.mean(dim=1, keepdim=True)  # Average over tokens for weights
                embeddings = torch.sum(token_embeddings * attention_expanded.unsqueeze(-1), dim=1)
            
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embeddings.cpu().numpy()
    
    def encode_multi_aspect(
        self, 
        texts: Union[str, List[str]],
        aspects: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """다면적 임베딩 생성"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.enable_multi_aspect or self.aspect_embedders is None:
            base_embeddings = self.primary_model.encode(texts, convert_to_numpy=True)
            return {"base": base_embeddings}
        
        aspects = aspects or list(self.aspect_embedders.keys())
        base_embeddings = self.primary_model.encode(texts, convert_to_numpy=True)
        
        results = {"base": base_embeddings}
        
        with torch.no_grad():
            base_tensor = torch.from_numpy(base_embeddings).to(self.device)
            
            for aspect in aspects:
                if aspect in self.aspect_embedders:
                    aspect_embeddings = self.aspect_embedders[aspect](base_tensor)
                    results[aspect] = aspect_embeddings.cpu().numpy()
        
        return results
    
    def compute_semantic_similarity(
        self, 
        text1: str, 
        text2: str,
        method: str = "cosine"
    ) -> float:
        """의미적 유사도 계산 (여러 방법 지원)"""
        emb1 = self.primary_model.encode([text1], convert_to_numpy=True)[0]
        emb2 = self.primary_model.encode([text2], convert_to_numpy=True)[0]
        
        if method == "cosine":
            return float(util.cos_sim(emb1, emb2))
        elif method == "dot":
            return float(np.dot(emb1, emb2))
        elif method == "euclidean":
            return float(1 / (1 + np.linalg.norm(emb1 - emb2)))
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def find_similar_fragrances(
        self,
        query: str,
        fragrance_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """유사한 향수 찾기"""
        query_embedding = self.primary_model.encode([query], convert_to_numpy=True)[0]
        
        similarities = []
        for fragrance_name, fragrance_emb in fragrance_embeddings.items():
            similarity = util.cos_sim(query_embedding, fragrance_emb).item()
            if similarity >= min_similarity:
                similarities.append((fragrance_name, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _get_cache_key(self, text: str, model_type: str, pooling_strategy: str) -> str:
        """캐시 키 생성"""
        content = f"{text}|{model_type}|{pooling_strategy}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _manage_cache_size(self):
        """캐시 크기 관리"""
        if len(self.embedding_cache) > self.cache_size:
            # Remove oldest entries (simple LRU-like behavior)
            items_to_remove = len(self.embedding_cache) - self.cache_size
            keys_to_remove = list(self.embedding_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.embedding_cache[key]
    
    def get_vocabulary_embeddings(self, category: Optional[str] = None) -> Dict[str, np.ndarray]:
        """어휘 임베딩 생성"""
        if category and category in self.fragrance_vocab:
            terms = self.fragrance_vocab[category]
        else:
            terms = []
            for term_list in self.fragrance_vocab.values():
                terms.extend(term_list)
        
        embeddings = self.primary_model.encode(terms, convert_to_numpy=True)
        return {term: emb for term, emb in zip(terms, embeddings)}
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        # Fragrance-specific vocabulary
        self.fragrance_vocab = self._init_fragrance_vocabulary()
        
    def _load_model(self) -> None:
        """임베딩 모델 로드"""
        try:
            # Use sentence-transformers for better multilingual support
            if "sentence-transformers" in self.model_name:
                self.model = SentenceTransformer(self.model_name)
                self.model.to(self.device)
                self.tokenizer = None  # sentence-transformers handles tokenization internally
            else:
                # Use transformers directly for more control
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
            
            logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_fragrance_vocabulary(self) -> Dict[str, List[str]]:
        """향수 특화 어휘 사전 초기화"""
        return {
            "top_notes": [
                "시트러스", "레몬", "베르가못", "자몽", "오렌지", "라임", "유자", 
                "만다린", "네롤리", "페티그레인", "상큼한", "신선한", "활기찬"
            ],
            "heart_notes": [
                "장미", "자스민", "라벤더", "일랑일랑", "제라늄", "프리지아",
                "백합", "피오니", "바이올렛", "부드러운", "우아한", "로맨틱한"
            ],
            "base_notes": [
                "바닐라", "머스크", "앰버", "샌달우드", "시더우드", "패츌리",
                "베티버", "통카빈", "오크모스", "깊은", "따뜻한", "관능적인"
            ],
            "mood_descriptors": [
                "신비로운", "우아한", "세련된", "자연스러운", "도시적인",
                "빈티지한", "모던한", "클래식한", "대담한", "은은한",
                "강렬한", "부드러운", "따뜻한", "차가운", "달콤한"
            ],
            "seasonal": [
                "봄", "여름", "가을", "겨울", "사계절", "데일리",
                "이브닝", "주간", "야간", "특별한날"
            ],
            "personality": [
                "활발한", "내향적인", "외향적인", "지적인", "감성적인",
                "카리스마", "순수한", "신비한", "도전적인", "안정적인"
            ]
        }
    
    def encode_single_text(self, text: str, enhance_fragrance_terms: bool = True) -> np.ndarray:
        """단일 텍스트를 벡터로 인코딩"""
        try:
            if enhance_fragrance_terms:
                text = self._enhance_fragrance_text(text)
            
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode([text], convert_to_numpy=True)[0]
            else:
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                    embedding = embedding.cpu().numpy().squeeze()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """배치 텍스트를 벡터로 인코딩"""
        try:
            enhanced_texts = [self._enhance_fragrance_text(text) for text in texts]
            
            if isinstance(self.model, SentenceTransformer):
                embeddings = self.model.encode(
                    enhanced_texts, 
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
            else:
                embeddings = []
                for i in range(0, len(enhanced_texts), batch_size):
                    batch = enhanced_texts[i:i + batch_size]
                    batch_embeddings = self._encode_batch_transformers(batch)
                    embeddings.extend(batch_embeddings)
                embeddings = np.array(embeddings)
            
            logger.info(f"Encoded {len(texts)} texts into embeddings of shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise
    
    def _encode_batch_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """Transformers 모델로 배치 인코딩"""
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            return embeddings.cpu().numpy()
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """평균 풀링"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _enhance_fragrance_text(self, text: str) -> str:
        """향수 관련 용어 강화"""
        enhanced_text = text
        
        # Add contextual terms based on fragrance vocabulary
        for category, terms in self.fragrance_vocab.items():
            for term in terms:
                if term in text:
                    # Add semantic context
                    if category == "top_notes":
                        enhanced_text += " 상단노트 첫인상"
                    elif category == "heart_notes":
                        enhanced_text += " 중간노트 핵심향"
                    elif category == "base_notes":
                        enhanced_text += " 베이스노트 지속향"
        
        return enhanced_text
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray, 
        top_k: int = 5
    ) -> List[int]:
        """가장 유사한 임베딩들의 인덱스 반환"""
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [idx for idx, _ in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Failed to find most similar: {e}")
            return []
    
    def create_fragrance_profile_embedding(self, fragrance_data: Dict[str, Any]) -> np.ndarray:
        """향수 프로필 전체를 종합한 임베딩 생성"""
        try:
            # Combine all fragrance information
            profile_text = []
            
            if "name" in fragrance_data:
                profile_text.append(f"향수명: {fragrance_data['name']}")
            
            if "brand" in fragrance_data:
                profile_text.append(f"브랜드: {fragrance_data['brand']}")
            
            if "description" in fragrance_data:
                profile_text.append(f"설명: {fragrance_data['description']}")
            
            if "top_notes" in fragrance_data:
                profile_text.append(f"톱노트: {', '.join(fragrance_data['top_notes'])}")
            
            if "heart_notes" in fragrance_data:
                profile_text.append(f"미들노트: {', '.join(fragrance_data['heart_notes'])}")
            
            if "base_notes" in fragrance_data:
                profile_text.append(f"베이스노트: {', '.join(fragrance_data['base_notes'])}")
            
            if "mood_tags" in fragrance_data:
                profile_text.append(f"무드: {', '.join(fragrance_data['mood_tags'])}")
            
            if "season" in fragrance_data:
                profile_text.append(f"계절: {fragrance_data['season']}")
            
            combined_text = " ".join(profile_text)
            return self.encode_single_text(combined_text)
            
        except Exception as e:
            logger.error(f"Failed to create fragrance profile embedding: {e}")
            raise
    
    def semantic_search_fragrance(
        self,
        query: str,
        fragrance_embeddings: np.ndarray,
        fragrance_metadata: List[Dict[str, Any]],
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """향수 특화 의미 검색"""
        try:
            query_embedding = self.encode_single_text(query)
            
            results = []
            for i, candidate_embedding in enumerate(fragrance_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate_embedding)
                
                if similarity >= min_similarity:
                    result = {
                        "index": i,
                        "similarity": similarity,
                        "metadata": fragrance_metadata[i] if i < len(fragrance_metadata) else {}
                    }
                    results.append(result)
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []