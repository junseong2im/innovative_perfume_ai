"""
Model Manager - 싱글톤 패턴으로 모델 관리
무거운 AI 모델들을 한 번만 로드하고 재사용
"""

import logging
from typing import Dict, Any, Optional
import threading
import torch
import gc
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """
    싱글톤 모델 매니저
    
    - 모든 AI 모델을 중앙에서 관리
    - 리소스 효율적 사용
    - 메모리 최적화
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager"""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self._models: Dict[str, Any] = {}
            self._model_configs: Dict[str, Dict] = {}
            self._load_counts: Dict[str, int] = {}  # 사용 통계
            
            # GPU 사용 가능 체크
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"ModelManager initialized with device: {self.device}")
            
            self._initialized = True
    
    def get_model(self, model_name: str, lazy_load: bool = True) -> Any:
        """
        모델 가져오기
        
        Args:
            model_name: 모델 이름
            lazy_load: 지연 로딩 사용 여부
            
        Returns:
            로드된 모델 인스턴스
        """
        # 이미 로드된 모델이면 반환
        if model_name in self._models:
            self._load_counts[model_name] = self._load_counts.get(model_name, 0) + 1
            logger.debug(f"Returning cached model: {model_name} (used {self._load_counts[model_name]} times)")
            return self._models[model_name]
        
        # Lazy loading
        if lazy_load:
            logger.info(f"Lazy loading model: {model_name}")
            model = self._load_model(model_name)
            if model:
                self._models[model_name] = model
                self._load_counts[model_name] = 1
            return model
        
        return None
    
    def _load_model(self, model_name: str) -> Any:
        """
        실제 모델 로드
        
        Args:
            model_name: 모델 이름
            
        Returns:
            로드된 모델
        """
        try:
            if model_name == "scientific_validator":
                from ..tools.validator_tool import ScientificValidator
                logger.info("Loading ScientificValidator...")
                return ScientificValidator()
                
            elif model_name == "fragrance_generator":
                from ..tools.generator_tool import get_generator
                logger.info("Loading FragranceGenerator...")
                return get_generator()
                
            elif model_name == "embedding_model":
                from ..models.embedding import AdvancedKoreanFragranceEmbedding
                logger.info("Loading Embedding Model...")
                return AdvancedKoreanFragranceEmbedding()
                
            elif model_name == "rag_system":
                from ..models.rag_system import FragranceRAGSystem, RAGMode
                logger.info("Loading RAG System...")
                return FragranceRAGSystem(rag_mode=RAGMode.HYBRID_RETRIEVAL)
                
            elif model_name == "master_perfumer":
                from ..models.master_perfumer import MasterPerfumerAI
                logger.info("Loading Master Perfumer AI...")
                model = MasterPerfumerAI()
                if self.device.type == 'cuda':
                    model = model.cuda()
                return model
                
            elif model_name == "ollama_client":
                from ..llm.ollama_client import OllamaClient
                logger.info("Loading Ollama Client...")
                return OllamaClient()
                
            elif model_name == "description_llm":
                from ..llm.perfume_description_llm import get_description_llm
                logger.info("Loading Description LLM...")
                return get_description_llm()
                
            else:
                logger.warning(f"Unknown model: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def preload_models(self, model_names: list):
        """
        모델 미리 로드
        
        Args:
            model_names: 로드할 모델 이름 리스트
        """
        logger.info(f"Preloading {len(model_names)} models...")
        
        for model_name in model_names:
            try:
                self.get_model(model_name, lazy_load=True)
                logger.info(f"Preloaded: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload {model_name}: {e}")
    
    def release_model(self, model_name: str):
        """
        모델 릴리스 (메모리 해제)
        
        Args:
            model_name: 릴리스할 모델 이름
        """
        if model_name in self._models:
            logger.info(f"Releasing model: {model_name}")
            
            # PyTorch 모델인 경우 GPU 메모리 해제
            model = self._models[model_name]
            if hasattr(model, 'cuda'):
                del model
                torch.cuda.empty_cache()
                
            del self._models[model_name]
            gc.collect()
            
            logger.info(f"Model {model_name} released")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        메모리 사용량 조회
        
        Returns:
            메모리 사용 정보
        """
        memory_info = {
            "loaded_models": list(self._models.keys()),
            "model_count": len(self._models),
            "load_counts": self._load_counts
        }
        
        if torch.cuda.is_available():
            memory_info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            memory_info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            
        return memory_info
    
    def cleanup(self):
        """
        전체 모델 정리
        """
        logger.info("Cleaning up all models...")
        
        model_names = list(self._models.keys())
        for model_name in model_names:
            self.release_model(model_name)
            
        self._models.clear()
        self._load_counts.clear()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        logger.info("Cleanup completed")


# 전역 인스턴스 생성 함수
def get_model_manager() -> ModelManager:
    """
    ModelManager 싱글톤 인스턴스 가져오기
    
    Returns:
        ModelManager 인스턴스
    """
    return ModelManager()
