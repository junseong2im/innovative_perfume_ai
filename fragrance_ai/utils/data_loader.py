from typing import List, Dict, Any, Optional, Tuple, Union
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

logger = logging.getLogger(__name__)

class FragranceDataset(Dataset):
    """향수 데이터셋 클래스"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer=None, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        if self.tokenizer:
            # 텍스트 토크나이징
            if 'text' in item:
                tokenized = self.tokenizer(
                    item['text'],
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                item.update({
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze()
                })
        
        return item

class EmbeddingDataset(Dataset):
    """임베딩 훈련용 데이터셋"""
    
    def __init__(self, queries: List[str], documents: List[str], labels: List[int]):
        self.queries = queries
        self.documents = documents
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'query': self.queries[idx],
            'document': self.documents[idx],
            'label': self.labels[idx]
        }

class GenerationDataset(Dataset):
    """생성 모델 훈련용 데이터셋"""
    
    def __init__(self, prompts: List[str], responses: List[str], tokenizer, max_length: int = 1024):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        # 프롬프트와 응답 결합
        full_text = f"{prompt} {response}"
        
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

class DatasetLoader:
    """데이터셋 로더 클래스"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.csv', '.jsonl']
    
    def load_embedding_dataset(
        self, 
        data_path: Union[str, Path], 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
        """임베딩 모델 훈련용 데이터셋 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            # 샘플 데이터 생성
            logger.warning(f"데이터 파일이 없습니다: {data_path}. 샘플 데이터를 생성합니다.")
            data = self._generate_sample_embedding_data()
        else:
            data = self._load_data_file(data_path)
        
        # 데이터 검증
        self._validate_embedding_data(data)
        
        # 쿼리, 문서, 라벨 분리
        queries = [item['query'] for item in data]
        documents = [item['document'] for item in data]
        labels = [item['label'] for item in data]
        
        # 훈련/검증 분할
        train_queries, val_queries, train_docs, val_docs, train_labels, val_labels = train_test_split(
            queries, documents, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        train_dataset = EmbeddingDataset(train_queries, train_docs, train_labels)
        val_dataset = EmbeddingDataset(val_queries, val_docs, val_labels)
        
        logger.info(f"임베딩 데이터셋 로드 완료: 훈련 {len(train_dataset)}, 검증 {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def load_generation_dataset(
        self, 
        data_path: Union[str, Path], 
        max_length: int = 1024,
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[GenerationDataset, GenerationDataset]:
        """생성 모델 훈련용 데이터셋 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"데이터 파일이 없습니다: {data_path}. 샘플 데이터를 생성합니다.")
            data = self._generate_sample_generation_data()
        else:
            data = self._load_data_file(data_path)
        
        # 데이터 검증
        self._validate_generation_data(data)
        
        # 프롬프트와 응답 분리
        prompts = [item['prompt'] for item in data]
        responses = [item['response'] for item in data]
        
        # 훈련/검증 분할
        train_prompts, val_prompts, train_responses, val_responses = train_test_split(
            prompts, responses, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # 토크나이저는 실제 사용 시에 주입될 예정
        train_dataset = GenerationDataset(train_prompts, train_responses, None, max_length)
        val_dataset = GenerationDataset(val_prompts, val_responses, None, max_length)
        
        logger.info(f"생성 데이터셋 로드 완료: 훈련 {len(train_dataset)}, 검증 {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def load_embedding_eval_dataset(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """임베딩 모델 평가용 데이터셋 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"평가 데이터 파일이 없습니다: {data_path}. 샘플 데이터를 생성합니다.")
            return self._generate_sample_embedding_eval_data()
        
        data = self._load_data_file(data_path)
        self._validate_embedding_eval_data(data)
        
        logger.info(f"임베딩 평가 데이터셋 로드 완료: {len(data)}개 항목")
        return data
    
    def load_generation_eval_dataset(self, data_path: Union[str, Path]) -> List[str]:
        """생성 모델 평가용 프롬프트 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"평가 데이터 파일이 없습니다: {data_path}. 샘플 프롬프트를 생성합니다.")
            return self._generate_sample_generation_prompts()
        
        data = self._load_data_file(data_path)
        
        # 프롬프트만 추출
        if isinstance(data[0], dict) and 'prompt' in data[0]:
            prompts = [item['prompt'] for item in data]
        elif isinstance(data[0], str):
            prompts = data
        else:
            raise ValueError("지원되지 않는 프롬프트 데이터 형식")
        
        logger.info(f"생성 평가 프롬프트 로드 완료: {len(prompts)}개")
        return prompts
    
    def load_search_eval_dataset(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """검색 시스템 평가용 데이터셋 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"검색 평가 데이터 파일이 없습니다: {data_path}. 샘플 데이터를 생성합니다.")
            return self._generate_sample_search_eval_data()
        
        data = self._load_data_file(data_path)
        self._validate_search_eval_data(data)
        
        logger.info(f"검색 평가 데이터셋 로드 완료: {len(data)}개 쿼리")
        return data
    
    def load_e2e_eval_dataset(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """End-to-end 평가용 시나리오 로드"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"E2E 평가 데이터 파일이 없습니다: {data_path}. 샘플 시나리오를 생성합니다.")
            return self._generate_sample_e2e_scenarios()
        
        data = self._load_data_file(data_path)
        self._validate_e2e_eval_data(data)
        
        logger.info(f"E2E 평가 시나리오 로드 완료: {len(data)}개")
        return data
    
    def _load_data_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """파일에서 데이터 로드"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif suffix == '.jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"지원되지 않는 파일 형식: {suffix}")
        
        return data
    
    def _validate_embedding_data(self, data: List[Dict[str, Any]]):
        """임베딩 데이터 검증"""
        required_fields = ['query', 'document', 'label']
        
        for i, item in enumerate(data[:5]):  # 처음 5개만 검사
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"임베딩 데이터 항목 {i}에 필수 필드 '{field}'가 없습니다.")
            
            if not isinstance(item['label'], (int, float)):
                raise ValueError(f"항목 {i}의 label은 숫자여야 합니다.")
    
    def _validate_generation_data(self, data: List[Dict[str, Any]]):
        """생성 데이터 검증"""
        required_fields = ['prompt', 'response']
        
        for i, item in enumerate(data[:5]):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"생성 데이터 항목 {i}에 필수 필드 '{field}'가 없습니다.")
    
    def _validate_embedding_eval_data(self, data: List[Dict[str, Any]]):
        """임베딩 평가 데이터 검증"""
        required_fields = ['query', 'document', 'relevance']
        
        for i, item in enumerate(data[:5]):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"임베딩 평가 데이터 항목 {i}에 필수 필드 '{field}'가 없습니다.")
    
    def _validate_search_eval_data(self, data: List[Dict[str, Any]]):
        """검색 평가 데이터 검증"""
        required_fields = ['query']
        
        for i, item in enumerate(data[:5]):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"검색 평가 데이터 항목 {i}에 필수 필드 '{field}'가 없습니다.")
    
    def _validate_e2e_eval_data(self, data: List[Dict[str, Any]]):
        """E2E 평가 데이터 검증"""
        required_fields = ['search_query', 'fragrance_family']
        
        for i, item in enumerate(data[:5]):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"E2E 평가 데이터 항목 {i}에 필수 필드 '{field}'가 없습니다.")
    
    def _generate_sample_embedding_data(self) -> List[Dict[str, Any]]:
        """샘플 임베딩 훈련 데이터 생성"""
        samples = [
            {
                "query": "상큼한 시트러스 향",
                "document": "레몬과 오렌지의 신선한 향이 특징인 시트러스 계열 향수",
                "label": 1
            },
            {
                "query": "로맨틱한 플로럴 향",
                "document": "장미와 재스민의 우아한 꽃 향기",
                "label": 1
            },
            {
                "query": "깊은 우디 향",
                "document": "삼나무와 샌달우드의 깊고 따뜻한 나무 향",
                "label": 1
            },
            {
                "query": "상큼한 시트러스 향",
                "document": "무거운 머스크와 앰버의 관능적인 향",
                "label": 0
            },
            {
                "query": "로맨틱한 플로럴 향",
                "document": "강한 오크모스와 파촐리의 남성적인 향",
                "label": 0
            }
        ]
        
        # 샘플 데이터를 확장하여 100개 생성
        extended_samples = []
        for _ in range(20):
            extended_samples.extend(samples)
        
        return extended_samples
    
    def _generate_sample_generation_data(self) -> List[Dict[str, Any]]:
        """샘플 생성 훈련 데이터 생성"""
        samples = [
            {
                "prompt": "상큼하고 로맨틱한 봄 향수 레시피를 만들어주세요.",
                "response": """
{
  "name": "Spring Romance",
  "description": "상큼한 시트러스와 로맨틱한 플로럴이 어우러진 봄 향수",
  "notes": {
    "top": ["bergamot", "lemon", "pink grapefruit"],
    "middle": ["rose", "peony", "lily of the valley"],
    "base": ["white musk", "cedar", "amber"]
  },
  "formula": {
    "bergamot": "15%",
    "lemon": "10%",
    "pink grapefruit": "5%",
    "rose": "20%",
    "peony": "15%",
    "lily of the valley": "10%",
    "white musk": "15%",
    "cedar": "7%",
    "amber": "3%"
  }
}
"""
            },
            {
                "prompt": "깊고 신비로운 겨울 향수 레시피를 생성해주세요.",
                "response": """
{
  "name": "Winter Mystery",
  "description": "깊은 우디 노트와 스파이시한 요소가 조화된 겨울 향수",
  "notes": {
    "top": ["black pepper", "cardamom", "bergamot"],
    "middle": ["cedar", "sandalwood", "leather"],
    "base": ["oud", "vanilla", "musk"]
  },
  "formula": {
    "black pepper": "5%",
    "cardamom": "8%",
    "bergamot": "12%",
    "cedar": "18%",
    "sandalwood": "15%",
    "leather": "10%",
    "oud": "20%",
    "vanilla": "8%",
    "musk": "4%"
  }
}
"""
            }
        ]
        
        # 샘플 확장
        extended_samples = []
        for _ in range(50):
            extended_samples.extend(samples)
        
        return extended_samples
    
    def _generate_sample_embedding_eval_data(self) -> List[Dict[str, Any]]:
        """샘플 임베딩 평가 데이터 생성"""
        return [
            {
                "query": "상큼한 여름 향수",
                "document": "시트러스 계열의 상큼하고 청량한 여름용 향수",
                "relevance": 1
            },
            {
                "query": "로맨틱한 데이트 향수",
                "document": "장미와 재스민의 우아한 꽃향기로 만든 로맨틱 향수",
                "relevance": 1
            },
            {
                "query": "상큼한 여름 향수",
                "document": "무거운 오리엔탈 계열의 겨울용 향수",
                "relevance": 0
            }
        ]
    
    def _generate_sample_generation_prompts(self) -> List[str]:
        """샘플 생성 평가 프롬프트 생성"""
        return [
            "상큼하고 활기찬 봄 향수 레시피를 만들어주세요.",
            "깊고 세련된 가을 향수 레시피를 생성해주세요.",
            "로맨틱하고 우아한 저녁 향수를 디자인해주세요.",
            "신선하고 깨끗한 오피스 향수를 만들어주세요.",
            "모험적이고 독특한 니치 향수를 창조해주세요."
        ]
    
    def _generate_sample_search_eval_data(self) -> List[Dict[str, Any]]:
        """샘플 검색 평가 데이터 생성"""
        return [
            {
                "query": "상큼한 시트러스 향수",
                "expected_results": ["citrus_fresh_001", "lemon_bright_002"]
            },
            {
                "query": "로맨틱한 플로럴 향수",
                "expected_results": ["rose_romance_001", "jasmine_love_002"]
            }
        ]
    
    def _generate_sample_e2e_scenarios(self) -> List[Dict[str, Any]]:
        """샘플 E2E 평가 시나리오 생성"""
        return [
            {
                "search_query": "봄에 어울리는 가벼운 향수",
                "fragrance_family": "floral",
                "mood": "fresh",
                "intensity": "light",
                "expected_characteristics": ["light", "fresh", "floral"]
            },
            {
                "search_query": "비즈니스 미팅용 세련된 향수",
                "fragrance_family": "woody",
                "mood": "professional",
                "intensity": "moderate",
                "expected_characteristics": ["professional", "sophisticated", "moderate"]
            }
        ]