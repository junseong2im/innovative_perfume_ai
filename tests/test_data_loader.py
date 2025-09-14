import pytest
import json
import pandas as pd
from pathlib import Path
from fragrance_ai.utils.data_loader import DatasetLoader, FragranceDataset, EmbeddingDataset, GenerationDataset

class TestDatasetLoader:
    """데이터셋 로더 테스트"""
    
    @pytest.fixture
    def data_loader(self):
        """데이터셋 로더 인스턴스"""
        return DatasetLoader()
    
    def test_load_embedding_dataset_from_file(self, data_loader, test_data_files):
        """파일에서 임베딩 데이터셋 로드 테스트"""
        train_dataset, val_dataset = data_loader.load_embedding_dataset(
            data_path=test_data_files['embedding'],
            test_size=0.2,
            random_state=42
        )
        
        # 데이터셋 타입 확인
        assert isinstance(train_dataset, EmbeddingDataset)
        assert isinstance(val_dataset, EmbeddingDataset)
        
        # 데이터 분할 비율 확인
        total_size = len(train_dataset) + len(val_dataset)
        val_ratio = len(val_dataset) / total_size
        assert 0.15 <= val_ratio <= 0.25  # 대략 20% (±5%)
        
        # 데이터 구조 확인
        sample_train = train_dataset[0]
        assert 'query' in sample_train
        assert 'document' in sample_train
        assert 'label' in sample_train
        assert sample_train['label'] in [0, 1]
    
    def test_load_generation_dataset_from_file(self, data_loader, test_data_files):
        """파일에서 생성 데이터셋 로드 테스트"""
        train_dataset, val_dataset = data_loader.load_generation_dataset(
            data_path=test_data_files['generation'],
            max_length=1024,
            test_size=0.2,
            random_state=42
        )
        
        # 데이터셋 타입 확인
        assert isinstance(train_dataset, GenerationDataset)
        assert isinstance(val_dataset, GenerationDataset)
        
        # 데이터 구조 확인 (토크나이저가 None이므로 raw 데이터 확인)
        assert len(train_dataset.prompts) > 0
        assert len(train_dataset.responses) > 0
        assert len(train_dataset.prompts) == len(train_dataset.responses)
    
    def test_load_nonexistent_file(self, data_loader, temp_dir):
        """존재하지 않는 파일 처리 테스트"""
        nonexistent_file = temp_dir / "nonexistent.json"
        
        # 샘플 데이터가 생성되어야 함
        train_dataset, val_dataset = data_loader.load_embedding_dataset(
            data_path=nonexistent_file,
            test_size=0.2
        )
        
        assert isinstance(train_dataset, EmbeddingDataset)
        assert isinstance(val_dataset, EmbeddingDataset)
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
    
    def test_load_different_file_formats(self, data_loader, temp_dir):
        """다양한 파일 형식 로드 테스트"""
        # JSON 파일
        json_data = [
            {"query": "test query 1", "document": "test doc 1", "label": 1},
            {"query": "test query 2", "document": "test doc 2", "label": 0}
        ]
        json_file = temp_dir / "test.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        train_ds, val_ds = data_loader.load_embedding_dataset(json_file)
        assert len(train_ds) + len(val_ds) == len(json_data)
        
        # JSONL 파일
        jsonl_file = temp_dir / "test.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in json_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        train_ds, val_ds = data_loader.load_embedding_dataset(jsonl_file)
        assert len(train_ds) + len(val_ds) == len(json_data)
        
        # CSV 파일
        csv_file = temp_dir / "test.csv"
        df = pd.DataFrame(json_data)
        df.to_csv(csv_file, index=False)
        
        train_ds, val_ds = data_loader.load_embedding_dataset(csv_file)
        assert len(train_ds) + len(val_ds) == len(json_data)
    
    def test_data_validation(self, data_loader, temp_dir):
        """데이터 검증 테스트"""
        # 잘못된 임베딩 데이터 (필수 필드 누락)
        invalid_embedding_data = [
            {"query": "test query", "label": 1}  # document 필드 누락
        ]
        
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            json.dump(invalid_embedding_data, f)
        
        with pytest.raises(ValueError, match="필수 필드.*document.*없습니다"):
            data_loader.load_embedding_dataset(invalid_file)
        
        # 잘못된 생성 데이터 (필수 필드 누락)
        invalid_generation_data = [
            {"prompt": "test prompt"}  # response 필드 누락
        ]
        
        with open(invalid_file, 'w') as f:
            json.dump(invalid_generation_data, f)
        
        with pytest.raises(ValueError, match="필수 필드.*response.*없습니다"):
            data_loader.load_generation_dataset(invalid_file)
    
    def test_load_evaluation_datasets(self, data_loader, temp_dir):
        """평가 데이터셋 로드 테스트"""
        # 임베딩 평가 데이터
        embedding_eval_data = [
            {
                "query": "test query",
                "document": "test document",
                "relevance": 1
            }
        ]
        
        eval_file = temp_dir / "eval.json"
        with open(eval_file, 'w') as f:
            json.dump(embedding_eval_data, f)
        
        eval_data = data_loader.load_embedding_eval_dataset(eval_file)
        assert len(eval_data) == 1
        assert eval_data[0]['relevance'] == 1
        
        # 생성 평가 데이터 (프롬프트 리스트)
        generation_prompts = ["prompt 1", "prompt 2", "prompt 3"]
        
        with open(eval_file, 'w') as f:
            json.dump(generation_prompts, f)
        
        prompts = data_loader.load_generation_eval_dataset(eval_file)
        assert len(prompts) == 3
        assert all(isinstance(p, str) for p in prompts)
    
    def test_sample_data_generation(self, data_loader):
        """샘플 데이터 생성 테스트"""
        # 샘플 임베딩 데이터 생성
        sample_embedding = data_loader._generate_sample_embedding_data()
        assert len(sample_embedding) > 0
        assert all('query' in item and 'document' in item and 'label' in item 
                  for item in sample_embedding)
        
        # 샘플 생성 데이터 생성
        sample_generation = data_loader._generate_sample_generation_data()
        assert len(sample_generation) > 0
        assert all('prompt' in item and 'response' in item 
                  for item in sample_generation)
        
        # 샘플 평가 데이터 생성
        sample_eval = data_loader._generate_sample_embedding_eval_data()
        assert len(sample_eval) > 0
        assert all('relevance' in item for item in sample_eval)

class TestFragranceDataset:
    """FragranceDataset 클래스 테스트"""
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        data = [
            {"text": "sample text 1", "label": 1},
            {"text": "sample text 2", "label": 0}
        ]
        
        dataset = FragranceDataset(data)
        assert len(dataset) == 2
        
        sample = dataset[0]
        assert sample["text"] == "sample text 1"
        assert sample["label"] == 1

class TestEmbeddingDataset:
    """EmbeddingDataset 클래스 테스트"""
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        queries = ["query 1", "query 2"]
        documents = ["document 1", "document 2"]
        labels = [1, 0]
        
        dataset = EmbeddingDataset(queries, documents, labels)
        assert len(dataset) == 2
        
        sample = dataset[0]
        assert sample["query"] == "query 1"
        assert sample["document"] == "document 1"
        assert sample["label"] == 1

class TestGenerationDataset:
    """GenerationDataset 클래스 테스트"""
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        prompts = ["prompt 1", "prompt 2"]
        responses = ["response 1", "response 2"]
        
        # 토크나이저 없이 테스트
        dataset = GenerationDataset(prompts, responses, None, 512)
        assert len(dataset) == 2
        
        # 실제 사용 시에는 토크나이저가 주입되어야 함
        sample = dataset[0]  # 토크나이저가 None이면 에러가 발생할 수 있음