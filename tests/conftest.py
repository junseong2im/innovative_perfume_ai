import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path
import json

# 테스트용 임시 디렉토리
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# 샘플 임베딩 데이터
@pytest.fixture
def sample_embeddings():
    """샘플 임베딩 데이터 생성"""
    np.random.seed(42)
    embeddings1 = np.random.randn(100, 384)  # 100 samples, 384 dimensions
    embeddings2 = np.random.randn(100, 384)
    labels = np.random.choice([0, 1], size=100)
    return embeddings1, embeddings2, labels

# 샘플 검색 데이터
@pytest.fixture
def sample_retrieval_data():
    """샘플 검색 데이터 생성"""
    np.random.seed(42)
    n_queries, n_docs = 50, 1000
    predictions = np.random.rand(n_queries, n_docs)
    ground_truth = np.zeros((n_queries, n_docs))
    
    # 각 쿼리마다 랜덤하게 몇 개의 문서를 관련 문서로 설정
    for i in range(n_queries):
        n_relevant = np.random.randint(1, 10)
        relevant_indices = np.random.choice(n_docs, size=n_relevant, replace=False)
        ground_truth[i, relevant_indices] = 1
    
    return predictions, ground_truth

# 샘플 생성 데이터
@pytest.fixture
def sample_generation_data():
    """샘플 생성 데이터"""
    generated_texts = [
        "This is a fresh and romantic spring fragrance with citrus notes.",
        "A sophisticated woody scent perfect for evening wear.",
        "Light floral fragrance with delicate rose and jasmine.",
        "Deep and mysterious oriental fragrance with amber and musk.",
        "Clean and fresh aquatic fragrance for daily use."
    ]
    
    reference_texts = [
        "Fresh spring fragrance featuring bright citrus elements.",
        "Elegant woody evening fragrance with sophisticated appeal.",
        "Delicate floral scent with rose and jasmine heart notes.",
        "Rich oriental fragrance with warm amber and musk base.",
        "Clean aquatic fragrance suitable for everyday wear."
    ]
    
    quality_scores = [8.5, 7.8, 9.2, 8.1, 7.5]
    
    return generated_texts, reference_texts, quality_scores

# Mock 설정
@pytest.fixture
def mock_embedding_model():
    """Mock 임베딩 모델"""
    mock = Mock()
    mock.encode_batch.return_value = np.random.randn(10, 384)
    mock.encode_query.return_value = np.random.randn(384)
    return mock

@pytest.fixture
def mock_generation_model():
    """Mock 생성 모델"""
    mock = Mock()
    mock.generate.return_value = {
        "name": "Test Fragrance",
        "description": "A test fragrance for unit testing",
        "notes": {
            "top": ["bergamot", "lemon"],
            "middle": ["rose", "jasmine"],
            "base": ["musk", "cedar"]
        }
    }
    return mock

# 테스트 데이터 파일 생성
@pytest.fixture
def test_data_files(temp_dir):
    """테스트용 데이터 파일들 생성"""
    files = {}
    
    # 임베딩 훈련 데이터
    embedding_data = [
        {
            "query": "fresh citrus fragrance",
            "document": "bright lemon and bergamot scent",
            "label": 1
        },
        {
            "query": "romantic floral scent",
            "document": "elegant rose and jasmine fragrance",
            "label": 1
        },
        {
            "query": "fresh citrus fragrance",
            "document": "heavy oriental musk and amber",
            "label": 0
        }
    ] * 100  # 300개 샘플
    
    embedding_file = temp_dir / "embedding_data.json"
    with open(embedding_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_data, f, ensure_ascii=False, indent=2)
    files['embedding'] = embedding_file
    
    # 생성 훈련 데이터
    generation_data = [
        {
            "prompt": "Create a fresh spring fragrance",
            "response": '{"name": "Spring Fresh", "notes": {"top": ["lemon"], "middle": ["rose"], "base": ["musk"]}}'
        },
        {
            "prompt": "Design a romantic evening scent",
            "response": '{"name": "Evening Romance", "notes": {"top": ["bergamot"], "middle": ["jasmine"], "base": ["amber"]}}'
        }
    ] * 150  # 300개 샘플
    
    generation_file = temp_dir / "generation_data.json"
    with open(generation_file, 'w', encoding='utf-8') as f:
        json.dump(generation_data, f, ensure_ascii=False, indent=2)
    files['generation'] = generation_file
    
    return files