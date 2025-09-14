import pytest
import numpy as np
from fragrance_ai.evaluation.mathematical_metrics import (
    MathematicalMetrics, MetricType, StatisticalResult, MetricResult
)

class TestMathematicalMetrics:
    """수학적 메트릭 테스트"""
    
    @pytest.fixture
    def metrics_calculator(self):
        """메트릭 계산기 인스턴스"""
        return MathematicalMetrics(confidence_level=0.95)
    
    def test_retrieval_metrics_basic(self, metrics_calculator, sample_retrieval_data):
        """기본 검색 메트릭 테스트"""
        predictions, ground_truth = sample_retrieval_data
        
        metrics = metrics_calculator.compute_retrieval_metrics(
            predictions=predictions,
            ground_truth=ground_truth,
            k_values=[1, 5, 10],
            compute_statistical_tests=True
        )
        
        # 필수 메트릭이 모두 계산되었는지 확인
        expected_metrics = [
            'precision@1', 'precision@5', 'precision@10',
            'recall@1', 'recall@5', 'recall@10',
            'map@1', 'map@5', 'map@10',
            'ndcg@1', 'ndcg@5', 'ndcg@10',
            'mrr'
        ]
        
        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], MetricResult)
            assert 0 <= metrics[metric_name].value <= 1  # 모든 메트릭은 0-1 범위
        
        # Precision@1 <= Precision@5 <= Precision@10 (일반적으로)
        # assert metrics['precision@1'].value <= metrics['precision@5'].value
        
        # 통계적 검정 결과 확인
        for metric_name in expected_metrics:
            result = metrics[metric_name]
            if result.statistical_result:
                assert isinstance(result.statistical_result, StatisticalResult)
                assert result.statistical_result.confidence_interval is not None
                assert result.statistical_result.p_value is not None
    
    def test_retrieval_metrics_edge_cases(self, metrics_calculator):
        """검색 메트릭 경계 조건 테스트"""
        # 모든 문서가 관련 있는 경우
        predictions = np.random.rand(5, 10)
        ground_truth = np.ones((5, 10))
        
        metrics = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth, k_values=[5]
        )
        
        # Precision@5와 Recall@5는 1.0이어야 함
        assert metrics['precision@5'].value == 1.0
        assert metrics['recall@5'].value == 1.0
        
        # 관련 문서가 없는 경우
        ground_truth_zero = np.zeros((5, 10))
        
        metrics_zero = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth_zero, k_values=[5]
        )
        
        # 모든 메트릭이 0이어야 함
        assert metrics_zero['precision@5'].value == 0.0
        assert metrics_zero['recall@5'].value == 0.0
    
    def test_generation_metrics_basic(self, metrics_calculator, sample_generation_data):
        """기본 생성 메트릭 테스트"""
        generated_texts, reference_texts, quality_scores = sample_generation_data
        
        metrics = metrics_calculator.compute_generation_metrics(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
            quality_scores=quality_scores,
            compute_statistical_tests=True
        )
        
        # 기본 메트릭 확인
        assert 'quality_score' in metrics
        assert 'avg_length' in metrics
        assert 'bleu_score' in metrics
        assert 'semantic_similarity' in metrics
        
        # 다양성 메트릭 확인
        diversity_metrics = [k for k in metrics.keys() if k.startswith('diversity_')]
        assert len(diversity_metrics) > 0
        
        # 값 범위 확인
        assert 0 <= metrics['bleu_score'].value <= 1
        assert 0 <= metrics['semantic_similarity'].value <= 1
        assert metrics['avg_length'].value > 0
    
    def test_embedding_metrics_basic(self, metrics_calculator, sample_embeddings):
        """기본 임베딩 메트릭 테스트"""
        embeddings1, embeddings2, labels = sample_embeddings
        
        metrics = metrics_calculator.compute_embedding_metrics(
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            labels=labels,
            distance_metric='cosine',
            compute_statistical_tests=True
        )
        
        # 기본 메트릭 확인
        assert 'mean_similarity' in metrics
        assert 'mean_distance' in metrics
        
        # 분류 성능 메트릭 확인 (labels가 제공되었으므로)
        classification_metrics = [
            'classification_precision', 'classification_recall',
            'classification_f1', 'classification_accuracy'
        ]
        
        for metric_name in classification_metrics:
            assert metric_name in metrics
            assert 0 <= metrics[metric_name].value <= 1
        
        # 거리와 유사도 관계 확인
        mean_similarity = metrics['mean_similarity'].value
        mean_distance = metrics['mean_distance'].value
        
        # 코사인 거리의 경우: distance = 1 - similarity
        assert abs((mean_similarity + mean_distance) - 1.0) < 1e-6
    
    def test_embedding_metrics_euclidean(self, metrics_calculator, sample_embeddings):
        """유클리드 거리 기반 임베딩 메트릭 테스트"""
        embeddings1, embeddings2, labels = sample_embeddings
        
        metrics = metrics_calculator.compute_embedding_metrics(
            embeddings1=embeddings1,
            embeddings2=embeddings2,
            labels=labels,
            distance_metric='euclidean'
        )
        
        assert 'mean_similarity' in metrics
        assert 'mean_distance' in metrics
        assert metrics['mean_distance'].value >= 0  # 유클리드 거리는 항상 비음수
    
    def test_statistical_significance_paired_ttest(self, metrics_calculator):
        """대응 t-검정 테스트"""
        # 첫 번째 모델이 더 좋은 성능을 보이는 경우
        scores1 = [0.8, 0.75, 0.82, 0.79, 0.81, 0.78, 0.83, 0.77, 0.80, 0.76]
        scores2 = [0.7, 0.68, 0.72, 0.69, 0.71, 0.67, 0.73, 0.66, 0.70, 0.65]
        
        result = metrics_calculator.compute_statistical_significance(
            scores1=scores1,
            scores2=scores2,
            test_type='paired_ttest'
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.value > 0  # scores1이 더 높으므로 차이는 양수
        assert result.p_value is not None
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        
        # 95% 신뢰구간이 0을 포함하지 않으면 유의미한 차이
        ci_lower, ci_upper = result.confidence_interval
        if ci_lower > 0:
            assert result.is_significant
    
    def test_statistical_significance_wilcoxon(self, metrics_calculator):
        """윌콕슨 부호순위 검정 테스트"""
        scores1 = [0.8, 0.75, 0.82, 0.79, 0.81]
        scores2 = [0.7, 0.68, 0.72, 0.69, 0.71]
        
        result = metrics_calculator.compute_statistical_significance(
            scores1=scores1,
            scores2=scores2,
            test_type='wilcoxon'
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.p_value is not None
    
    def test_input_validation(self, metrics_calculator):
        """입력 검증 테스트"""
        # 잘못된 shape의 검색 데이터
        predictions = np.random.rand(10, 5)
        ground_truth = np.random.randint(0, 2, (10, 8))  # 다른 shape
        
        with pytest.raises(ValueError):
            metrics_calculator.compute_retrieval_metrics(predictions, ground_truth)
        
        # 잘못된 ground_truth 값
        predictions = np.random.rand(10, 5)
        ground_truth = np.random.rand(10, 5)  # 0/1이 아닌 값
        
        with pytest.raises(ValueError):
            metrics_calculator.compute_retrieval_metrics(predictions, ground_truth)
        
        # 잘못된 shape의 임베딩 데이터
        embeddings1 = np.random.rand(10, 384)
        embeddings2 = np.random.rand(10, 512)  # 다른 차원
        
        with pytest.raises(ValueError):
            metrics_calculator.compute_embedding_metrics(embeddings1, embeddings2)
    
    def test_ndcg_calculation_accuracy(self, metrics_calculator):
        """NDCG 계산 정확도 테스트"""
        # 간단한 예제로 NDCG 계산 검증
        predictions = np.array([[3.0, 2.0, 1.0, 0.0]])  # 1개 쿼리, 4개 문서
        ground_truth = np.array([[1, 1, 0, 0]])  # 첫 두 문서가 관련 문서
        
        metrics = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth, k_values=[2]
        )
        
        # 이 경우 예상 NDCG@2 계산:
        # DCG = 1 + 1/log2(2) = 1 + 1 = 2
        # IDCG = 1 + 1/log2(2) = 1 + 1 = 2  (최적 순서도 동일)
        # NDCG = DCG/IDCG = 2/2 = 1.0
        
        ndcg_2 = metrics['ndcg@2'].value
        assert abs(ndcg_2 - 1.0) < 1e-6  # 거의 1.0이어야 함
    
    def test_precision_recall_relationship(self, metrics_calculator):
        """Precision과 Recall 관계 테스트"""
        # 모든 검색된 문서가 관련 있는 경우
        predictions = np.array([[1.0, 0.9, 0.8, 0.1, 0.0]])
        ground_truth = np.array([[1, 1, 1, 0, 0]])  # 처음 3개가 관련 문서
        
        metrics = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth, k_values=[2, 3]
        )
        
        # k=2일 때: 2개 검색, 2개 관련 -> Precision=1.0, Recall=2/3
        assert abs(metrics['precision@2'].value - 1.0) < 1e-6
        assert abs(metrics['recall@2'].value - 2.0/3.0) < 1e-6
        
        # k=3일 때: 3개 검색, 3개 관련 -> Precision=1.0, Recall=1.0
        assert abs(metrics['precision@3'].value - 1.0) < 1e-6
        assert abs(metrics['recall@3'].value - 1.0) < 1e-6
    
    def test_confidence_intervals(self, metrics_calculator):
        """신뢰구간 계산 테스트"""
        # 충분한 샘플로 신뢰구간 테스트
        np.random.seed(42)
        scores = np.random.normal(0.75, 0.05, 50)  # 평균 0.75, 표준편차 0.05
        
        result = metrics_calculator._create_metric_result(
            name="test_metric",
            scores=scores.tolist(),
            compute_statistical_tests=True
        )
        
        assert result.statistical_result is not None
        ci_lower, ci_upper = result.statistical_result.confidence_interval
        
        # 신뢰구간이 평균 주변에 있는지 확인
        mean_score = result.value
        assert ci_lower < mean_score < ci_upper
        
        # 신뢰구간의 크기가 합리적인지 확인 (너무 크거나 작지 않은지)
        ci_width = ci_upper - ci_lower
        assert 0.001 < ci_width < 0.5  # 실제 데이터에 따라 조정 가능
    
    def test_metric_consistency(self, metrics_calculator):
        """메트릭 일관성 테스트"""
        np.random.seed(42)
        
        # 동일한 데이터로 여러 번 계산했을 때 결과가 일관되는지 확인
        predictions = np.random.rand(20, 100)
        ground_truth = np.random.randint(0, 2, (20, 100))
        
        metrics1 = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth, k_values=[5, 10]
        )
        
        metrics2 = metrics_calculator.compute_retrieval_metrics(
            predictions, ground_truth, k_values=[5, 10]
        )
        
        # 동일한 입력에 대해 동일한 결과가 나와야 함
        for metric_name in metrics1.keys():
            assert abs(metrics1[metric_name].value - metrics2[metric_name].value) < 1e-10