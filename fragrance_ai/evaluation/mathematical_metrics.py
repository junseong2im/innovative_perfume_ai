"""
정확한 수학적 성능 검증을 위한 메트릭 모듈
과학적 정밀도와 통계적 유의성을 보장하는 평가 지표
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """메트릭 타입 정의"""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SIMILARITY = "similarity"

@dataclass
class StatisticalResult:
    """통계적 검정 결과"""
    value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significance_level: float = 0.05
    is_significant: bool = None
    
    def __post_init__(self):
        if self.is_significant is None:
            self.is_significant = self.p_value < self.significance_level

@dataclass
class MetricResult:
    """메트릭 계산 결과"""
    name: str
    value: float
    statistical_result: Optional[StatisticalResult] = None
    std_error: Optional[float] = None
    sample_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class MathematicalMetrics:
    """정확한 수학적 메트릭 계산 클래스"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def compute_retrieval_metrics(
        self,
        predictions: np.ndarray,  # 예측 점수나 순위
        ground_truth: np.ndarray,  # 실제 관련성 (0/1)
        k_values: List[int] = [1, 5, 10, 20],
        compute_statistical_tests: bool = True
    ) -> Dict[str, MetricResult]:
        """
        정보 검색 메트릭 계산
        
        Args:
            predictions: 예측 점수 또는 순위 (N, M) - N개 쿼리, M개 문서
            ground_truth: 실제 관련성 (N, M) - 1: 관련, 0: 비관련
            k_values: 평가할 k 값들
            compute_statistical_tests: 통계적 검정 수행 여부
        """
        metrics = {}
        
        # 입력 검증
        self._validate_retrieval_inputs(predictions, ground_truth)
        
        n_queries = predictions.shape[0]
        
        # 각 k 값에 대해 메트릭 계산
        for k in k_values:
            # Precision@K
            precision_k_scores = []
            recall_k_scores = []
            ap_k_scores = []
            ndcg_k_scores = []
            
            for i in range(n_queries):
                # 상위 k개 문서 인덱스
                top_k_indices = np.argsort(predictions[i])[::-1][:k]
                
                # Precision@K
                relevant_retrieved = np.sum(ground_truth[i, top_k_indices])
                precision_k = relevant_retrieved / k if k > 0 else 0.0
                precision_k_scores.append(precision_k)
                
                # Recall@K
                total_relevant = np.sum(ground_truth[i])
                recall_k = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
                recall_k_scores.append(recall_k)
                
                # Average Precision@K
                ap_k = self._calculate_average_precision_k(
                    predictions[i], ground_truth[i], k
                )
                ap_k_scores.append(ap_k)
                
                # NDCG@K
                ndcg_k = self._calculate_ndcg_k(
                    predictions[i], ground_truth[i], k
                )
                ndcg_k_scores.append(ndcg_k)
            
            # 평균 및 통계적 검정
            metrics[f'precision@{k}'] = self._create_metric_result(
                f'precision@{k}', precision_k_scores, compute_statistical_tests
            )
            metrics[f'recall@{k}'] = self._create_metric_result(
                f'recall@{k}', recall_k_scores, compute_statistical_tests
            )
            metrics[f'map@{k}'] = self._create_metric_result(
                f'map@{k}', ap_k_scores, compute_statistical_tests
            )
            metrics[f'ndcg@{k}'] = self._create_metric_result(
                f'ndcg@{k}', ndcg_k_scores, compute_statistical_tests
            )
        
        # MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for i in range(n_queries):
            sorted_indices = np.argsort(predictions[i])[::-1]
            for rank, doc_idx in enumerate(sorted_indices):
                if ground_truth[i, doc_idx] == 1:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0.0)
        
        metrics['mrr'] = self._create_metric_result(
            'mrr', mrr_scores, compute_statistical_tests
        )
        
        return metrics
    
    def compute_generation_metrics(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
        quality_scores: Optional[List[float]] = None,
        diversity_window: int = 4,
        compute_statistical_tests: bool = True
    ) -> Dict[str, MetricResult]:
        """
        텍스트 생성 메트릭 계산
        
        Args:
            generated_texts: 생성된 텍스트들
            reference_texts: 참조 텍스트들 (옵션)
            quality_scores: 품질 점수들 (옵션)
            diversity_window: 다양성 계산을 위한 n-gram 윈도우
            compute_statistical_tests: 통계적 검정 수행 여부
        """
        metrics = {}
        n_texts = len(generated_texts)
        
        # 품질 메트릭 (제공된 경우)
        if quality_scores:
            metrics['quality_score'] = self._create_metric_result(
                'quality_score', quality_scores, compute_statistical_tests
            )
        
        # 다양성 메트릭
        diversity_scores = self._calculate_diversity_scores(
            generated_texts, diversity_window
        )
        
        for n_gram_size, scores in diversity_scores.items():
            metrics[f'diversity_{n_gram_size}gram'] = self._create_metric_result(
                f'diversity_{n_gram_size}gram', scores, compute_statistical_tests
            )
        
        # 길이 관련 메트릭
        text_lengths = [len(text.split()) for text in generated_texts]
        metrics['avg_length'] = self._create_metric_result(
            'avg_length', text_lengths, compute_statistical_tests
        )
        
        # 참조 텍스트와 비교 메트릭 (제공된 경우)
        if reference_texts and len(reference_texts) == n_texts:
            # BLEU 점수 (간단 버전)
            bleu_scores = []
            for gen_text, ref_text in zip(generated_texts, reference_texts):
                bleu_score = self._calculate_simple_bleu(gen_text, ref_text)
                bleu_scores.append(bleu_score)
            
            metrics['bleu_score'] = self._create_metric_result(
                'bleu_score', bleu_scores, compute_statistical_tests
            )
            
            # 의미적 유사도
            semantic_similarities = []
            for gen_text, ref_text in zip(generated_texts, reference_texts):
                similarity = self._calculate_semantic_similarity(gen_text, ref_text)
                semantic_similarities.append(similarity)
            
            metrics['semantic_similarity'] = self._create_metric_result(
                'semantic_similarity', semantic_similarities, compute_statistical_tests
            )
        
        return metrics
    
    def compute_embedding_metrics(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        labels: Optional[np.ndarray] = None,
        distance_metric: str = 'cosine',
        compute_statistical_tests: bool = True
    ) -> Dict[str, MetricResult]:
        """
        임베딩 품질 메트릭 계산
        
        Args:
            embeddings1: 첫 번째 임베딩 집합
            embeddings2: 두 번째 임베딩 집합
            labels: 페어별 라벨 (유사/비유사)
            distance_metric: 거리 메트릭 ('cosine', 'euclidean')
            compute_statistical_tests: 통계적 검정 수행 여부
        """
        metrics = {}
        
        # 입력 검증
        self._validate_embedding_inputs(embeddings1, embeddings2)
        
        # 거리/유사도 계산
        if distance_metric == 'cosine':
            similarities = np.array([
                1 - cosine(emb1, emb2) 
                for emb1, emb2 in zip(embeddings1, embeddings2)
            ])
            distances = 1 - similarities
        elif distance_metric == 'euclidean':
            distances = np.array([
                euclidean(emb1, emb2)
                for emb1, emb2 in zip(embeddings1, embeddings2)
            ])
            # 정규화된 유사도 (0-1 범위)
            max_dist = np.max(distances)
            similarities = 1 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)
        else:
            raise ValueError(f"지원되지 않는 거리 메트릭: {distance_metric}")
        
        # 기본 통계
        metrics['mean_similarity'] = self._create_metric_result(
            'mean_similarity', similarities.tolist(), compute_statistical_tests
        )
        
        metrics['mean_distance'] = self._create_metric_result(
            'mean_distance', distances.tolist(), compute_statistical_tests
        )
        
        # 라벨이 제공된 경우 분류 성능 평가
        if labels is not None:
            # 임계값 기반 이진 분류
            thresholds = np.linspace(0.1, 0.9, 9)
            best_threshold = 0.5
            best_f1 = 0.0
            
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                f1 = f1_score(labels, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # 최적 임계값으로 예측
            final_predictions = (similarities >= best_threshold).astype(int)
            
            # 분류 메트릭 계산
            precision = precision_score(labels, final_predictions, zero_division=0)
            recall = recall_score(labels, final_predictions, zero_division=0)
            f1 = f1_score(labels, final_predictions, zero_division=0)
            accuracy = accuracy_score(labels, final_predictions)
            
            metrics.update({
                'classification_precision': MetricResult(
                    'classification_precision', precision, sample_size=len(labels)
                ),
                'classification_recall': MetricResult(
                    'classification_recall', recall, sample_size=len(labels)
                ),
                'classification_f1': MetricResult(
                    'classification_f1', f1, sample_size=len(labels)
                ),
                'classification_accuracy': MetricResult(
                    'classification_accuracy', accuracy, sample_size=len(labels)
                ),
                'optimal_threshold': MetricResult(
                    'optimal_threshold', best_threshold, sample_size=len(labels)
                )
            })
            
            # ROC AUC 계산 (예외 처리 포함)
            try:
                roc_auc = roc_auc_score(labels, similarities)
                metrics['roc_auc'] = MetricResult(
                    'roc_auc', roc_auc, sample_size=len(labels)
                )
            except ValueError as e:
                logger.warning(f"ROC AUC 계산 실패: {e}")
        
        return metrics
    
    def compute_statistical_significance(
        self,
        scores1: List[float],
        scores2: List[float],
        test_type: str = 'paired_ttest'
    ) -> StatisticalResult:
        """
        두 모델 간 성능 차이의 통계적 유의성 검정
        
        Args:
            scores1: 첫 번째 모델의 점수들
            scores2: 두 번째 모델의 점수들
            test_type: 검정 타입 ('paired_ttest', 'wilcoxon', 'mann_whitney')
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        
        if test_type == 'paired_ttest':
            if len(scores1) != len(scores2):
                raise ValueError("paired_ttest requires equal length arrays")
            
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            
            # 평균 차이와 신뢰구간
            diff = np.mean(scores1) - np.mean(scores2)
            se_diff = stats.sem(scores1 - scores2)
            
            # 95% 신뢰구간
            df = len(scores1) - 1
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            margin_of_error = t_critical * se_diff
            
            ci = (diff - margin_of_error, diff + margin_of_error)
            
        elif test_type == 'wilcoxon':
            if len(scores1) != len(scores2):
                raise ValueError("wilcoxon test requires equal length arrays")
            
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            
            # 중앙값 차이 (근사적 신뢰구간)
            diff = np.median(scores1) - np.median(scores2)
            # 비모수적 방법이므로 정확한 신뢰구간 계산이 복잡함
            ci = (diff - np.std(scores1 - scores2), diff + np.std(scores1 - scores2))
            
        elif test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
            
            # 효과크기 (r = Z / sqrt(N))
            n1, n2 = len(scores1), len(scores2)
            z_score = stats.norm.ppf(p_value/2) if p_value > 0 else 0
            effect_size = abs(z_score) / np.sqrt(n1 + n2)
            
            diff = effect_size
            ci = (-1.0, 1.0)  # 정확한 계산은 복잡하므로 범위만 표시
            
        else:
            raise ValueError(f"지원되지 않는 검정 타입: {test_type}")
        
        return StatisticalResult(
            value=diff,
            confidence_interval=ci,
            p_value=p_value,
            significance_level=self.alpha
        )
    
    def _validate_retrieval_inputs(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """검색 메트릭 입력 검증"""
        if predictions.shape != ground_truth.shape:
            raise ValueError(f"예측과 실제값의 shape이 다릅니다: {predictions.shape} vs {ground_truth.shape}")
        
        if not np.all(np.isin(ground_truth, [0, 1])):
            raise ValueError("ground_truth는 0과 1로만 구성되어야 합니다")
        
        if len(predictions.shape) != 2:
            raise ValueError("예측값은 2차원 배열이어야 합니다 (n_queries, n_docs)")
    
    def _validate_embedding_inputs(self, embeddings1: np.ndarray, embeddings2: np.ndarray):
        """임베딩 입력 검증"""
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(f"임베딩 shape이 다릅니다: {embeddings1.shape} vs {embeddings2.shape}")
        
        if len(embeddings1.shape) != 2:
            raise ValueError("임베딩은 2차원 배열이어야 합니다 (n_samples, n_features)")
    
    def _calculate_average_precision_k(self, predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        """Average Precision@K 계산"""
        sorted_indices = np.argsort(predictions)[::-1][:k]
        
        relevant_docs = 0
        precision_sum = 0.0
        
        for i, doc_idx in enumerate(sorted_indices):
            if ground_truth[doc_idx] == 1:
                relevant_docs += 1
                precision_sum += relevant_docs / (i + 1)
        
        total_relevant = np.sum(ground_truth)
        if total_relevant == 0:
            return 0.0
        
        return precision_sum / min(total_relevant, k)
    
    def _calculate_ndcg_k(self, predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
        """NDCG@K 계산"""
        sorted_indices = np.argsort(predictions)[::-1][:k]
        
        # DCG 계산
        dcg = 0.0
        for i, doc_idx in enumerate(sorted_indices):
            gain = ground_truth[doc_idx]
            if i == 0:
                dcg += gain
            else:
                dcg += gain / np.log2(i + 1)
        
        # IDCG 계산
        ideal_gains = np.sort(ground_truth)[::-1][:k]
        idcg = 0.0
        for i, gain in enumerate(ideal_gains):
            if i == 0:
                idcg += gain
            else:
                idcg += gain / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_diversity_scores(self, texts: List[str], max_n: int = 4) -> Dict[str, List[float]]:
        """텍스트 다양성 점수 계산"""
        diversity_scores = {}
        
        for n in range(1, max_n + 1):
            all_ngrams = []
            text_ngram_counts = []
            
            for text in texts:
                words = text.lower().split()
                ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)
                text_ngram_counts.append(len(ngrams))
            
            # 전체 유니크한 n-gram의 비율
            if all_ngrams:
                unique_ratio = len(set(all_ngrams)) / len(all_ngrams)
                diversity_scores[n] = [unique_ratio] * len(texts)  # 각 텍스트에 같은 점수 부여
            else:
                diversity_scores[n] = [0.0] * len(texts)
        
        return diversity_scores
    
    def _calculate_simple_bleu(self, generated: str, reference: str, max_n: int = 4) -> float:
        """간단한 BLEU 점수 계산"""
        gen_words = generated.lower().split()
        ref_words = reference.lower().split()
        
        if not gen_words or not ref_words:
            return 0.0
        
        # n-gram precision 계산
        precisions = []
        for n in range(1, max_n + 1):
            gen_ngrams = [' '.join(gen_words[i:i+n]) for i in range(len(gen_words) - n + 1)]
            ref_ngrams = [' '.join(ref_words[i:i+n]) for i in range(len(ref_words) - n + 1)]
            
            if not gen_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngrams)
            precision = matches / len(gen_ngrams)
            precisions.append(precision)
        
        # 기하평균 계산
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_words) / len(gen_words)))
        
        return bp * bleu
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """의미적 유사도 계산 (간단 버전 - 실제로는 임베딩 모델 사용)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard 유사도
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_metric_result(
        self,
        name: str,
        scores: List[float],
        compute_statistical_tests: bool = True
    ) -> MetricResult:
        """메트릭 결과 객체 생성"""
        scores = np.array(scores)
        mean_score = np.mean(scores)
        
        if compute_statistical_tests and len(scores) > 1:
            # 표준오차 계산
            std_error = stats.sem(scores)
            
            # 신뢰구간 계산
            df = len(scores) - 1
            t_critical = stats.t.ppf(1 - self.alpha/2, df) if df > 0 else 1.96
            margin_of_error = t_critical * std_error
            
            ci = (mean_score - margin_of_error, mean_score + margin_of_error)
            
            # 단일 표본 t-검정 (평균이 0과 다른지)
            if std_error > 0:
                t_stat, p_value = stats.ttest_1samp(scores, 0)
            else:
                t_stat, p_value = 0, 1.0
            
            statistical_result = StatisticalResult(
                value=mean_score,
                confidence_interval=ci,
                p_value=p_value
            )
        else:
            statistical_result = None
            std_error = np.std(scores) if len(scores) > 1 else None
        
        return MetricResult(
            name=name,
            value=mean_score,
            statistical_result=statistical_result,
            std_error=std_error,
            sample_size=len(scores)
        )