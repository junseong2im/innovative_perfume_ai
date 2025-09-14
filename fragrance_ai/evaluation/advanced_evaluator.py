"""
고급 모델 평가 시스템
A/B 테스트, 성능 벤치마킹, 자동 하이퍼파라미터 튜닝 포함
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import optuna
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
import wandb

from ..core.config import settings
from ..models.embedding import AdvancedKoreanFragranceEmbedding
from ..models.generator import FragranceRecipeGenerator
from ..core.vector_store import VectorStore
from .metrics import EvaluationMetrics, QualityAssessment

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_id: str
    name: str
    description: str
    model_type: str  # 'embedding', 'generation', 'search'
    parameters: Dict[str, Any]
    evaluation_metrics: List[str]
    test_data_path: Optional[str] = None
    validation_split: float = 0.2
    random_seed: int = 42
    max_trials: int = 100
    timeout_hours: float = 24.0

@dataclass
class ExperimentResult:
    """실험 결과"""
    experiment_id: str
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    timestamp: datetime
    model_artifacts: Dict[str, Any]
    validation_results: Dict[str, Any]

@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""
    test_id: str
    name: str
    description: str
    control_model: Dict[str, Any]
    variant_models: List[Dict[str, Any]]
    traffic_split: Dict[str, float]  # {'control': 0.5, 'variant_a': 0.3, 'variant_b': 0.2}
    success_metrics: List[str]
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    test_duration_hours: float = 168.0  # 1 week

@dataclass
class ABTestResult:
    """A/B 테스트 결과"""
    test_id: str
    variant_name: str
    sample_size: int
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    lift: float  # % improvement over control
    statistical_power: float

class AdvancedModelEvaluator:
    """고급 모델 평가 시스템"""

    def __init__(
        self,
        results_dir: str = "./evaluation_results",
        enable_wandb: bool = True,
        wandb_project: str = None
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.enable_wandb = enable_wandb
        if enable_wandb:
            wandb.init(
                project=wandb_project or settings.wandb_project,
                config={"evaluation_mode": "advanced"}
            )

        # 실행 중인 실험들
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.active_ab_tests: Dict[str, ABTestConfig] = {}

        # 결과 저장소
        self.experiment_results: List[ExperimentResult] = []
        self.ab_test_results: Dict[str, List[ABTestResult]] = {}

        # 평가 모듈들
        self.metrics_evaluator = EvaluationMetrics()
        self.quality_assessor = QualityAssessment()

        logger.info("AdvancedModelEvaluator initialized")

    async def run_hyperparameter_optimization(
        self,
        config: ExperimentConfig,
        objective_function: Callable,
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """하이퍼파라미터 최적화 실행"""

        logger.info(f"Starting hyperparameter optimization: {config.name}")

        def optimize_objective(trial):
            """Optuna 목적 함수"""
            try:
                # 트라이얼 파라미터 생성
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                # 목적 함수 실행
                start_time = time.time()
                result = asyncio.run(objective_function(params))
                execution_time = time.time() - start_time

                # 결과 저장
                experiment_result = ExperimentResult(
                    experiment_id=config.experiment_id,
                    trial_id=f"trial_{trial.number}",
                    parameters=params,
                    metrics=result['metrics'],
                    execution_time=execution_time,
                    timestamp=datetime.utcnow(),
                    model_artifacts=result.get('artifacts', {}),
                    validation_results=result.get('validation', {})
                )
                self.experiment_results.append(experiment_result)

                # W&B 로깅
                if self.enable_wandb:
                    wandb.log({
                        "trial": trial.number,
                        **params,
                        **result['metrics'],
                        "execution_time": execution_time
                    })

                # 주 메트릭 반환 (최대화 목표)
                primary_metric = config.evaluation_metrics[0]
                return result['metrics'].get(primary_metric, 0.0)

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return 0.0

        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='maximize',
            study_name=config.name,
            sampler=optuna.samplers.TPESampler(seed=config.random_seed)
        )

        # 최적화 실행
        study.optimize(
            optimize_objective,
            n_trials=config.max_trials,
            timeout=config.timeout_hours * 3600
        )

        # 결과 정리
        best_params = study.best_params
        best_value = study.best_value

        optimization_results = {
            'best_parameters': best_params,
            'best_score': best_value,
            'n_trials': len(study.trials),
            'study_stats': {
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration)
            }
        }

        # 최적화 히스토리 시각화
        await self._visualize_optimization_history(study, config)

        logger.info(f"Hyperparameter optimization completed. Best score: {best_value:.4f}")
        return optimization_results

    async def setup_ab_test(self, config: ABTestConfig) -> str:
        """A/B 테스트 설정"""

        logger.info(f"Setting up A/B test: {config.name}")

        # 테스트 검증
        if sum(config.traffic_split.values()) != 1.0:
            raise ValueError("Traffic split must sum to 1.0")

        if 'control' not in config.traffic_split:
            raise ValueError("Control group must be defined")

        # 테스트 등록
        self.active_ab_tests[config.test_id] = config

        # 테스트 설정 저장
        config_file = self.results_dir / f"ab_test_{config.test_id}_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        logger.info(f"A/B test {config.test_id} configured successfully")
        return config.test_id

    async def collect_ab_test_data(
        self,
        test_id: str,
        user_id: str,
        variant: str,
        success_events: Dict[str, Any]
    ) -> None:
        """A/B 테스트 데이터 수집"""

        if test_id not in self.active_ab_tests:
            logger.warning(f"Unknown A/B test: {test_id}")
            return

        # 데이터 포인트 저장
        data_point = {
            'test_id': test_id,
            'user_id': user_id,
            'variant': variant,
            'timestamp': datetime.utcnow().isoformat(),
            'success_events': success_events
        }

        # 파일에 추가
        data_file = self.results_dir / f"ab_test_{test_id}_data.jsonl"
        with open(data_file, 'a') as f:
            f.write(json.dumps(data_point) + '\n')

    async def analyze_ab_test(self, test_id: str) -> Dict[str, ABTestResult]:
        """A/B 테스트 분석"""

        logger.info(f"Analyzing A/B test: {test_id}")

        if test_id not in self.active_ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        config = self.active_ab_tests[test_id]

        # 데이터 로드
        data_file = self.results_dir / f"ab_test_{test_id}_data.jsonl"
        if not data_file.exists():
            raise ValueError(f"No data found for A/B test {test_id}")

        data_points = []
        with open(data_file, 'r') as f:
            for line in f:
                data_points.append(json.loads(line))

        df = pd.DataFrame(data_points)

        # 변형별 분석
        results = {}
        control_data = df[df['variant'] == 'control']

        for variant_name in df['variant'].unique():
            if variant_name == 'control':
                continue

            variant_data = df[df['variant'] == variant_name]

            # 통계적 유의성 검정
            result = await self._perform_statistical_test(
                control_data, variant_data, config.success_metrics
            )

            results[variant_name] = result

        # 결과 저장
        self.ab_test_results[test_id] = list(results.values())

        # 시각화
        await self._visualize_ab_test_results(test_id, results)

        logger.info(f"A/B test analysis completed for {test_id}")
        return results

    async def _perform_statistical_test(
        self,
        control_data: pd.DataFrame,
        variant_data: pd.DataFrame,
        success_metrics: List[str]
    ) -> ABTestResult:
        """통계적 검정 수행"""

        # 주 성공 메트릭으로 분석 (첫 번째 메트릭)
        primary_metric = success_metrics[0]

        # 성공률 계산
        control_successes = sum(
            1 for _, row in control_data.iterrows()
            if row['success_events'].get(primary_metric, False)
        )
        variant_successes = sum(
            1 for _, row in variant_data.iterrows()
            if row['success_events'].get(primary_metric, False)
        )

        control_rate = control_successes / len(control_data) if len(control_data) > 0 else 0
        variant_rate = variant_successes / len(variant_data) if len(variant_data) > 0 else 0

        # 비율 차이 검정 (Z-test)
        n1, n2 = len(control_data), len(variant_data)
        p1, p2 = control_rate, variant_rate

        # 풀링된 비율
        p_pooled = (control_successes + variant_successes) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se > 0:
            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1.0

        # 신뢰구간 계산
        ci_low = p2 - 1.96 * np.sqrt(p2 * (1 - p2) / n2)
        ci_high = p2 + 1.96 * np.sqrt(p2 * (1 - p2) / n2)

        # 효과 크기 (lift)
        lift = ((variant_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0

        # 통계적 검정력 계산 (간단한 근사)
        effect_size = abs(p2 - p1)
        statistical_power = self._calculate_statistical_power(n2, effect_size, 0.05)

        return ABTestResult(
            test_id="",  # 호출하는 쪽에서 설정
            variant_name="",  # 호출하는 쪽에서 설정
            sample_size=n2,
            conversion_rate=variant_rate,
            confidence_interval=(ci_low, ci_high),
            p_value=p_value,
            is_significant=p_value < 0.05,
            lift=lift,
            statistical_power=statistical_power
        )

    def _calculate_statistical_power(
        self,
        sample_size: int,
        effect_size: float,
        alpha: float
    ) -> float:
        """통계적 검정력 계산 (근사)"""
        # 간단한 검정력 계산 (실제로는 더 정교한 계산 필요)
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = (effect_size * np.sqrt(sample_size/2)) - z_alpha
        power = stats.norm.cdf(z_beta)
        return max(0, min(1, power))

    async def run_cross_validation_evaluation(
        self,
        model_factory: Callable,
        data: List[Dict[str, Any]],
        k_folds: int = 5,
        stratify_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """교차 검증 평가"""

        logger.info(f"Starting {k_folds}-fold cross-validation")

        if stratify_field:
            # 층화 교차 검증
            y = [item[stratify_field] for item in data]
            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            splits = list(kf.split(data, y))
        else:
            # 일반 교차 검증
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            splits = list(kf.split(data))

        fold_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Processing fold {fold_idx + 1}/{k_folds}")

            # 데이터 분할
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            try:
                # 모델 생성 및 훈련
                model = await model_factory()

                # 훈련 (모델 타입에 따라 다르게 처리)
                start_time = time.time()
                if hasattr(model, 'train'):
                    await model.train(train_data)
                training_time = time.time() - start_time

                # 검증
                start_time = time.time()
                val_results = await self._evaluate_model_on_data(model, val_data)
                validation_time = time.time() - start_time

                fold_result = {
                    'fold': fold_idx + 1,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'training_time': training_time,
                    'validation_time': validation_time,
                    'metrics': val_results
                }

                fold_results.append(fold_result)

                # W&B 로깅
                if self.enable_wandb:
                    wandb.log({
                        f"fold_{fold_idx}_" + k: v
                        for k, v in val_results.items()
                        if isinstance(v, (int, float))
                    })

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                continue

        # 결과 집계
        cv_results = self._aggregate_cv_results(fold_results)

        logger.info(f"Cross-validation completed. Mean score: {cv_results['mean_score']:.4f}")
        return cv_results

    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """교차 검증 결과 집계"""

        if not fold_results:
            return {'error': 'No successful folds'}

        # 메트릭별 평균/표준편차 계산
        all_metrics = {}
        for fold in fold_results:
            for metric, value in fold['metrics'].items():
                if isinstance(value, (int, float)):
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        aggregated_metrics = {}
        for metric, values in all_metrics.items():
            aggregated_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        # 주요 점수 (첫 번째 메트릭의 평균)
        primary_metric = list(aggregated_metrics.keys())[0]
        mean_score = aggregated_metrics[primary_metric]['mean']

        return {
            'mean_score': mean_score,
            'std_score': aggregated_metrics[primary_metric]['std'],
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'aggregated_metrics': aggregated_metrics,
            'total_training_time': sum(f['training_time'] for f in fold_results),
            'total_validation_time': sum(f['validation_time'] for f in fold_results)
        }

    async def _evaluate_model_on_data(
        self,
        model: Any,
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """데이터에 대한 모델 평가"""

        # 모델 타입별 평가
        if isinstance(model, AdvancedKoreanFragranceEmbedding):
            return await self._evaluate_embedding_model(model, data)
        elif isinstance(model, FragranceRecipeGenerator):
            return await self._evaluate_generation_model(model, data)
        else:
            return await self._evaluate_generic_model(model, data)

    async def _evaluate_embedding_model(
        self,
        model: AdvancedKoreanFragranceEmbedding,
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """임베딩 모델 평가"""

        # 임베딩 생성
        texts = [item['text'] for item in data if 'text' in item]
        embeddings = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = await model.encode_async(batch_texts)
            embeddings.extend(batch_embeddings.embeddings)

        # 클러스터링 품질 평가
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans

        if len(embeddings) > 10:
            # 클러스터 개수 추정
            n_clusters = min(10, len(embeddings) // 5)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                silhouette = silhouette_score(embeddings, cluster_labels)
            else:
                silhouette = 0.0
        else:
            silhouette = 0.0

        # 임베딩 품질 메트릭
        embeddings_array = np.array(embeddings)
        embedding_norm = np.mean(np.linalg.norm(embeddings_array, axis=1))
        embedding_variance = np.var(embeddings_array)

        return {
            'silhouette_score': silhouette,
            'embedding_norm': embedding_norm,
            'embedding_variance': embedding_variance,
            'dimension': embeddings_array.shape[1]
        }

    async def _evaluate_generation_model(
        self,
        model: FragranceRecipeGenerator,
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """생성 모델 평가"""

        results = []

        for item in data[:50]:  # 샘플 제한
            if 'prompt' not in item:
                continue

            try:
                # 레시피 생성
                recipe = model.generate_recipe(
                    prompt=item['prompt'],
                    recipe_type='basic'
                )

                # 품질 평가
                quality_scores = self.quality_assessor.evaluate_recipe_quality(recipe)
                results.append(quality_scores)

            except Exception as e:
                logger.warning(f"Generation failed for prompt: {e}")
                continue

        if not results:
            return {'error': 'No successful generations'}

        # 메트릭 평균 계산
        avg_metrics = {}
        for key in results[0]:
            if isinstance(results[0][key], (int, float)):
                avg_metrics[f'avg_{key}'] = np.mean([r[key] for r in results])

        avg_metrics['generation_success_rate'] = len(results) / min(50, len(data))

        return avg_metrics

    async def _evaluate_generic_model(
        self,
        model: Any,
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """일반 모델 평가"""

        # 기본적인 성능 메트릭
        return {
            'data_processed': len(data),
            'processing_time': time.time()  # 실제로는 측정된 시간
        }

    async def _visualize_optimization_history(
        self,
        study: optuna.Study,
        config: ExperimentConfig
    ) -> None:
        """최적화 히스토리 시각화"""

        try:
            # 최적화 히스토리 플롯
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Hyperparameter Optimization: {config.name}')

            # 1. 목적 함수 값 히스토리
            trials = study.trials
            values = [t.value for t in trials if t.value is not None]
            axes[0, 0].plot(values)
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Objective Value')

            # 2. 파라미터 중요도
            try:
                importance = optuna.importance.get_param_importances(study)
                params, importances = zip(*importance.items())
                axes[0, 1].barh(params, importances)
                axes[0, 1].set_title('Parameter Importance')
            except:
                axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available',
                              ha='center', va='center', transform=axes[0, 1].transAxes)

            # 3. 수렴 플롯
            best_values = []
            best_so_far = float('-inf')
            for value in values:
                if value > best_so_far:
                    best_so_far = value
                best_values.append(best_so_far)

            axes[1, 0].plot(best_values)
            axes[1, 0].set_title('Best Value History')
            axes[1, 0].set_xlabel('Trial')
            axes[1, 0].set_ylabel('Best Objective Value')

            # 4. 파라미터 분포 (첫 번째 파라미터)
            if len(study.best_params) > 0:
                first_param = list(study.best_params.keys())[0]
                param_values = [t.params.get(first_param) for t in trials if first_param in t.params]
                if param_values:
                    axes[1, 1].hist(param_values, bins=20, alpha=0.7)
                    axes[1, 1].axvline(study.best_params[first_param], color='red', linestyle='--',
                                     label='Best Value')
                    axes[1, 1].set_title(f'Parameter Distribution: {first_param}')
                    axes[1, 1].legend()

            plt.tight_layout()

            # 저장
            plot_path = self.results_dir / f"optimization_{config.experiment_id}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # W&B 로깅
            if self.enable_wandb:
                wandb.log({"optimization_plot": wandb.Image(str(plot_path))})

        except Exception as e:
            logger.error(f"Failed to create optimization visualization: {e}")

    async def _visualize_ab_test_results(
        self,
        test_id: str,
        results: Dict[str, ABTestResult]
    ) -> None:
        """A/B 테스트 결과 시각화"""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'A/B Test Results: {test_id}')

            # 결과 데이터 준비
            variants = list(results.keys())
            conversion_rates = [results[v].conversion_rate for v in variants]
            sample_sizes = [results[v].sample_size for v in variants]
            lifts = [results[v].lift for v in variants]

            # 1. 전환율 비교
            bars = axes[0, 0].bar(variants, conversion_rates)
            axes[0, 0].set_title('Conversion Rates by Variant')
            axes[0, 0].set_ylabel('Conversion Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 유의성 표시
            for i, (variant, result) in enumerate(results.items()):
                if result.is_significant:
                    axes[0, 0].text(i, conversion_rates[i] + 0.01, '*',
                                   ha='center', va='bottom', fontsize=16, color='red')

            # 2. 샘플 크기
            axes[0, 1].bar(variants, sample_sizes)
            axes[0, 1].set_title('Sample Sizes')
            axes[0, 1].set_ylabel('Number of Users')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Lift 비교
            colors = ['green' if lift > 0 else 'red' for lift in lifts]
            axes[1, 0].bar(variants, lifts, color=colors, alpha=0.7)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].set_title('Lift vs Control (%)')
            axes[1, 0].set_ylabel('Lift (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 4. 신뢰구간
            for i, (variant, result) in enumerate(results.items()):
                ci_low, ci_high = result.confidence_interval
                axes[1, 1].errorbar(i, result.conversion_rate,
                                   yerr=[[result.conversion_rate - ci_low],
                                        [ci_high - result.conversion_rate]],
                                   fmt='o', capsize=5)
                axes[1, 1].text(i, ci_high + 0.01, f'p={result.p_value:.3f}',
                               ha='center', va='bottom', fontsize=8)

            axes[1, 1].set_title('Confidence Intervals (95%)')
            axes[1, 1].set_ylabel('Conversion Rate')
            axes[1, 1].set_xticks(range(len(variants)))
            axes[1, 1].set_xticklabels(variants, rotation=45)

            plt.tight_layout()

            # 저장
            plot_path = self.results_dir / f"ab_test_{test_id}_results.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # W&B 로깅
            if self.enable_wandb:
                wandb.log({f"ab_test_{test_id}_plot": wandb.Image(str(plot_path))})

        except Exception as e:
            logger.error(f"Failed to create A/B test visualization: {e}")

    async def generate_evaluation_report(
        self,
        experiment_id: Optional[str] = None,
        test_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """종합 평가 리포트 생성"""

        logger.info("Generating evaluation report")

        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {},
            'experiments': [],
            'ab_tests': [],
            'recommendations': []
        }

        # 실험 결과 요약
        if experiment_id:
            exp_results = [r for r in self.experiment_results if r.experiment_id == experiment_id]
        else:
            exp_results = self.experiment_results

        if exp_results:
            report['experiments'] = [asdict(r) for r in exp_results]
            report['summary']['total_experiments'] = len(exp_results)
            report['summary']['avg_execution_time'] = np.mean([r.execution_time for r in exp_results])

        # A/B 테스트 결과 요약
        if test_id:
            ab_results = self.ab_test_results.get(test_id, [])
        else:
            ab_results = []
            for results_list in self.ab_test_results.values():
                ab_results.extend(results_list)

        if ab_results:
            report['ab_tests'] = [asdict(r) for r in ab_results]
            significant_tests = [r for r in ab_results if r.is_significant]
            report['summary']['significant_ab_tests'] = len(significant_tests)
            report['summary']['avg_lift'] = np.mean([r.lift for r in ab_results])

        # 추천사항 생성
        report['recommendations'] = await self._generate_recommendations(exp_results, ab_results)

        # 리포트 저장
        report_path = self.results_dir / f"evaluation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {report_path}")
        return report

    async def _generate_recommendations(
        self,
        exp_results: List[ExperimentResult],
        ab_results: List[ABTestResult]
    ) -> List[str]:
        """추천사항 생성"""

        recommendations = []

        # 실험 기반 추천
        if exp_results:
            # 최고 성능 파라미터 추천
            best_result = max(exp_results, key=lambda r: max(r.metrics.values()))
            recommendations.append(
                f"Best performing parameters: {best_result.parameters} "
                f"achieved score: {max(best_result.metrics.values()):.4f}"
            )

            # 실행 시간 최적화 추천
            fast_results = sorted(exp_results, key=lambda r: r.execution_time)[:3]
            recommendations.append(
                f"Fastest configurations average {np.mean([r.execution_time for r in fast_results]):.2f}s execution time"
            )

        # A/B 테스트 기반 추천
        if ab_results:
            significant_results = [r for r in ab_results if r.is_significant]
            if significant_results:
                best_variant = max(significant_results, key=lambda r: r.lift)
                recommendations.append(
                    f"Deploy variant {best_variant.variant_name} - shows {best_variant.lift:.1f}% improvement"
                )

            # 샘플 크기 추천
            insufficient_power = [r for r in ab_results if r.statistical_power < 0.8]
            if insufficient_power:
                recommendations.append(
                    f"{len(insufficient_power)} tests need larger sample sizes for adequate statistical power"
                )

        if not recommendations:
            recommendations.append("Insufficient data for specific recommendations. Continue collecting data.")

        return recommendations

    def cleanup_old_results(self, days_to_keep: int = 30) -> None:
        """오래된 결과 파일 정리"""

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        for result_file in self.results_dir.glob("*"):
            if result_file.is_file():
                file_date = datetime.fromtimestamp(result_file.stat().st_mtime)
                if file_date < cutoff_date:
                    result_file.unlink()
                    logger.info(f"Cleaned up old result file: {result_file.name}")

    def __del__(self):
        """리소스 정리"""
        if self.enable_wandb:
            try:
                wandb.finish()
            except:
                pass