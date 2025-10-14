"""
A/B Testing Infrastructure
탐색(creative) vs 수렴(balanced) 트래픽 분할 및 성능 비교
"""

import random
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# A/B Test Variants
# =============================================================================

class Variant(str, Enum):
    """A/B test variant"""
    CONTROL = "control"      # Baseline (balanced mode)
    TREATMENT_A = "treatment_a"  # Creative mode
    TREATMENT_B = "treatment_b"  # Fast mode (optional)


# =============================================================================
# Traffic Split Configuration
# =============================================================================

@dataclass
class TrafficSplitConfig:
    """Traffic split configuration"""
    control_ratio: float = 0.5      # 50% control (balanced)
    treatment_a_ratio: float = 0.4  # 40% treatment A (creative)
    treatment_b_ratio: float = 0.1  # 10% treatment B (fast)

    def validate(self):
        """Validate ratios sum to 1.0"""
        total = self.control_ratio + self.treatment_a_ratio + self.treatment_b_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Traffic split ratios must sum to 1.0, got {total}")


# =============================================================================
# Experiment Metrics
# =============================================================================

@dataclass
class ExperimentMetrics:
    """Metrics for a single variant"""
    variant: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance metrics
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Quality metrics
    avg_user_rating: float = 0.0
    avg_reward: float = 0.0
    ifra_compliance_rate: float = 0.0

    # RL metrics (for creative/balanced)
    avg_entropy: float = 0.0
    avg_kl_divergence: float = 0.0
    avg_clip_fraction: float = 0.0

    # Business metrics
    conversion_rate: float = 0.0
    user_satisfaction_score: float = 0.0

    def update_latency(self, latency_samples: List[float]):
        """Update latency metrics"""
        if not latency_samples:
            return

        import numpy as np
        self.avg_latency = float(np.mean(latency_samples))
        self.p95_latency = float(np.percentile(latency_samples, 95))
        self.p99_latency = float(np.percentile(latency_samples, 99))

    def update_success_rate(self):
        """Update success rate"""
        if self.total_requests > 0:
            return self.successful_requests / self.total_requests
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# =============================================================================
# Experiment Result
# =============================================================================

@dataclass
class ExperimentResult:
    """Complete experiment result with statistical significance"""
    experiment_id: str
    start_time: str
    end_time: Optional[str]
    duration_hours: float

    control_metrics: ExperimentMetrics
    treatment_a_metrics: ExperimentMetrics
    treatment_b_metrics: Optional[ExperimentMetrics]

    # Statistical significance
    is_significant: bool = False
    p_value: Optional[float] = None
    confidence_level: float = 0.95

    # Winner
    winner: Optional[str] = None
    improvement_percentage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": self.duration_hours,
            "control_metrics": self.control_metrics.to_dict(),
            "treatment_a_metrics": self.treatment_a_metrics.to_dict(),
            "treatment_b_metrics": self.treatment_b_metrics.to_dict() if self.treatment_b_metrics else None,
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "winner": self.winner,
            "improvement_percentage": self.improvement_percentage
        }


# =============================================================================
# A/B Test Manager
# =============================================================================

class ABTestManager:
    """
    A/B testing manager for RL experiments

    Splits traffic between:
    - Control: Balanced mode (baseline)
    - Treatment A: Creative mode (more exploration)
    - Treatment B: Fast mode (optional)

    Tracks performance metrics and determines winner with statistical significance.
    """

    def __init__(
        self,
        experiment_id: str,
        traffic_split: TrafficSplitConfig,
        results_dir: str = "ab_test_results",
        min_sample_size: int = 100
    ):
        """
        Initialize A/B test manager

        Args:
            experiment_id: Unique experiment identifier
            traffic_split: Traffic split configuration
            results_dir: Directory to save results
            min_sample_size: Minimum sample size per variant
        """
        self.experiment_id = experiment_id
        self.traffic_split = traffic_split
        self.traffic_split.validate()
        self.min_sample_size = min_sample_size

        # Results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Experiment start time
        self.start_time = datetime.now().isoformat()

        # Metrics tracking
        self.control_metrics = ExperimentMetrics(variant=Variant.CONTROL)
        self.treatment_a_metrics = ExperimentMetrics(variant=Variant.TREATMENT_A)
        self.treatment_b_metrics = ExperimentMetrics(variant=Variant.TREATMENT_B) if traffic_split.treatment_b_ratio > 0 else None

        # Latency samples (for percentile calculation)
        self.control_latencies: List[float] = []
        self.treatment_a_latencies: List[float] = []
        self.treatment_b_latencies: List[float] = []

        logger.info(f"A/B test initialized: {experiment_id}, "
                   f"split={traffic_split.control_ratio}/{traffic_split.treatment_a_ratio}/{traffic_split.treatment_b_ratio}")

    def assign_variant(self, user_id: str) -> Variant:
        """
        Assign user to variant using consistent hashing

        Args:
            user_id: User identifier

        Returns:
            Assigned variant
        """
        # Consistent hashing for stable assignment
        hash_value = int(hashlib.md5(f"{self.experiment_id}:{user_id}".encode()).hexdigest(), 16)
        ratio = (hash_value % 10000) / 10000.0

        # Assign based on traffic split
        if ratio < self.traffic_split.control_ratio:
            return Variant.CONTROL
        elif ratio < self.traffic_split.control_ratio + self.traffic_split.treatment_a_ratio:
            return Variant.TREATMENT_A
        else:
            return Variant.TREATMENT_B

    def record_request(
        self,
        variant: Variant,
        success: bool,
        latency: float,
        user_rating: Optional[float] = None,
        reward: Optional[float] = None,
        ifra_compliant: bool = True,
        entropy: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        converted: bool = False
    ):
        """
        Record request metrics for a variant

        Args:
            variant: Variant that handled the request
            success: Whether request was successful
            latency: Request latency in seconds
            user_rating: User rating (0-5 scale)
            reward: RL reward value
            ifra_compliant: Whether formula was IFRA compliant
            entropy: Policy entropy
            kl_divergence: KL divergence from old policy
            clip_fraction: Fraction of clipped updates
            converted: Whether user completed purchase/action
        """
        # Get metrics object
        if variant == Variant.CONTROL:
            metrics = self.control_metrics
            latencies = self.control_latencies
        elif variant == Variant.TREATMENT_A:
            metrics = self.treatment_a_metrics
            latencies = self.treatment_a_latencies
        elif variant == Variant.TREATMENT_B:
            if not self.treatment_b_metrics:
                logger.warning(f"Treatment B not configured, ignoring request")
                return
            metrics = self.treatment_b_metrics
            latencies = self.treatment_b_latencies
        else:
            logger.error(f"Unknown variant: {variant}")
            return

        # Update counters
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update latency
        latencies.append(latency)
        if len(latencies) % 100 == 0:  # Update percentiles every 100 samples
            metrics.update_latency(latencies)

        # Update quality metrics
        if user_rating is not None:
            # Running average
            n = metrics.total_requests
            metrics.avg_user_rating = (metrics.avg_user_rating * (n - 1) + user_rating) / n

        if reward is not None:
            n = metrics.total_requests
            metrics.avg_reward = (metrics.avg_reward * (n - 1) + reward) / n

        if ifra_compliant:
            n = metrics.total_requests
            compliant_count = metrics.ifra_compliance_rate * (n - 1) + 1
            metrics.ifra_compliance_rate = compliant_count / n

        # Update RL metrics
        if entropy is not None:
            n = metrics.total_requests
            metrics.avg_entropy = (metrics.avg_entropy * (n - 1) + entropy) / n

        if kl_divergence is not None:
            n = metrics.total_requests
            metrics.avg_kl_divergence = (metrics.avg_kl_divergence * (n - 1) + kl_divergence) / n

        if clip_fraction is not None:
            n = metrics.total_requests
            metrics.avg_clip_fraction = (metrics.avg_clip_fraction * (n - 1) + clip_fraction) / n

        # Update business metrics
        if converted:
            n = metrics.total_requests
            conversion_count = metrics.conversion_rate * (n - 1) + 1
            metrics.conversion_rate = conversion_count / n

    def is_ready_for_analysis(self) -> bool:
        """Check if experiment has enough samples for analysis"""
        return (
            self.control_metrics.total_requests >= self.min_sample_size and
            self.treatment_a_metrics.total_requests >= self.min_sample_size
        )

    def calculate_statistical_significance(
        self,
        control_success_rate: float,
        treatment_success_rate: float,
        control_n: int,
        treatment_n: int,
        confidence_level: float = 0.95
    ) -> tuple[bool, float]:
        """
        Calculate statistical significance using two-proportion z-test

        Args:
            control_success_rate: Control group success rate
            treatment_success_rate: Treatment group success rate
            control_n: Control group sample size
            treatment_n: Treatment group sample size
            confidence_level: Confidence level (default 0.95 = 95%)

        Returns:
            (is_significant, p_value)
        """
        import numpy as np
        from scipy import stats

        # Pooled proportion
        p_pool = (control_success_rate * control_n + treatment_success_rate * treatment_n) / (control_n + treatment_n)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/treatment_n))

        # Z-score
        z_score = (treatment_success_rate - control_success_rate) / se

        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Is significant?
        alpha = 1 - confidence_level
        is_significant = p_value < alpha

        return is_significant, p_value

    def analyze_results(self) -> ExperimentResult:
        """
        Analyze experiment results and determine winner

        Returns:
            ExperimentResult with statistical analysis
        """
        if not self.is_ready_for_analysis():
            logger.warning(f"Insufficient samples for analysis: "
                          f"control={self.control_metrics.total_requests}, "
                          f"treatment_a={self.treatment_a_metrics.total_requests}")

        # Update final latency metrics
        self.control_metrics.update_latency(self.control_latencies)
        self.treatment_a_metrics.update_latency(self.treatment_a_latencies)
        if self.treatment_b_metrics:
            self.treatment_b_metrics.update_latency(self.treatment_b_latencies)

        # Calculate statistical significance (control vs treatment_a)
        control_success_rate = self.control_metrics.update_success_rate()
        treatment_a_success_rate = self.treatment_a_metrics.update_success_rate()

        is_significant, p_value = self.calculate_statistical_significance(
            control_success_rate=control_success_rate,
            treatment_success_rate=treatment_a_success_rate,
            control_n=self.control_metrics.total_requests,
            treatment_n=self.treatment_a_metrics.total_requests
        )

        # Determine winner
        winner = None
        improvement_percentage = None

        if is_significant:
            # Use avg_user_rating as primary metric
            if self.treatment_a_metrics.avg_user_rating > self.control_metrics.avg_user_rating:
                winner = Variant.TREATMENT_A
                improvement_percentage = ((self.treatment_a_metrics.avg_user_rating - self.control_metrics.avg_user_rating) /
                                        self.control_metrics.avg_user_rating * 100)
            else:
                winner = Variant.CONTROL
                improvement_percentage = ((self.control_metrics.avg_user_rating - self.treatment_a_metrics.avg_user_rating) /
                                        self.treatment_a_metrics.avg_user_rating * 100)

        # Create result
        end_time = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration_hours = (end_dt - start_dt).total_seconds() / 3600

        result = ExperimentResult(
            experiment_id=self.experiment_id,
            start_time=self.start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            control_metrics=self.control_metrics,
            treatment_a_metrics=self.treatment_a_metrics,
            treatment_b_metrics=self.treatment_b_metrics,
            is_significant=is_significant,
            p_value=p_value,
            winner=winner,
            improvement_percentage=improvement_percentage
        )

        # Save result
        self._save_result(result)

        return result

    def _save_result(self, result: ExperimentResult):
        """Save experiment result to file"""
        result_path = self.results_dir / f"{self.experiment_id}_result.json"

        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"A/B test result saved: {result_path}")
        except Exception as e:
            logger.error(f"Failed to save A/B test result: {e}")

    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*80)
        print(f"A/B Test Summary: {self.experiment_id}")
        print("="*80)

        print(f"\n[Control - Balanced Mode]")
        print(f"  Total Requests: {self.control_metrics.total_requests}")
        print(f"  Success Rate: {self.control_metrics.update_success_rate():.2%}")
        print(f"  Avg Latency: {self.control_metrics.avg_latency:.3f}s")
        print(f"  Avg User Rating: {self.control_metrics.avg_user_rating:.2f}/5.0")
        print(f"  Avg Reward: {self.control_metrics.avg_reward:.3f}")
        print(f"  IFRA Compliance: {self.control_metrics.ifra_compliance_rate:.2%}")

        print(f"\n[Treatment A - Creative Mode]")
        print(f"  Total Requests: {self.treatment_a_metrics.total_requests}")
        print(f"  Success Rate: {self.treatment_a_metrics.update_success_rate():.2%}")
        print(f"  Avg Latency: {self.treatment_a_metrics.avg_latency:.3f}s")
        print(f"  Avg User Rating: {self.treatment_a_metrics.avg_user_rating:.2f}/5.0")
        print(f"  Avg Reward: {self.treatment_a_metrics.avg_reward:.3f}")
        print(f"  IFRA Compliance: {self.treatment_a_metrics.ifra_compliance_rate:.2%}")
        print(f"  Avg Entropy: {self.treatment_a_metrics.avg_entropy:.4f}")
        print(f"  Avg KL Divergence: {self.treatment_a_metrics.avg_kl_divergence:.4f}")

        if self.is_ready_for_analysis():
            result = self.analyze_results()
            print(f"\n[Statistical Significance]")
            print(f"  Is Significant: {result.is_significant}")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Winner: {result.winner}")
            if result.improvement_percentage:
                print(f"  Improvement: {result.improvement_percentage:.1f}%")
        else:
            print(f"\n[Status]")
            print(f"  Not enough samples for analysis")
            print(f"  Control: {self.control_metrics.total_requests}/{self.min_sample_size}")
            print(f"  Treatment A: {self.treatment_a_metrics.total_requests}/{self.min_sample_size}")

        print("="*80 + "\n")


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'ABTestManager',
    'Variant',
    'TrafficSplitConfig',
    'ExperimentMetrics',
    'ExperimentResult'
]
