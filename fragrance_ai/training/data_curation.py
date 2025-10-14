"""
Data Curation System
피드백 샘플 품질 기준: 이상치 제거, 스팸 차단, 품질 필터링
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Filters
# =============================================================================

@dataclass
class QualityThresholds:
    """Data quality thresholds"""
    # Reward thresholds
    min_reward: float = -10.0
    max_reward: float = 100.0

    # Rating thresholds
    min_rating: float = 0.0
    max_rating: float = 5.0

    # Statistical outlier detection
    z_score_threshold: float = 3.0  # Remove samples > 3 standard deviations

    # Duplicate detection
    max_duplicate_submissions: int = 5  # Max submissions from same user in window
    duplicate_window_minutes: int = 60  # Time window for duplicate detection

    # Response time filter
    min_response_time_seconds: float = 2.0  # Minimum time to give meaningful feedback
    max_response_time_seconds: float = 600.0  # 10 minutes max

    # IFRA compliance requirement
    require_ifra_compliance: bool = True


# =============================================================================
# Sample Metadata
# =============================================================================

@dataclass
class FeedbackSample:
    """Feedback sample with metadata"""
    sample_id: str
    user_id: str
    timestamp: str
    reward: float
    user_rating: Optional[float]
    response_time_seconds: float
    ifra_compliant: bool

    # Additional metadata
    experiment_id: Optional[str] = None
    mode: Optional[str] = None  # fast/balanced/creative
    formula_complexity: Optional[float] = None

    def is_valid_range(self, thresholds: QualityThresholds) -> bool:
        """Check if sample is within valid ranges"""
        # Check reward range
        if not (thresholds.min_reward <= self.reward <= thresholds.max_reward):
            return False

        # Check rating range
        if self.user_rating is not None:
            if not (thresholds.min_rating <= self.user_rating <= thresholds.max_rating):
                return False

        # Check response time
        if not (thresholds.min_response_time_seconds <= self.response_time_seconds <= thresholds.max_response_time_seconds):
            return False

        # Check IFRA compliance
        if thresholds.require_ifra_compliance and not self.ifra_compliant:
            return False

        return True


# =============================================================================
# Data Curator
# =============================================================================

class DataCurator:
    """
    Data curation system for RL feedback samples

    Filters out:
    - Outliers (statistical anomalies)
    - Spam (duplicate submissions, bot activity)
    - Low-quality samples (too fast/slow responses)
    - IFRA non-compliant samples
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize data curator

        Args:
            thresholds: Quality thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or QualityThresholds()

        # Tracking
        self.total_samples = 0
        self.accepted_samples = 0
        self.rejected_samples = 0

        # Rejection reasons
        self.rejection_reasons: Dict[str, int] = defaultdict(int)

        # User submission tracking (for spam detection)
        self.user_submissions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Statistical tracking (for outlier detection)
        self.reward_history: deque = deque(maxlen=1000)
        self.rating_history: deque = deque(maxlen=1000)

        logger.info(f"DataCurator initialized with thresholds: {self.thresholds}")

    def filter_sample(self, sample: FeedbackSample) -> Tuple[bool, Optional[str]]:
        """
        Filter feedback sample

        Args:
            sample: Feedback sample to filter

        Returns:
            (is_accepted, rejection_reason)
        """
        self.total_samples += 1

        # 1. Range validation
        if not sample.is_valid_range(self.thresholds):
            reason = self._get_range_violation_reason(sample)
            self._reject_sample(reason)
            return False, reason

        # 2. Spam detection
        is_spam, spam_reason = self._is_spam(sample)
        if is_spam:
            self._reject_sample(f"spam_{spam_reason}")
            return False, f"Spam detected: {spam_reason}"

        # 3. Statistical outlier detection
        is_outlier, outlier_reason = self._is_statistical_outlier(sample)
        if is_outlier:
            self._reject_sample(f"outlier_{outlier_reason}")
            return False, f"Statistical outlier: {outlier_reason}"

        # Accept sample
        self.accepted_samples += 1

        # Update statistics
        self.reward_history.append(sample.reward)
        if sample.user_rating is not None:
            self.rating_history.append(sample.user_rating)

        # Track user submission
        timestamp = datetime.fromisoformat(sample.timestamp)
        self.user_submissions[sample.user_id].append(timestamp)

        return True, None

    def _get_range_violation_reason(self, sample: FeedbackSample) -> str:
        """Get specific reason for range violation"""
        if not (self.thresholds.min_reward <= sample.reward <= self.thresholds.max_reward):
            return f"reward_out_of_range"

        if sample.user_rating is not None:
            if not (self.thresholds.min_rating <= sample.user_rating <= self.thresholds.max_rating):
                return f"rating_out_of_range"

        if not (self.thresholds.min_response_time_seconds <= sample.response_time_seconds <= self.thresholds.max_response_time_seconds):
            if sample.response_time_seconds < self.thresholds.min_response_time_seconds:
                return "response_too_fast"
            else:
                return "response_too_slow"

        if self.thresholds.require_ifra_compliance and not sample.ifra_compliant:
            return "ifra_non_compliant"

        return "unknown_range_violation"

    def _is_spam(self, sample: FeedbackSample) -> Tuple[bool, Optional[str]]:
        """
        Detect spam submissions

        Args:
            sample: Feedback sample

        Returns:
            (is_spam, reason)
        """
        # Get user's recent submissions
        user_history = self.user_submissions[sample.user_id]

        if not user_history:
            return False, None

        # Count submissions in time window
        current_time = datetime.fromisoformat(sample.timestamp)
        window_start = current_time - timedelta(minutes=self.thresholds.duplicate_window_minutes)

        recent_submissions = sum(
            1 for timestamp in user_history
            if timestamp > window_start
        )

        # Check if exceeds threshold
        if recent_submissions >= self.thresholds.max_duplicate_submissions:
            return True, f"too_many_submissions_{recent_submissions}_in_{self.thresholds.duplicate_window_minutes}min"

        return False, None

    def _is_statistical_outlier(self, sample: FeedbackSample) -> Tuple[bool, Optional[str]]:
        """
        Detect statistical outliers using z-score

        Args:
            sample: Feedback sample

        Returns:
            (is_outlier, reason)
        """
        # Need sufficient history for outlier detection
        if len(self.reward_history) < 30:
            return False, None

        # Calculate z-score for reward
        reward_array = np.array(self.reward_history)
        reward_mean = np.mean(reward_array)
        reward_std = np.std(reward_array)

        if reward_std > 0:
            reward_z_score = abs(sample.reward - reward_mean) / reward_std

            if reward_z_score > self.thresholds.z_score_threshold:
                return True, f"reward_z_score_{reward_z_score:.2f}"

        # Calculate z-score for rating (if available)
        if sample.user_rating is not None and len(self.rating_history) >= 30:
            rating_array = np.array(self.rating_history)
            rating_mean = np.mean(rating_array)
            rating_std = np.std(rating_array)

            if rating_std > 0:
                rating_z_score = abs(sample.user_rating - rating_mean) / rating_std

                if rating_z_score > self.thresholds.z_score_threshold:
                    return True, f"rating_z_score_{rating_z_score:.2f}"

        return False, None

    def _reject_sample(self, reason: str):
        """Track rejected sample"""
        self.rejected_samples += 1
        self.rejection_reasons[reason] += 1
        logger.debug(f"Sample rejected: {reason}")

    def get_statistics(self) -> Dict[str, any]:
        """Get curation statistics"""
        acceptance_rate = self.accepted_samples / self.total_samples if self.total_samples > 0 else 0.0

        return {
            "total_samples": self.total_samples,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "acceptance_rate": acceptance_rate,
            "rejection_reasons": dict(self.rejection_reasons),
            "reward_history_size": len(self.reward_history),
            "rating_history_size": len(self.rating_history),
            "tracked_users": len(self.user_submissions)
        }

    def reset_statistics(self):
        """Reset statistics (keep history for outlier detection)"""
        self.total_samples = 0
        self.accepted_samples = 0
        self.rejected_samples = 0
        self.rejection_reasons.clear()


# =============================================================================
# Batch Filtering
# =============================================================================

def filter_feedback_batch(
    samples: List[FeedbackSample],
    curator: DataCurator
) -> Tuple[List[FeedbackSample], List[Tuple[FeedbackSample, str]]]:
    """
    Filter a batch of feedback samples

    Args:
        samples: List of feedback samples
        curator: Data curator instance

    Returns:
        (accepted_samples, rejected_samples_with_reasons)
    """
    accepted = []
    rejected = []

    for sample in samples:
        is_accepted, reason = curator.filter_sample(sample)

        if is_accepted:
            accepted.append(sample)
        else:
            rejected.append((sample, reason))

    return accepted, rejected


# =============================================================================
# Quality Report
# =============================================================================

def generate_quality_report(curator: DataCurator) -> str:
    """Generate quality report"""
    stats = curator.get_statistics()

    report = []
    report.append("="*80)
    report.append("DATA CURATION QUALITY REPORT")
    report.append("="*80)
    report.append("")

    report.append(f"Total Samples: {stats['total_samples']}")
    report.append(f"Accepted: {stats['accepted_samples']} ({stats['acceptance_rate']:.1%})")
    report.append(f"Rejected: {stats['rejected_samples']} ({1-stats['acceptance_rate']:.1%})")
    report.append("")

    report.append("Rejection Reasons:")
    for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: -x[1]):
        percentage = count / stats['rejected_samples'] * 100 if stats['rejected_samples'] > 0 else 0
        report.append(f"  - {reason}: {count} ({percentage:.1f}%)")
    report.append("")

    report.append(f"Reward History Size: {stats['reward_history_size']}")
    report.append(f"Rating History Size: {stats['rating_history_size']}")
    report.append(f"Tracked Users: {stats['tracked_users']}")
    report.append("")

    # Calculate reward statistics
    if curator.reward_history:
        reward_array = np.array(curator.reward_history)
        report.append("Reward Statistics:")
        report.append(f"  Mean: {np.mean(reward_array):.3f}")
        report.append(f"  Std: {np.std(reward_array):.3f}")
        report.append(f"  Min: {np.min(reward_array):.3f}")
        report.append(f"  Max: {np.max(reward_array):.3f}")
        report.append(f"  Median: {np.median(reward_array):.3f}")
        report.append("")

    # Calculate rating statistics
    if curator.rating_history:
        rating_array = np.array(curator.rating_history)
        report.append("Rating Statistics:")
        report.append(f"  Mean: {np.mean(rating_array):.2f}")
        report.append(f"  Std: {np.std(rating_array):.2f}")
        report.append(f"  Min: {np.min(rating_array):.2f}")
        report.append(f"  Max: {np.max(rating_array):.2f}")
        report.append(f"  Median: {np.median(rating_array):.2f}")
        report.append("")

    report.append("="*80)

    return "\n".join(report)


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'DataCurator',
    'FeedbackSample',
    'QualityThresholds',
    'filter_feedback_batch',
    'generate_quality_report'
]
