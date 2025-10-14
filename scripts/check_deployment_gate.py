"""
Go/No-Go Deployment Gate Checker
신호등 시스템: 배포 가능 여부 자동 판단

Go (녹색): 스키마 실패 0%, API 에러율 < 0.5%, p95 기준 충족, RL reward 상승 추세
No-Go (적색): 스키마 실패>0%, RL loss 폭주, creative p95>4.5s 지속
"""

import requests
import json
import sys
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Gate Status Enum
# =============================================================================

class GateStatus(str, Enum):
    """배포 게이트 상태"""
    GO = "GO"           # 녹색 - 배포 가능
    NO_GO = "NO_GO"     # 적색 - 배포 불가
    WARNING = "WARNING" # 황색 - 주의 필요


# =============================================================================
# Configuration
# =============================================================================

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
API_URL = "http://localhost:8001"

# Go/No-Go Thresholds
THRESHOLDS = {
    # API Performance
    "api_error_rate_max": 0.005,        # 0.5%
    "fast_p95_max": 2.5,                # seconds
    "balanced_p95_max": 3.2,            # seconds
    "creative_p95_max": 4.5,            # seconds

    # Schema Validation
    "schema_failure_rate_max": 0.0,     # 0% (zero tolerance)

    # RL Performance
    "rl_reward_min_trend": 0.0,         # Must be non-negative (stable or increasing)
    "rl_loss_max": 2.0,                 # Maximum acceptable loss
    "rl_loss_increase_threshold": 3.0,  # Loss runaway detection (3x increase)

    # Cache Performance
    "cache_hit_rate_min": 0.60,         # 60% for fast/balanced

    # System Health
    "cpu_usage_max": 0.85,              # 85%
    "memory_usage_max": 0.85,           # 85%
    "vram_headroom_min": 0.20,          # 20%
}

# Time windows for queries
TIME_WINDOWS = {
    "short": "5m",   # 5 minutes
    "medium": "30m", # 30 minutes
    "long": "2h"     # 2 hours
}


# =============================================================================
# Prometheus Query Functions
# =============================================================================

def query_prometheus(query: str, time_window: Optional[str] = None) -> Optional[float]:
    """
    Query Prometheus and return single value

    Args:
        query: PromQL query
        time_window: Time window (e.g., "5m", "1h")

    Returns:
        Float value or None if query fails
    """
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query"

        if time_window:
            # Add time window to query if not already present
            if '[' not in query:
                query = f"{query}[{time_window}]"

        params = {"query": query}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            result = data["data"]["result"][0]
            value = float(result["value"][1])
            return value

        return None

    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        logger.debug(f"Query: {query}")
        return None


def query_prometheus_range(query: str, duration: str = "2h") -> List[Tuple[float, float]]:
    """
    Query Prometheus range and return time series

    Args:
        query: PromQL query
        duration: Time range (e.g., "2h")

    Returns:
        List of (timestamp, value) tuples
    """
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query_range"

        end_time = time.time()
        start_time = end_time - _parse_duration(duration)

        params = {
            "query": query,
            "start": start_time,
            "end": end_time,
            "step": "60s"  # 1 minute resolution
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            result = data["data"]["result"][0]
            values = [(float(v[0]), float(v[1])) for v in result["values"]]
            return values

        return []

    except Exception as e:
        logger.error(f"Prometheus range query failed: {e}")
        return []


def _parse_duration(duration: str) -> int:
    """Parse duration string to seconds"""
    multipliers = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400
    }

    unit = duration[-1]
    value = int(duration[:-1])

    return value * multipliers.get(unit, 60)


# =============================================================================
# Gate Checks
# =============================================================================

class GateCheck:
    """Individual gate check result"""

    def __init__(self, name: str, status: GateStatus, value: Optional[float],
                 threshold: Optional[float], message: str):
        self.name = name
        self.status = status
        self.value = value
        self.threshold = threshold
        self.message = message
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


def check_api_error_rate() -> GateCheck:
    """Check API error rate (< 0.5%)"""
    query = """
    (
        sum(rate(http_requests_total{status=~"5.."}[5m]))
        /
        sum(rate(http_requests_total[5m]))
    )
    """

    error_rate = query_prometheus(query)
    threshold = THRESHOLDS["api_error_rate_max"]

    if error_rate is None:
        return GateCheck(
            name="API Error Rate",
            status=GateStatus.WARNING,
            value=None,
            threshold=threshold,
            message="Unable to query error rate"
        )

    if error_rate <= threshold:
        return GateCheck(
            name="API Error Rate",
            status=GateStatus.GO,
            value=error_rate,
            threshold=threshold,
            message=f"Error rate: {error_rate*100:.3f}% (< {threshold*100}%)"
        )
    else:
        return GateCheck(
            name="API Error Rate",
            status=GateStatus.NO_GO,
            value=error_rate,
            threshold=threshold,
            message=f"Error rate too high: {error_rate*100:.3f}% (threshold: {threshold*100}%)"
        )


def check_p95_latency() -> List[GateCheck]:
    """Check p95 latency for all modes"""
    checks = []

    modes = {
        "fast": THRESHOLDS["fast_p95_max"],
        "balanced": THRESHOLDS["balanced_p95_max"],
        "creative": THRESHOLDS["creative_p95_max"]
    }

    for mode, threshold in modes.items():
        query = f'histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{{mode="{mode}"}}[5m]))'
        p95 = query_prometheus(query)

        if p95 is None:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} p95 Latency",
                status=GateStatus.WARNING,
                value=None,
                threshold=threshold,
                message=f"Unable to query {mode} p95"
            ))
            continue

        if p95 <= threshold:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} p95 Latency",
                status=GateStatus.GO,
                value=p95,
                threshold=threshold,
                message=f"p95: {p95:.2f}s (< {threshold}s)"
            ))
        else:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} p95 Latency",
                status=GateStatus.NO_GO,
                value=p95,
                threshold=threshold,
                message=f"p95 too high: {p95:.2f}s (threshold: {threshold}s)"
            ))

    return checks


def check_schema_failure_rate() -> GateCheck:
    """Check schema failure rate (must be 0%)"""
    query = """
    (
        sum(rate(llm_schema_fix_count_total[30m]))
        /
        sum(rate(llm_brief_total[30m]))
    )
    """

    failure_rate = query_prometheus(query)
    threshold = THRESHOLDS["schema_failure_rate_max"]

    if failure_rate is None:
        return GateCheck(
            name="Schema Failure Rate",
            status=GateStatus.WARNING,
            value=None,
            threshold=threshold,
            message="Unable to query schema failure rate"
        )

    if failure_rate <= threshold:
        return GateCheck(
            name="Schema Failure Rate",
            status=GateStatus.GO,
            value=failure_rate,
            threshold=threshold,
            message=f"Schema failures: {failure_rate*100:.3f}% (= 0%)"
        )
    else:
        return GateCheck(
            name="Schema Failure Rate",
            status=GateStatus.NO_GO,
            value=failure_rate,
            threshold=threshold,
            message=f"Schema failures detected: {failure_rate*100:.3f}% (must be 0%)"
        )


def check_rl_reward_trend() -> GateCheck:
    """Check RL reward trend (must be stable or increasing)"""
    query = 'rl_reward_ma{window="100"}'

    # Get time series for last 2 hours
    values = query_prometheus_range(query, duration="2h")

    if not values or len(values) < 10:
        return GateCheck(
            name="RL Reward Trend",
            status=GateStatus.WARNING,
            value=None,
            threshold=THRESHOLDS["rl_reward_min_trend"],
            message="Insufficient data for trend analysis"
        )

    # Calculate trend (simple linear regression slope)
    timestamps = [v[0] for v in values]
    rewards = [v[1] for v in values]

    # Normalize timestamps
    t_min = min(timestamps)
    t_normalized = [(t - t_min) for t in timestamps]

    # Calculate slope
    n = len(t_normalized)
    sum_t = sum(t_normalized)
    sum_r = sum(rewards)
    sum_tr = sum(t * r for t, r in zip(t_normalized, rewards))
    sum_tt = sum(t * t for t in t_normalized)

    slope = (n * sum_tr - sum_t * sum_r) / (n * sum_tt - sum_t * sum_t)

    current_reward = rewards[-1]

    if slope >= THRESHOLDS["rl_reward_min_trend"]:
        trend = "increasing" if slope > 0.01 else "stable"
        return GateCheck(
            name="RL Reward Trend",
            status=GateStatus.GO,
            value=current_reward,
            threshold=THRESHOLDS["rl_reward_min_trend"],
            message=f"Reward trend: {trend} (current: {current_reward:.2f})"
        )
    else:
        return GateCheck(
            name="RL Reward Trend",
            status=GateStatus.NO_GO,
            value=current_reward,
            threshold=THRESHOLDS["rl_reward_min_trend"],
            message=f"Reward declining (slope: {slope:.4f})"
        )


def check_rl_loss_runaway() -> GateCheck:
    """Check for RL loss runaway (폭주)"""
    query_current = 'rl_loss{loss_type="total_loss"}'
    query_baseline = f'avg_over_time(rl_loss{{loss_type="total_loss"}}[{TIME_WINDOWS["long"]}])'

    current_loss = query_prometheus(query_current)
    baseline_loss = query_prometheus(query_baseline)

    threshold = THRESHOLDS["rl_loss_max"]
    runaway_threshold = THRESHOLDS["rl_loss_increase_threshold"]

    if current_loss is None or baseline_loss is None:
        return GateCheck(
            name="RL Loss Runaway",
            status=GateStatus.WARNING,
            value=None,
            threshold=threshold,
            message="Unable to query RL loss"
        )

    # Check absolute loss
    if current_loss > threshold:
        return GateCheck(
            name="RL Loss Runaway",
            status=GateStatus.NO_GO,
            value=current_loss,
            threshold=threshold,
            message=f"Loss too high: {current_loss:.4f} (threshold: {threshold})"
        )

    # Check for runaway (3x increase from baseline)
    if baseline_loss > 0 and (current_loss / baseline_loss) > runaway_threshold:
        return GateCheck(
            name="RL Loss Runaway",
            status=GateStatus.NO_GO,
            value=current_loss,
            threshold=baseline_loss * runaway_threshold,
            message=f"Loss runaway detected: {current_loss:.4f} (baseline: {baseline_loss:.4f}, {current_loss/baseline_loss:.1f}x increase)"
        )

    return GateCheck(
        name="RL Loss Runaway",
        status=GateStatus.GO,
        value=current_loss,
        threshold=threshold,
        message=f"Loss stable: {current_loss:.4f}"
    )


def check_cache_hit_rate() -> List[GateCheck]:
    """Check cache hit rate (≥60% for fast/balanced)"""
    checks = []
    threshold = THRESHOLDS["cache_hit_rate_min"]

    for mode in ["fast", "balanced"]:
        query = f'cache_hit_rate{{mode="{mode}"}}'
        hit_rate = query_prometheus(query)

        if hit_rate is None:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} Cache Hit Rate",
                status=GateStatus.WARNING,
                value=None,
                threshold=threshold,
                message=f"Unable to query {mode} cache hit rate"
            ))
            continue

        if hit_rate >= threshold:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} Cache Hit Rate",
                status=GateStatus.GO,
                value=hit_rate,
                threshold=threshold,
                message=f"Hit rate: {hit_rate*100:.1f}% (≥ {threshold*100}%)"
            ))
        else:
            checks.append(GateCheck(
                name=f"{mode.capitalize()} Cache Hit Rate",
                status=GateStatus.WARNING,  # Warning, not No-Go
                value=hit_rate,
                threshold=threshold,
                message=f"Hit rate below target: {hit_rate*100:.1f}% (target: {threshold*100}%)"
            ))

    return checks


def check_system_health() -> List[GateCheck]:
    """Check system health (CPU, memory, VRAM)"""
    checks = []

    # CPU usage
    query_cpu = 'avg(rate(container_cpu_usage_seconds_total[5m]))'
    cpu_usage = query_prometheus(query_cpu)

    if cpu_usage is not None:
        threshold = THRESHOLDS["cpu_usage_max"]
        if cpu_usage <= threshold:
            checks.append(GateCheck(
                name="CPU Usage",
                status=GateStatus.GO,
                value=cpu_usage,
                threshold=threshold,
                message=f"CPU: {cpu_usage*100:.1f}% (< {threshold*100}%)"
            ))
        else:
            checks.append(GateCheck(
                name="CPU Usage",
                status=GateStatus.WARNING,
                value=cpu_usage,
                threshold=threshold,
                message=f"CPU high: {cpu_usage*100:.1f}% (threshold: {threshold*100}%)"
            ))

    # Memory usage
    query_mem = 'avg(container_memory_usage_bytes / container_memory_max_bytes)'
    mem_usage = query_prometheus(query_mem)

    if mem_usage is not None:
        threshold = THRESHOLDS["memory_usage_max"]
        if mem_usage <= threshold:
            checks.append(GateCheck(
                name="Memory Usage",
                status=GateStatus.GO,
                value=mem_usage,
                threshold=threshold,
                message=f"Memory: {mem_usage*100:.1f}% (< {threshold*100}%)"
            ))
        else:
            checks.append(GateCheck(
                name="Memory Usage",
                status=GateStatus.WARNING,
                value=mem_usage,
                threshold=threshold,
                message=f"Memory high: {mem_usage*100:.1f}% (threshold: {threshold*100}%)"
            ))

    # VRAM headroom (if GPU metrics available)
    query_vram = '(nvidia_gpu_memory_free_bytes / nvidia_gpu_memory_total_bytes)'
    vram_free = query_prometheus(query_vram)

    if vram_free is not None:
        threshold = THRESHOLDS["vram_headroom_min"]
        if vram_free >= threshold:
            checks.append(GateCheck(
                name="VRAM Headroom",
                status=GateStatus.GO,
                value=vram_free,
                threshold=threshold,
                message=f"VRAM free: {vram_free*100:.1f}% (≥ {threshold*100}%)"
            ))
        else:
            checks.append(GateCheck(
                name="VRAM Headroom",
                status=GateStatus.NO_GO,
                value=vram_free,
                threshold=threshold,
                message=f"VRAM low: {vram_free*100:.1f}% (threshold: {threshold*100}%)"
            ))

    return checks


# =============================================================================
# Gate Decision
# =============================================================================

def run_all_checks() -> Tuple[GateStatus, List[GateCheck]]:
    """
    Run all gate checks and return overall status

    Returns:
        (overall_status, list_of_checks)
    """
    logger.info("Running deployment gate checks...")

    all_checks = []

    # Critical checks (any NO_GO = deployment blocked)
    all_checks.append(check_api_error_rate())
    all_checks.append(check_schema_failure_rate())
    all_checks.extend(check_p95_latency())
    all_checks.append(check_rl_reward_trend())
    all_checks.append(check_rl_loss_runaway())

    # Non-blocking checks (WARNING only)
    all_checks.extend(check_cache_hit_rate())
    all_checks.extend(check_system_health())

    # Determine overall status
    has_no_go = any(c.status == GateStatus.NO_GO for c in all_checks)
    has_warning = any(c.status == GateStatus.WARNING for c in all_checks)

    if has_no_go:
        overall_status = GateStatus.NO_GO
    elif has_warning:
        overall_status = GateStatus.WARNING
    else:
        overall_status = GateStatus.GO

    return overall_status, all_checks


# =============================================================================
# Output and Reporting
# =============================================================================

def print_gate_report(status: GateStatus, checks: List[GateCheck]):
    """Print formatted gate report with traffic light"""

    # Traffic light symbols
    symbols = {
        GateStatus.GO: "🟢",
        GateStatus.WARNING: "🟡",
        GateStatus.NO_GO: "🔴"
    }

    print("=" * 80)
    print("DEPLOYMENT GATE CHECK")
    print("=" * 80)
    print()

    # Overall status
    symbol = symbols.get(status, "⚪")
    print(f"Overall Status: {symbol} {status.value}")
    print()

    if status == GateStatus.GO:
        print("✅ All checks passed - Deployment is GO")
    elif status == GateStatus.WARNING:
        print("⚠️  Some warnings detected - Review before deployment")
    else:
        print("❌ Deployment is NO-GO - Critical issues detected")

    print()
    print("=" * 80)
    print("CHECK DETAILS")
    print("=" * 80)
    print()

    # Group checks by status
    for check_status in [GateStatus.NO_GO, GateStatus.WARNING, GateStatus.GO]:
        status_checks = [c for c in checks if c.status == check_status]

        if not status_checks:
            continue

        symbol = symbols.get(check_status, "⚪")
        print(f"{symbol} {check_status.value} ({len(status_checks)} checks)")
        print("-" * 80)

        for check in status_checks:
            print(f"  • {check.name}")
            print(f"    {check.message}")
            if check.value is not None and check.threshold is not None:
                print(f"    Value: {check.value:.4f} | Threshold: {check.threshold:.4f}")
            print()

    print("=" * 80)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 80)


def save_gate_report(status: GateStatus, checks: List[GateCheck], output_file: str = "gate_report.json"):
    """Save gate report to JSON file"""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": status.value,
        "checks": [c.to_dict() for c in checks],
        "summary": {
            "total": len(checks),
            "go": len([c for c in checks if c.status == GateStatus.GO]),
            "warning": len([c for c in checks if c.status == GateStatus.WARNING]),
            "no_go": len([c for c in checks if c.status == GateStatus.NO_GO])
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Gate report saved to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point"""
    import argparse

    # Declare global before using
    global PROMETHEUS_URL

    parser = argparse.ArgumentParser(
        description="Go/No-Go deployment gate checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run gate check
  python scripts/check_deployment_gate.py

  # Save report to file
  python scripts/check_deployment_gate.py --output gate_report.json

  # Exit with error code on NO_GO
  python scripts/check_deployment_gate.py --strict

Exit codes:
  0 - GO (all checks passed)
  1 - WARNING (some warnings)
  2 - NO_GO (deployment blocked)
        """
    )

    parser.add_argument(
        "--output", "-o",
        help="Save report to JSON file"
    )

    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Exit with error code on NO_GO (for CI/CD)"
    )

    parser.add_argument(
        "--prometheus-url",
        default=PROMETHEUS_URL,
        help=f"Prometheus URL (default: {PROMETHEUS_URL})"
    )

    args = parser.parse_args()

    # Override Prometheus URL if provided
    PROMETHEUS_URL = args.prometheus_url

    # Run checks
    try:
        status, checks = run_all_checks()
    except Exception as e:
        logger.error(f"Gate check failed with exception: {e}")
        print()
        print("🔴 GATE CHECK FAILED")
        print(f"Error: {e}")
        sys.exit(2)

    # Print report
    print_gate_report(status, checks)

    # Save report if requested
    if args.output:
        save_gate_report(status, checks, args.output)

    # Exit with appropriate code
    if args.strict:
        exit_codes = {
            GateStatus.GO: 0,
            GateStatus.WARNING: 1,
            GateStatus.NO_GO: 2
        }
        sys.exit(exit_codes[status])
    else:
        # Always exit 0 in non-strict mode
        sys.exit(0)


if __name__ == "__main__":
    main()
