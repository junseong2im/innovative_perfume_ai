"""
Comprehensive Fallback Strategy and Error Recovery Tests
Tests various fallback patterns and recovery mechanisms
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional

from fragrance_ai.orchestrator.artisan_orchestrator import ArtisanOrchestrator
from fragrance_ai.orchestrator.customer_service_orchestrator import CustomerServiceOrchestrator


class TestFallbackStrategies:
    """Test various fallback strategies when primary operations fail"""

    @pytest.mark.asyncio
    async def test_multi_tier_fallback_system(self):
        """Test multi-tier fallback from premium to basic services"""

        class ServiceTier:
            def __init__(self, name: str, quality: float, cost: float):
                self.name = name
                self.quality = quality
                self.cost = cost
                self.attempts = 0

        tiers = [
            ServiceTier("premium_llm", quality=1.0, cost=1.0),
            ServiceTier("standard_llm", quality=0.7, cost=0.5),
            ServiceTier("basic_llm", quality=0.4, cost=0.1),
            ServiceTier("cached_response", quality=0.2, cost=0.0)
        ]

        orchestrator = ArtisanOrchestrator()
        used_tier = None

        async def tiered_service(tier_index=0):
            nonlocal used_tier
            if tier_index >= len(tiers):
                return {"response": "All services failed", "quality": 0}

            tier = tiers[tier_index]
            tier.attempts += 1

            # Premium and standard fail
            if tier_index < 2:
                raise Exception(f"{tier.name} unavailable")

            used_tier = tier
            return {
                "response": f"Response from {tier.name}",
                "quality": tier.quality,
                "cost": tier.cost
            }

        # Setup fallback chain
        async def smart_fallback(*args, **kwargs):
            for i, tier in enumerate(tiers):
                try:
                    return await tiered_service(i)
                except Exception as e:
                    if i == len(tiers) - 1:
                        return {"response": "Using offline cache", "quality": 0.1}
                    continue

        orchestrator.tools = {"smart_service": smart_fallback}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["smart_service"]
        })

        result = await orchestrator.orchestrate("Get response")

        # Verify fallback chain worked
        assert tiers[0].attempts == 1  # Premium tried
        assert tiers[1].attempts == 1  # Standard tried
        assert tiers[2].attempts == 1  # Basic succeeded
        assert used_tier.name == "basic_llm"

    @pytest.mark.asyncio
    async def test_intelligent_cache_fallback(self):
        """Test fallback to cached responses when live services fail"""

        orchestrator = ArtisanOrchestrator()

        # Simulate cache
        cache = {
            "floral perfume": {
                "response": "Cached: Rose and jasmine blend",
                "timestamp": datetime.utcnow() - timedelta(hours=1),
                "relevance": 0.95
            },
            "woody perfume": {
                "response": "Cached: Sandalwood and cedar",
                "timestamp": datetime.utcnow() - timedelta(days=1),
                "relevance": 0.85
            },
            "citrus perfume": {
                "response": "Cached: Lemon and bergamot",
                "timestamp": datetime.utcnow() - timedelta(days=7),
                "relevance": 0.70
            }
        }

        async def cache_aware_tool(query, *args, **kwargs):
            # Try live service first
            if random.random() > 0.7:  # 30% success rate
                return {"response": f"Live: {query}", "source": "live"}

            # Fallback to cache
            best_match = None
            best_score = 0

            for cached_query, cached_data in cache.items():
                # Simple similarity check
                similarity = len(set(query.lower().split()) & set(cached_query.lower().split()))
                age_penalty = (datetime.utcnow() - cached_data["timestamp"]).days * 0.1
                score = similarity - age_penalty

                if score > best_score:
                    best_score = score
                    best_match = cached_data

            if best_match:
                return {
                    "response": best_match["response"],
                    "source": "cache",
                    "age_hours": (datetime.utcnow() - best_match["timestamp"]).total_seconds() / 3600
                }

            return {"response": "No data available", "source": "none"}

        orchestrator.tools = {"cache_tool": cache_aware_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["cache_tool"]
        })

        # Import random for the test
        import random
        random.seed(42)  # Make test deterministic

        result = await orchestrator.orchestrate("Find floral perfume")

        assert result is not None
        # Should use cache or live service

    @pytest.mark.asyncio
    async def test_degraded_mode_operation(self):
        """Test system operating in degraded mode with reduced functionality"""

        orchestrator = ArtisanOrchestrator()

        system_health = {
            "database": False,
            "llm": False,
            "cache": True,
            "basic_rules": True
        }

        degraded_features = []

        async def adaptive_tool(*args, **kwargs):
            features_used = []

            # Try full features
            if system_health["database"] and system_health["llm"]:
                features_used.append("full_ai")
                return {"mode": "full", "features": features_used}

            # Degraded mode 1: No database
            if system_health["llm"] and not system_health["database"]:
                features_used.append("llm_only")
                degraded_features.append("no_personalization")
                return {
                    "mode": "degraded_1",
                    "features": features_used,
                    "limitations": ["no_personalization", "no_history"]
                }

            # Degraded mode 2: No LLM
            if system_health["cache"] and not system_health["llm"]:
                features_used.append("cache_only")
                degraded_features.append("no_generation")
                return {
                    "mode": "degraded_2",
                    "features": features_used,
                    "limitations": ["no_generation", "limited_responses"]
                }

            # Minimal mode: Rules only
            if system_health["basic_rules"]:
                features_used.append("rules_only")
                degraded_features.append("minimal_functionality")
                return {
                    "mode": "minimal",
                    "features": features_used,
                    "limitations": ["basic_responses_only"]
                }

            return {"mode": "offline", "features": []}

        orchestrator.tools = {"adaptive_tool": adaptive_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["adaptive_tool"]
        })

        result = await orchestrator.orchestrate("Get service")

        # Should operate in degraded mode
        assert result["mode"] in ["degraded_1", "degraded_2", "minimal"]
        assert len(degraded_features) > 0

    @pytest.mark.asyncio
    async def test_progressive_quality_degradation(self):
        """Test progressive reduction in response quality to maintain availability"""

        orchestrator = ArtisanOrchestrator()

        quality_levels = []
        response_times = []

        async def quality_adaptive_tool(required_quality=1.0, *args, **kwargs):
            start_time = datetime.utcnow()

            # Higher quality takes longer and may fail
            processing_time = required_quality * 2.0  # seconds

            if required_quality > 0.8:
                # High quality might fail
                if random.random() > 0.3:  # 70% failure rate
                    raise TimeoutError("High quality processing timeout")

            await asyncio.sleep(min(processing_time, 0.5))  # Cap for testing

            response_time = (datetime.utcnow() - start_time).total_seconds()
            response_times.append(response_time)
            quality_levels.append(required_quality)

            return {
                "response": f"Result with quality {required_quality}",
                "actual_quality": required_quality,
                "response_time": response_time
            }

        # Try progressively lower quality
        async def progressive_degradation(*args, **kwargs):
            import random
            qualities = [1.0, 0.8, 0.6, 0.4, 0.2]

            for quality in qualities:
                try:
                    return await quality_adaptive_tool(quality, *args, **kwargs)
                except:
                    if quality == qualities[-1]:
                        return {
                            "response": "Minimum quality response",
                            "actual_quality": 0.1
                        }
                    continue

        orchestrator.tools = {"quality_tool": progressive_degradation}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["quality_tool"]
        })

        import random
        random.seed(42)

        result = await orchestrator.orchestrate("Get response")

        # Should have tried multiple quality levels
        assert len(quality_levels) >= 1
        # Final quality should be lower than initial attempt
        if len(quality_levels) > 1:
            assert quality_levels[-1] < quality_levels[0]

    @pytest.mark.asyncio
    async def test_predictive_fallback_preloading(self):
        """Test preloading fallback options based on failure prediction"""

        orchestrator = ArtisanOrchestrator()

        preloaded_fallbacks = []
        primary_attempts = 0

        # Failure predictor based on load
        class LoadPredictor:
            def __init__(self):
                self.current_load = 0.3

            def predict_failure_probability(self):
                self.current_load += 0.1
                return min(self.current_load, 0.9)

        predictor = LoadPredictor()

        async def predictive_tool(*args, **kwargs):
            nonlocal primary_attempts

            # Preload fallbacks if failure likely
            failure_prob = predictor.predict_failure_probability()

            if failure_prob > 0.5:
                # Preload fallback options
                preloaded_fallbacks.append({
                    "timestamp": datetime.utcnow(),
                    "options": ["cache", "basic_model", "rules"]
                })

            primary_attempts += 1

            # Simulate failure based on prediction
            if failure_prob > 0.6:
                raise Exception("Primary service failed as predicted")

            return {"response": "Primary service success"}

        async def smart_executor(*args, **kwargs):
            try:
                return await predictive_tool(*args, **kwargs)
            except:
                if preloaded_fallbacks:
                    # Use preloaded fallback
                    return {
                        "response": "Using preloaded fallback",
                        "fallback_ready": True
                    }
                else:
                    # Cold fallback
                    return {
                        "response": "Cold fallback",
                        "fallback_ready": False
                    }

        orchestrator.tools = {"smart_tool": smart_executor}
        orchestrator.llm_client = AsyncMock()

        # Multiple executions to test prediction
        results = []
        for i in range(5):
            orchestrator.llm_client.generate.return_value = json.dumps({
                "tools": ["smart_tool"]
            })
            result = await orchestrator.orchestrate(f"Query {i}")
            results.append(result)

        # Should have preloaded fallbacks when failure was likely
        assert len(preloaded_fallbacks) > 0
        # Later queries should use fallbacks
        assert any("fallback" in str(r).lower() for r in results[2:])

    @pytest.mark.asyncio
    async def test_distributed_fallback_consensus(self):
        """Test consensus mechanism when multiple fallback sources disagree"""

        orchestrator = ArtisanOrchestrator()

        fallback_sources = {
            "source_a": {"response": "Rose fragrance", "confidence": 0.8},
            "source_b": {"response": "Jasmine fragrance", "confidence": 0.6},
            "source_c": {"response": "Rose fragrance", "confidence": 0.7},
            "source_d": {"response": "Lily fragrance", "confidence": 0.5}
        }

        async def consensus_tool(*args, **kwargs):
            # Simulate primary failure
            if random.random() > 0.1:  # 90% failure
                # Gather fallback responses
                responses = {}
                for source, data in fallback_sources.items():
                    if random.random() > 0.3:  # Some sources might fail too
                        responses[source] = data

                if not responses:
                    return {"response": "No consensus possible"}

                # Find consensus
                response_counts = {}
                weighted_scores = {}

                for source, data in responses.items():
                    resp = data["response"]
                    conf = data["confidence"]

                    response_counts[resp] = response_counts.get(resp, 0) + 1
                    weighted_scores[resp] = weighted_scores.get(resp, 0) + conf

                # Get response with highest weighted score
                best_response = max(weighted_scores, key=weighted_scores.get)
                consensus_confidence = weighted_scores[best_response] / len(responses)

                return {
                    "response": best_response,
                    "consensus_confidence": consensus_confidence,
                    "sources_agreed": response_counts[best_response],
                    "total_sources": len(responses)
                }

            return {"response": "Primary service worked"}

        orchestrator.tools = {"consensus_tool": consensus_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["consensus_tool"]
        })

        import random
        random.seed(42)

        result = await orchestrator.orchestrate("Get fragrance")

        # Should have consensus from multiple sources
        if "sources_agreed" in result:
            assert result["sources_agreed"] >= 1
            assert result["consensus_confidence"] > 0


class TestRecoveryMechanisms:
    """Test various recovery mechanisms after failures"""

    @pytest.mark.asyncio
    async def test_stateful_recovery_with_checkpoint(self):
        """Test recovery from checkpointed state after failure"""

        orchestrator = ArtisanOrchestrator()

        checkpoints = []
        processing_stages = [
            "data_fetch",
            "preprocessing",
            "analysis",
            "generation",
            "validation",
            "finalization"
        ]

        async def checkpointed_tool(*args, **kwargs):
            completed_stages = []
            last_checkpoint = kwargs.get("resume_from", 0)

            for i, stage in enumerate(processing_stages[last_checkpoint:], last_checkpoint):
                try:
                    # Process stage
                    await asyncio.sleep(0.1)
                    completed_stages.append(stage)

                    # Save checkpoint
                    checkpoints.append({
                        "stage_index": i,
                        "stage_name": stage,
                        "timestamp": datetime.utcnow(),
                        "data": completed_stages.copy()
                    })

                    # Simulate failure at analysis stage
                    if stage == "analysis" and random.random() > 0.5:
                        raise Exception(f"Failed at {stage}")

                except Exception as e:
                    # Can resume from last checkpoint
                    return {
                        "status": "failed",
                        "completed": completed_stages,
                        "failed_at": stage,
                        "checkpoint": i,
                        "can_resume": True
                    }

            return {
                "status": "success",
                "completed": completed_stages,
                "result": "Processing complete"
            }

        # Implement recovery logic
        async def resilient_processor(*args, **kwargs):
            max_retries = 3
            last_checkpoint = 0

            for attempt in range(max_retries):
                result = await checkpointed_tool(
                    *args,
                    resume_from=last_checkpoint,
                    **kwargs
                )

                if result["status"] == "success":
                    return result

                if result.get("can_resume"):
                    last_checkpoint = result["checkpoint"]
                    continue

                break

            return result

        orchestrator.tools = {"processor": resilient_processor}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["processor"]
        })

        import random
        random.seed(42)

        result = await orchestrator.orchestrate("Process data")

        # Should have checkpoints
        assert len(checkpoints) > 0
        # Should eventually complete or have partial results
        assert result is not None

    @pytest.mark.asyncio
    async def test_compensating_transactions(self):
        """Test compensating transactions to rollback partial operations"""

        orchestrator = ArtisanOrchestrator()

        operations_log = []
        compensation_log = []

        async def transactional_tool(*args, **kwargs):
            operations = [
                ("allocate_resources", "release_resources"),
                ("write_temp_data", "delete_temp_data"),
                ("update_state", "revert_state"),
                ("send_notification", "cancel_notification")
            ]

            completed_ops = []

            try:
                for op, compensation in operations:
                    # Execute operation
                    operations_log.append({
                        "operation": op,
                        "timestamp": datetime.utcnow()
                    })
                    completed_ops.append((op, compensation))

                    # Simulate failure
                    if op == "update_state":
                        raise Exception("State update failed")

                return {"status": "success", "operations": len(completed_ops)}

            except Exception as e:
                # Execute compensating transactions in reverse order
                for op, compensation in reversed(completed_ops):
                    compensation_log.append({
                        "compensation": compensation,
                        "for_operation": op,
                        "timestamp": datetime.utcnow()
                    })

                return {
                    "status": "rolled_back",
                    "operations_attempted": len(completed_ops),
                    "compensations_executed": len(compensation_log)
                }

        orchestrator.tools = {"transactional": transactional_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["transactional"]
        })

        result = await orchestrator.orchestrate("Execute transaction")

        # Should have executed compensating transactions
        assert len(compensation_log) > 0
        # Compensations should match operations
        assert len(compensation_log) == len(operations_log) - 1  # -1 for failed operation
        # Should be rolled back
        assert result["status"] == "rolled_back"

    @pytest.mark.asyncio
    async def test_adaptive_timeout_adjustment(self):
        """Test dynamic timeout adjustment based on system conditions"""

        orchestrator = ArtisanOrchestrator()

        timeout_history = []
        response_times = []

        class AdaptiveTimeout:
            def __init__(self):
                self.base_timeout = 1.0
                self.history = []

            def calculate_timeout(self):
                if not self.history:
                    return self.base_timeout

                # Calculate adaptive timeout based on recent history
                recent = self.history[-5:]
                avg_time = sum(recent) / len(recent)
                std_dev = (sum((x - avg_time) ** 2 for x in recent) / len(recent)) ** 0.5

                # Set timeout to avg + 2 * std_dev
                adaptive_timeout = avg_time + 2 * std_dev
                return max(self.base_timeout, min(adaptive_timeout, 10.0))

            def record_response_time(self, response_time):
                self.history.append(response_time)

        timeout_manager = AdaptiveTimeout()

        async def adaptive_tool(*args, **kwargs):
            # Variable response time
            response_time = random.uniform(0.5, 2.0)
            timeout = timeout_manager.calculate_timeout()
            timeout_history.append(timeout)

            try:
                start = datetime.utcnow()
                await asyncio.wait_for(
                    asyncio.sleep(response_time),
                    timeout=timeout
                )
                actual_time = (datetime.utcnow() - start).total_seconds()
                response_times.append(actual_time)
                timeout_manager.record_response_time(actual_time)

                return {"status": "success", "response_time": actual_time}

            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "timeout_used": timeout,
                    "attempted_time": response_time
                }

        orchestrator.tools = {"adaptive": adaptive_tool}
        orchestrator.llm_client = AsyncMock()

        import random
        random.seed(42)

        # Multiple calls to test adaptation
        for i in range(10):
            orchestrator.llm_client.generate.return_value = json.dumps({
                "tools": ["adaptive"]
            })
            await orchestrator.orchestrate(f"Request {i}")

        # Timeouts should adapt over time
        if len(timeout_history) > 5:
            # Later timeouts should be different from initial
            assert timeout_history[-1] != timeout_history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])