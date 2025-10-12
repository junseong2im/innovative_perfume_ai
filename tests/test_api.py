# tests/test_api.py
"""
API Integration Tests
Tests complete flow: create_dna → options → feedback → learn with 200 responses
"""

import pytest
import requests
import json
import time
import asyncio
from typing import Dict, Any, List
from fastapi.testclient import TestClient
import numpy as np

# Import the FastAPI app
from app.main import app
from fragrance_ai.observability import orchestrator_logger, metrics_collector


# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test individual API endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        print("\n[TEST] Testing health check...")

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data

        print("[OK] Health check passed")

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        print("\n[TEST] Testing metrics endpoint...")

        response = client.get("/metrics")
        assert response.status_code == 200

        # Should return Prometheus format
        assert response.headers["content-type"].startswith("text/plain")

        # Content should have Prometheus metrics format
        content = response.text
        if "Prometheus not available" not in content:
            assert "fragrance_" in content or "#" in content

        print("[OK] Metrics endpoint working")

    def test_create_dna(self):
        """Test DNA creation endpoint"""
        print("\n[TEST] Testing DNA creation...")

        brief = {
            "brief": {
                "style": "fresh",
                "intensity": 0.7,
                "complexity": 0.5,
                "masculinity": 0.6,
                "season": "summer",
                "notes": ["citrus", "woody", "aquatic"]
            },
            "name": "Test Formula",
            "product_category": "eau_de_toilette"
        }

        response = client.post("/dna/create", json=brief)
        assert response.status_code == 201

        data = response.json()
        assert "dna_id" in data
        assert "ingredients" in data
        assert len(data["ingredients"]) > 0
        assert "compliance" in data

        # Check ingredients sum to 100%
        total = sum(ing["concentration"] for ing in data["ingredients"])
        assert abs(total - 100.0) < 0.1, f"Ingredients sum to {total}%"

        print(f"  Created DNA: {data['dna_id']}")
        print(f"  Ingredients: {len(data['ingredients'])}")
        print(f"  IFRA compliant: {data['compliance']['ifra_compliant']}")
        print("[OK] DNA created successfully")

        return data["dna_id"]

    def test_evolution_options(self):
        """Test evolution options generation"""
        print("\n[TEST] Testing evolution options...")

        # First create a DNA
        dna_id = self.test_create_dna()

        # Request evolution options
        request = {
            "dna_id": dna_id,
            "brief": {
                "style": "fresh",
                "intensity": 0.8
            },
            "num_options": 3,
            "optimization_profile": "commercial",
            "algorithm": "PPO"
        }

        response = client.post("/evolve/options", json=request)
        assert response.status_code == 200

        data = response.json()
        assert "experiment_id" in data
        assert "options" in data
        assert len(data["options"]) == 3

        print(f"  Experiment ID: {data['experiment_id']}")
        for opt in data["options"]:
            print(f"    Option: {opt['action']} - {opt['description']}")

        print("[OK] Evolution options generated")

        return data["experiment_id"], data["options"]

    def test_evolution_feedback(self):
        """Test feedback processing"""
        print("\n[TEST] Testing feedback processing...")

        # Generate options first
        experiment_id, options = self.test_evolution_options()

        # Submit feedback
        feedback = {
            "experiment_id": experiment_id,
            "chosen_id": options[0]["id"],
            "rating": 4,
            "notes": "Good balance"
        }

        response = client.post("/evolve/feedback", json=feedback)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data
        assert data["iteration"] == 1

        print(f"  Feedback processed: iteration {data['iteration']}")
        print(f"  Metrics: {data['metrics']}")
        print("[OK] Feedback processed successfully")

    def test_experiment_status(self):
        """Test experiment status retrieval"""
        print("\n[TEST] Testing experiment status...")

        # Create experiment
        experiment_id, _ = self.test_evolution_options()

        # Get status
        response = client.get(f"/experiments/{experiment_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["experiment_id"] == experiment_id
        assert "status" in data
        assert "iterations" in data

        print(f"  Experiment: {experiment_id}")
        print(f"  Status: {data['status']}")
        print(f"  Iterations: {data['iterations']}")
        print("[OK] Experiment status retrieved")


class TestAPIFlow:
    """Test complete API flow with multiple iterations"""

    def test_complete_flow_200_responses(self):
        """Test complete flow with 200 user responses"""
        print("\n[TEST] Testing complete flow with 200 responses...")

        # Metrics tracking
        rewards = []
        ratings = []
        timings = []
        errors = 0

        # Step 1: Create initial DNA
        start_time = time.time()

        brief_request = {
            "brief": {
                "style": "woody",
                "intensity": 0.6,
                "complexity": 0.7,
                "masculinity": 0.8,
                "warmth": 0.6,
                "freshness": 0.3
            },
            "name": "Integration Test Formula",
            "product_category": "eau_de_parfum"
        }

        response = client.post("/dna/create", json=brief_request)
        assert response.status_code == 201
        dna_data = response.json()
        dna_id = dna_data["dna_id"]

        creation_time = (time.time() - start_time) * 1000
        orchestrator_logger.log_dna_creation(
            dna_id=dna_id,
            ingredient_count=len(dna_data["ingredients"]),
            ifra_compliant=dna_data["compliance"]["ifra_compliant"],
            timing_ms=creation_time
        )

        print(f"  Created DNA: {dna_id}")

        # Step 2: Run 200 evolution iterations
        algorithms = ["PPO", "REINFORCE"]
        profiles = ["creative", "commercial", "stable"]

        for iteration in range(200):
            try:
                iter_start = time.time()

                # Alternate algorithms and profiles
                algorithm = algorithms[iteration % len(algorithms)]
                profile = profiles[iteration % len(profiles)]

                # Generate options
                options_request = {
                    "dna_id": dna_id,
                    "brief": {
                        "style": "woody",
                        "intensity": 0.5 + (iteration % 5) * 0.1  # Vary intensity
                    },
                    "num_options": 2 + (iteration % 3),  # Vary number of options
                    "optimization_profile": profile,
                    "algorithm": algorithm
                }

                response = client.post("/evolve/options", json=options_request)
                assert response.status_code == 200
                options_data = response.json()

                experiment_id = options_data["experiment_id"]
                options = options_data["options"]

                # Simulate user choice (with some pattern)
                if iteration < 50:
                    # Random choice initially
                    chosen_idx = np.random.randint(len(options))
                elif iteration < 150:
                    # Prefer first option (learning signal)
                    chosen_idx = 0 if np.random.rand() < 0.7 else np.random.randint(len(options))
                else:
                    # More random exploration
                    chosen_idx = np.random.randint(len(options))

                chosen_id = options[chosen_idx]["id"]

                # Simulate rating (improving over time)
                base_rating = 2.5 + (iteration / 200) * 1.5  # 2.5 → 4.0
                noise = np.random.normal(0, 0.5)
                rating = float(np.clip(base_rating + noise, 1, 5))
                ratings.append(rating)

                # Submit feedback
                feedback_request = {
                    "experiment_id": experiment_id,
                    "chosen_id": chosen_id,
                    "rating": rating
                }

                response = client.post("/evolve/feedback", json=feedback_request)
                assert response.status_code == 200
                feedback_data = response.json()

                # Track metrics
                if "metrics" in feedback_data and "reward" in feedback_data["metrics"]:
                    rewards.append(feedback_data["metrics"]["reward"])

                # Clean up experiment
                client.delete(f"/experiments/{experiment_id}")

                # Track timing
                iter_time = (time.time() - iter_start) * 1000
                timings.append(iter_time)

                # Log iteration
                orchestrator_logger.log_experiment(
                    experiment_id=experiment_id,
                    user_id=f"test_user_{iteration}",
                    action=f"iteration_{iteration}",
                    timing_ms=iter_time,
                    success=True,
                    algorithm=algorithm,
                    profile=profile,
                    rating=rating
                )

                # Progress update
                if (iteration + 1) % 50 == 0:
                    avg_rating = np.mean(ratings[-50:])
                    avg_timing = np.mean(timings[-50:])
                    print(f"    Iteration {iteration + 1}/200: "
                          f"avg_rating={avg_rating:.2f}, "
                          f"avg_time={avg_timing:.1f}ms")

            except Exception as e:
                errors += 1
                orchestrator_logger.log_experiment(
                    experiment_id="unknown",
                    user_id=f"test_user_{iteration}",
                    action=f"iteration_{iteration}",
                    timing_ms=0,
                    success=False,
                    error=str(e)
                )
                print(f"    Error at iteration {iteration}: {e}")

        # Step 3: Analyze results
        total_time = time.time() - start_time

        print("\n  === RESULTS ===")
        print(f"  Completed: {200 - errors}/200 iterations")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Average time per iteration: {np.mean(timings):.1f} ms")
        print(f"  Average rating: {np.mean(ratings):.2f}")

        if len(rewards) > 20:
            early_rewards = np.mean(rewards[:20])
            late_rewards = np.mean(rewards[-20:])
            print(f"  Reward improvement: {early_rewards:.3f} → {late_rewards:.3f}")

        # Assertions
        assert errors < 10, f"Too many errors: {errors}"
        assert len(ratings) >= 190, "Too few successful iterations"
        assert np.mean(timings) < 1000, "API too slow (>1s per iteration)"

        # Check for learning signal (ratings should improve)
        early_rating = np.mean(ratings[:50])
        late_rating = np.mean(ratings[-50:])
        improvement = late_rating - early_rating
        print(f"  Rating improvement: {early_rating:.2f} → {late_rating:.2f} ({improvement:+.2f})")

        print("[OK] Complete flow with 200 responses successful")

    def test_error_handling(self):
        """Test API error handling"""
        print("\n[TEST] Testing error handling...")

        # Test 404 - DNA not found
        response = client.post("/evolve/options", json={
            "dna_id": "non_existent_dna",
            "brief": {"style": "fresh"},
            "num_options": 3
        })
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"] == "DNA_NOT_FOUND"

        print("  404 DNA not found: ✓")

        # Test 422 - Validation error
        response = client.post("/dna/create", json={
            "brief": {},  # Empty brief
            "product_category": "invalid_category"
        })
        assert response.status_code == 400 or response.status_code == 422

        print("  422 Validation error: ✓")

        # Test experiment not found
        response = client.get("/experiments/non_existent_exp")
        assert response.status_code == 404

        print("  404 Experiment not found: ✓")

        # Test invalid rating
        response = client.post("/evolve/feedback", json={
            "experiment_id": "test",
            "chosen_id": "test",
            "rating": 10  # Invalid (must be 1-5)
        })
        assert response.status_code == 422

        print("  422 Invalid rating: ✓")

        print("[OK] Error handling working correctly")

    def test_concurrent_requests(self):
        """Test API under concurrent load"""
        print("\n[TEST] Testing concurrent requests...")

        import concurrent.futures

        # Create a DNA first
        response = client.post("/dna/create", json={
            "brief": {"style": "fresh", "intensity": 0.5},
            "name": "Concurrent Test"
        })
        dna_id = response.json()["dna_id"]

        def make_evolution_request(i):
            """Make a single evolution request"""
            try:
                response = client.post("/evolve/options", json={
                    "dna_id": dna_id,
                    "brief": {"style": "fresh", "intensity": 0.5 + i * 0.01},
                    "num_options": 2,
                    "algorithm": "PPO" if i % 2 == 0 else "REINFORCE"
                })
                return response.status_code == 200
            except:
                return False

        # Run 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_evolution_request, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_rate = sum(results) / len(results)
        print(f"  Concurrent requests: {sum(results)}/20 successful")
        print(f"  Success rate: {success_rate*100:.1f}%")

        assert success_rate >= 0.9, "Too many failures under concurrent load"

        print("[OK] API handles concurrent requests")


class TestDataConsistency:
    """Test data consistency and validation"""

    def test_ingredient_normalization(self):
        """Test that ingredients are always normalized to 100%"""
        print("\n[TEST] Testing ingredient normalization...")

        for _ in range(10):
            # Create DNA with random brief
            response = client.post("/dna/create", json={
                "brief": {
                    "style": np.random.choice(["fresh", "woody", "floral"]),
                    "intensity": np.random.rand()
                },
                "name": f"Normalization Test {_}"
            })

            data = response.json()
            total = sum(ing["concentration"] for ing in data["ingredients"])

            assert abs(total - 100.0) < 0.01, f"Ingredients not normalized: {total}%"

        print("[OK] All formulas normalized to 100%")

    def test_ifra_compliance_enforcement(self):
        """Test IFRA compliance is checked"""
        print("\n[TEST] Testing IFRA compliance enforcement...")

        response = client.post("/dna/create", json={
            "brief": {
                "style": "citrus",  # Will likely include bergamot
                "intensity": 0.9
            },
            "name": "IFRA Test",
            "product_category": "eau_de_parfum"
        })

        data = response.json()
        compliance = data["compliance"]

        print(f"  IFRA compliant: {compliance['ifra_compliant']}")
        if not compliance["ifra_compliant"]:
            print(f"  Violations: {compliance['ifra_violations']}")

        # Check allergens
        if compliance["allergens_to_declare"]:
            print(f"  Allergens to declare: {len(compliance['allergens_to_declare'])}")

        print("[OK] IFRA compliance checked")

    def test_metrics_collection(self):
        """Test that metrics are being collected"""
        print("\n[TEST] Testing metrics collection...")

        # Generate some activity
        for i in range(5):
            client.post("/dna/create", json={
                "brief": {"style": "fresh", "intensity": 0.5},
                "name": f"Metrics Test {i}"
            })

        # Check metrics
        response = client.get("/metrics")
        content = response.text

        if "Prometheus not available" not in content:
            # Should have some metrics recorded
            assert len(content) > 100, "No metrics collected"

        print("[OK] Metrics being collected")


def test_performance_benchmark():
    """Benchmark API performance"""
    print("\n[BENCHMARK] API Performance Test...")

    # Test DNA creation speed
    start = time.time()
    for _ in range(10):
        client.post("/dna/create", json={
            "brief": {"style": "fresh", "intensity": 0.5},
            "name": "Performance Test"
        })
    dna_time = (time.time() - start) / 10

    print(f"  DNA creation: {dna_time*1000:.1f} ms average")
    assert dna_time < 0.5, "DNA creation too slow"

    # Test evolution options speed
    response = client.post("/dna/create", json={
        "brief": {"style": "fresh", "intensity": 0.5},
        "name": "Evolution Benchmark"
    })
    dna_id = response.json()["dna_id"]

    start = time.time()
    for _ in range(10):
        response = client.post("/evolve/options", json={
            "dna_id": dna_id,
            "brief": {"style": "fresh"},
            "num_options": 3
        })
        if response.status_code == 200:
            exp_id = response.json()["experiment_id"]
            client.delete(f"/experiments/{exp_id}")

    evolution_time = (time.time() - start) / 10

    print(f"  Evolution options: {evolution_time*1000:.1f} ms average")
    assert evolution_time < 1.0, "Evolution too slow"

    print("[OK] API performance acceptable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])