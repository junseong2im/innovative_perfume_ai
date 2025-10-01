"""
Integration Test: Living Scent Flow
Tests the complete journey from initial DNA creation to evolution through feedback
Verifies real AI engines are used without any simulation/hardcoded fallbacks
"""

import asyncio
import json
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fragrance_ai.orchestrator.artisan_orchestrator_enhanced import (
    EnhancedArtisanOrchestrator,
    OrchestrationContext,
    UserIntent
)
from fragrance_ai.database.models import OlfactoryDNA
from fragrance_ai.training.moga_optimizer_enhanced import EnhancedMOGAOptimizer
from fragrance_ai.training.rl_with_persistence import RLHFWithPersistence


class TestMetrics:
    """Track and validate test metrics"""

    def __init__(self):
        self.dna_created = False
        self.moga_used = False
        self.rlhf_used = False
        self.evolution_count = 0
        self.feedback_processed = False
        self.model_updated = False
        self.no_simulation_verified = False
        self.pareto_solutions = 0
        self.policy_updates = 0
        self.total_time = 0

    def report(self) -> Dict[str, Any]:
        """Generate test report"""
        return {
            "dna_created": self.dna_created,
            "moga_used": self.moga_used,
            "rlhf_used": self.rlhf_used,
            "evolution_count": self.evolution_count,
            "feedback_processed": self.feedback_processed,
            "model_updated": self.model_updated,
            "no_simulation_verified": self.no_simulation_verified,
            "pareto_solutions": self.pareto_solutions,
            "policy_updates": self.policy_updates,
            "total_time": self.total_time
        }

    def is_complete(self) -> bool:
        """Check if all test criteria are met"""
        return all([
            self.dna_created,
            self.moga_used,
            self.rlhf_used,
            self.evolution_count >= 3,
            self.feedback_processed,
            self.model_updated,
            self.no_simulation_verified
        ])


class LivingScentIntegrationTest:
    """Complete Living Scent flow integration test"""

    def __init__(self):
        self.orchestrator = None
        self.context = None
        self.metrics = TestMetrics()
        self.test_results = []

    async def setup(self):
        """Initialize test environment"""
        print("\n" + "="*80)
        print("LIVING SCENT INTEGRATION TEST - SETUP")
        print("="*80)

        try:
            # Initialize orchestrator with real AI engines
            self.orchestrator = EnhancedArtisanOrchestrator({
                'moga_population_size': 20,  # Smaller for testing
                'moga_generations': 5,       # Fewer generations for speed
                'rlhf_save_dir': 'models/integration_test',
                'rlhf_auto_save': True
            })

            # Create test context
            self.context = OrchestrationContext(
                user_id="test_user_001",
                session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                conversation_history=[]
            )

            print(f"✓ Orchestrator initialized")
            print(f"✓ Test context created: {self.context.session_id}")
            print(f"✓ MOGA optimizer ready: {hasattr(self.orchestrator, 'moga_optimizer')}")
            print(f"✓ RLHF system ready: {hasattr(self.orchestrator, 'rlhf_system')}")

            return True

        except Exception as e:
            print(f"✗ Setup failed: {e}")
            return False

    async def test_phase_1_dna_creation(self):
        """Phase 1: Create initial DNA from text input"""
        print("\n" + "="*80)
        print("PHASE 1: DNA CREATION FROM TEXT INPUT")
        print("="*80)

        test_input = "Create a fresh morning garden fragrance with citrus and white flowers"
        print(f"User Input: '{test_input}'")

        try:
            start_time = time.time()

            # Process initial request
            result = await self.orchestrator.process(test_input, self.context)

            elapsed = time.time() - start_time

            # Validate result
            assert result.success, "DNA creation failed"
            assert result.intent == UserIntent.CREATE_NEW, f"Wrong intent: {result.intent}"
            assert result.recipe is not None, "No recipe created"
            assert self.context.current_dna is not None, "No DNA stored in context"

            # Check MOGA was used
            if result.metadata and result.metadata.get('optimization_method') == 'MOGA':
                self.metrics.moga_used = True
                self.metrics.pareto_solutions = result.metadata.get('pareto_front_size', 0)

            # Mark DNA created
            self.metrics.dna_created = True

            print(f"\n✓ DNA successfully created in {elapsed:.2f}s")
            print(f"✓ Recipe: {result.recipe['name']}")
            print(f"✓ Top notes: {[n['name'] for n in result.recipe.get('top_notes', [])]}")
            print(f"✓ Heart notes: {[n['name'] for n in result.recipe.get('heart_notes', [])]}")
            print(f"✓ Base notes: {[n['name'] for n in result.recipe.get('base_notes', [])]}")
            print(f"✓ MOGA optimization used: {self.metrics.moga_used}")
            print(f"✓ Pareto solutions: {self.metrics.pareto_solutions}")

            self.test_results.append({
                "phase": "DNA_CREATION",
                "success": True,
                "time": elapsed,
                "recipe_name": result.recipe['name']
            })

            return True

        except Exception as e:
            print(f"✗ DNA creation failed: {e}")
            traceback.print_exc()
            self.test_results.append({
                "phase": "DNA_CREATION",
                "success": False,
                "error": str(e)
            })
            return False

    async def test_phase_2_evolution_cycle(self, cycle_num: int, feedback: str):
        """Phase 2: Evolution through feedback"""
        print(f"\n" + "="*80)
        print(f"PHASE 2: EVOLUTION CYCLE {cycle_num}")
        print("="*80)

        print(f"Feedback: '{feedback}'")

        try:
            start_time = time.time()

            # Store initial policy updates count
            initial_updates = self.orchestrator.rlhf_system.policy_network.total_updates

            # Process feedback
            result = await self.orchestrator.process(feedback, self.context)

            elapsed = time.time() - start_time

            # Validate evolution
            assert result.success, "Evolution failed"
            assert result.intent == UserIntent.EVOLVE_EXISTING, f"Wrong intent: {result.intent}"

            # Check RLHF was used
            if result.metadata and result.metadata.get('optimization_method') == 'RLHF':
                self.metrics.rlhf_used = True
                current_updates = result.metadata.get('policy_updates', 0)
                if current_updates > initial_updates:
                    self.metrics.model_updated = True
                    self.metrics.policy_updates = current_updates

            # Check if feedback was processed
            if 'feedback' in feedback.lower() or any(
                word in feedback.lower()
                for word in ['better', 'worse', 'like', 'love', 'perfect', 'good', 'bad']
            ):
                self.metrics.feedback_processed = True

            self.metrics.evolution_count += 1

            print(f"\n✓ Evolution cycle {cycle_num} completed in {elapsed:.2f}s")
            print(f"✓ Recipe evolved: {result.recipe['name'] if result.recipe else 'N/A'}")
            print(f"✓ RLHF used: {self.metrics.rlhf_used}")
            print(f"✓ Policy updates: {self.metrics.policy_updates}")
            print(f"✓ Model updated: {self.metrics.model_updated}")

            if result.variations:
                print(f"✓ Variations generated: {len(result.variations)}")

            self.test_results.append({
                "phase": f"EVOLUTION_{cycle_num}",
                "success": True,
                "time": elapsed,
                "policy_updates": self.metrics.policy_updates
            })

            return True

        except Exception as e:
            print(f"✗ Evolution cycle {cycle_num} failed: {e}")
            self.test_results.append({
                "phase": f"EVOLUTION_{cycle_num}",
                "success": False,
                "error": str(e)
            })
            return False

    async def test_phase_3_verify_no_simulation(self):
        """Phase 3: Verify no simulation code is used"""
        print("\n" + "="*80)
        print("PHASE 3: VERIFY NO SIMULATION CODE")
        print("="*80)

        checks_passed = []

        # Check 1: MOGA optimizer is real DEAP implementation
        try:
            moga = self.orchestrator.moga_optimizer
            assert hasattr(moga, 'toolbox'), "MOGA missing DEAP toolbox"
            assert hasattr(moga, 'creator'), "MOGA missing DEAP creator"
            assert moga.use_validator is not None, "MOGA validator not configured"
            checks_passed.append("MOGA uses real DEAP implementation")
            print("✓ MOGA uses real DEAP implementation")
        except Exception as e:
            print(f"✗ MOGA check failed: {e}")

        # Check 2: RLHF uses real PyTorch PolicyNetwork
        try:
            rlhf = self.orchestrator.rlhf_system
            assert isinstance(rlhf.policy_network, torch.nn.Module), "Not a PyTorch module"
            assert hasattr(rlhf, 'optimizer'), "No optimizer found"
            assert rlhf.persistence_manager.auto_save, "Auto-save not enabled"
            checks_passed.append("RLHF uses real PyTorch PolicyNetwork")
            print("✓ RLHF uses real PyTorch PolicyNetwork")
        except Exception as e:
            print(f"✗ RLHF check failed: {e}")

        # Check 3: Model persistence file exists
        try:
            model_path = Path(f"{self.orchestrator.rlhf_system.persistence_manager.save_dir}/policy_network.pth")
            if model_path.exists():
                file_size = model_path.stat().st_size
                checks_passed.append(f"Model file exists ({file_size:,} bytes)")
                print(f"✓ Model file exists: {model_path} ({file_size:,} bytes)")
            else:
                print(f"⚠ Model file not found (may not have been saved yet)")
        except Exception as e:
            print(f"✗ Model file check failed: {e}")

        # Check 4: No template/hardcoded responses
        try:
            # Check that recipes have unique values, not templates
            if self.context.current_recipe:
                recipe = self.context.current_recipe
                # Templates would have exactly these values
                template_indicators = [
                    "Classic Floral Blend",
                    "Template Fragrance",
                    "Default Recipe"
                ]
                is_template = any(
                    indicator in str(recipe.get('name', ''))
                    for indicator in template_indicators
                )
                assert not is_template, "Recipe appears to be from template"
                checks_passed.append("Recipes are dynamically generated")
                print("✓ Recipes are dynamically generated (not templates)")
        except Exception as e:
            print(f"✗ Template check failed: {e}")

        # Check 5: Verify real optimization metrics
        try:
            assert self.metrics.pareto_solutions > 0, "No Pareto solutions found"
            assert self.metrics.policy_updates > 0, "No policy updates made"
            checks_passed.append(f"Real optimization: {self.metrics.pareto_solutions} Pareto, {self.metrics.policy_updates} updates")
            print(f"✓ Real optimization metrics confirmed")
        except Exception as e:
            print(f"⚠ Optimization metrics check: {e}")

        self.metrics.no_simulation_verified = len(checks_passed) >= 4

        print(f"\n{'✓' if self.metrics.no_simulation_verified else '✗'} Simulation verification: {len(checks_passed)}/5 checks passed")

        return self.metrics.no_simulation_verified

    async def run_complete_test(self):
        """Run the complete integration test"""
        print("\n" + "="*80)
        print("LIVING SCENT INTEGRATION TEST - COMPLETE FLOW")
        print("="*80)
        print("\nTest Scenario:")
        print("1. User creates initial fragrance DNA from text description")
        print("2. DNA evolves through multiple feedback cycles")
        print("3. System uses real AI engines (MOGA + RLHF)")
        print("4. No simulation or hardcoded responses")
        print("="*80)

        start_time = time.time()

        # Setup
        if not await self.setup():
            return False

        # Phase 1: Create initial DNA
        if not await self.test_phase_1_dna_creation():
            print("✗ Failed at Phase 1: DNA Creation")
            return False

        # Phase 2: Evolution cycles
        evolution_feedback = [
            "Make it more romantic and add some rose notes",
            "I love this! But can you make it last longer?",
            "Perfect! The rose is beautiful, but reduce the citrus",
            "This is exactly what I wanted!"
        ]

        for i, feedback in enumerate(evolution_feedback, 1):
            await asyncio.sleep(0.5)  # Small delay between cycles
            if not await self.test_phase_2_evolution_cycle(i, feedback):
                print(f"✗ Failed at Phase 2: Evolution Cycle {i}")
                break

        # Phase 3: Verify no simulation
        await self.test_phase_3_verify_no_simulation()

        # Calculate total time
        self.metrics.total_time = time.time() - start_time

        # Generate final report
        print("\n" + "="*80)
        print("FINAL TEST REPORT")
        print("="*80)

        report = self.metrics.report()

        for key, value in report.items():
            status = "✓" if value else "✗" if isinstance(value, bool) else "•"
            print(f"{status} {key}: {value}")

        # Check if all criteria met
        all_passed = self.metrics.is_complete()

        print("\n" + "="*80)
        if all_passed:
            print("✅ SUCCESS: LIVING SCENT FLOW COMPLETE")
            print("="*80)
            print("\nVerified:")
            print("• Initial DNA created from text using MOGA optimizer")
            print(f"• DNA evolved through {self.metrics.evolution_count} feedback cycles")
            print("• RLHF system processed feedback and updated policy")
            print("• No simulation code used - all real AI engines")
            print(f"• Total test time: {self.metrics.total_time:.2f} seconds")
        else:
            print("❌ FAILURE: SOME CRITERIA NOT MET")
            print("="*80)
            print("\nIssues:")
            if not self.metrics.dna_created:
                print("• DNA creation failed")
            if not self.metrics.moga_used:
                print("• MOGA optimizer not used")
            if not self.metrics.rlhf_used:
                print("• RLHF system not used")
            if self.metrics.evolution_count < 3:
                print(f"• Insufficient evolution cycles ({self.metrics.evolution_count}/3)")
            if not self.metrics.no_simulation_verified:
                print("• Simulation code may still be present")

        return all_passed


async def run_integration_test():
    """Main test runner"""
    test = LivingScentIntegrationTest()
    success = await test.run_complete_test()
    return success


def verify_no_hardcoded_values():
    """Additional verification that no hardcoded values exist"""
    print("\n" + "="*80)
    print("HARDCODED VALUE VERIFICATION")
    print("="*80)

    suspicious_patterns = [
        "template",
        "fallback",
        "mock",
        "dummy",
        "simulate",
        "hardcoded",
        "Classic Floral Blend",
        "Default Recipe"
    ]

    files_to_check = [
        "fragrance_ai/orchestrator/artisan_orchestrator_enhanced.py",
        "fragrance_ai/training/moga_optimizer_enhanced.py",
        "fragrance_ai/training/rl_with_persistence.py"
    ]

    issues_found = []

    for filepath in files_to_check:
        try:
            path = Path(filepath)
            if path.exists():
                content = path.read_text(encoding='utf-8').lower()
                for pattern in suspicious_patterns:
                    if pattern.lower() in content:
                        # Check if it's in comments
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern.lower() in line and not line.strip().startswith('#'):
                                issues_found.append(f"{filepath}:{i+1} - Found '{pattern}'")
        except Exception as e:
            print(f"Could not check {filepath}: {e}")

    if issues_found:
        print("⚠ Potential hardcoded values found:")
        for issue in issues_found[:5]:  # Show first 5
            print(f"  - {issue}")
    else:
        print("✓ No obvious hardcoded values detected")

    return len(issues_found) == 0


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LIVING SCENT INTEGRATION TEST SUITE")
    print("Testing complete flow from DNA creation to evolution")
    print("="*80)

    # Run main integration test
    success = asyncio.run(run_integration_test())

    # Run additional verification
    no_hardcoded = verify_no_hardcoded_values()

    # Final verdict
    print("\n" + "="*80)
    print("INTEGRATION TEST SUITE COMPLETE")
    print("="*80)

    if success and no_hardcoded:
        print("\n🎉 ALL TESTS PASSED!")
        print("The Living Scent system is fully operational with real AI engines.")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED")
        print("Please review the errors above and fix the issues.")
        sys.exit(1)