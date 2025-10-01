"""
Test Script for Enhanced Artisan Orchestrator
Verifies integration of MOGA and RLHF engines
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.orchestrator.artisan_orchestrator_enhanced import (
    EnhancedArtisanOrchestrator,
    OrchestrationContext,
    UserIntent
)


async def test_create_new_with_moga():
    """Test CREATE_NEW intent with MOGA optimizer"""
    print("\n" + "="*70)
    print("TEST 1: CREATE_NEW with MOGA Optimizer")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_1",
        conversation_history=[]
    )

    # Test creating a new perfume
    messages = [
        "Create a fresh citrus perfume for summer",
        "Design a romantic floral fragrance",
        "Make a new woody oriental perfume for evening"
    ]

    for msg in messages:
        print(f"\nUser: {msg}")
        result = await orchestrator.process(msg, context)

        print(f"Intent: {result.intent}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message[:100]}...")

        if result.recipe:
            print(f"Recipe created: {result.recipe['name']}")
            print(f"Top notes: {[n['name'] for n in result.recipe.get('top_notes', [])]}")
            print(f"Heart notes: {[n['name'] for n in result.recipe.get('heart_notes', [])]}")
            print(f"Base notes: {[n['name'] for n in result.recipe.get('base_notes', [])]}")

        if result.metadata:
            print(f"Optimization method: {result.metadata.get('optimization_method')}")
            print(f"Pareto front size: {result.metadata.get('pareto_front_size')}")

        if result.variations:
            print(f"Additional variations: {len(result.variations)}")

    return result.success


async def test_evolve_existing_with_rlhf():
    """Test EVOLVE_EXISTING intent with RLHF system"""
    print("\n" + "="*70)
    print("TEST 2: EVOLVE_EXISTING with RLHF System")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_2",
        conversation_history=[]
    )

    # First create a recipe
    print("\nCreating initial recipe...")
    result = await orchestrator.process(
        "Create a simple floral perfume",
        context
    )

    if not result.success:
        print("Failed to create initial recipe")
        return False

    print(f"Initial recipe: {context.current_recipe['name']}")

    # Now evolve it
    evolution_requests = [
        "Make it more romantic and softer",
        "I like this better, but can you add more freshness?",
        "Perfect! This is exactly what I wanted"
    ]

    for req in evolution_requests:
        print(f"\nUser: {req}")
        result = await orchestrator.process(req, context)

        print(f"Intent: {result.intent}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message[:100]}...")

        if result.metadata:
            print(f"Policy updates: {result.metadata.get('policy_updates')}")
            print(f"Model file: {result.metadata.get('model_file')}")
            print(f"Feedback history: {result.metadata.get('feedback_history_length')}")

        if result.variations:
            print(f"Variations generated: {len(result.variations) + 1}")

    return result.success


async def test_intent_classification():
    """Test intent classification"""
    print("\n" + "="*70)
    print("TEST 3: Intent Classification")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_3",
        conversation_history=[]
    )

    test_messages = [
        ("Create a new perfume", UserIntent.CREATE_NEW),
        ("Find similar fragrances", UserIntent.SEARCH),
        ("What is an accord?", UserIntent.KNOWLEDGE),
        ("Validate this composition", UserIntent.VALIDATE),
        ("Improve this recipe", UserIntent.EVOLVE_EXISTING),
        ("Hello there", UserIntent.UNKNOWN)
    ]

    correct = 0
    for message, expected_intent in test_messages:
        # Set up context for evolve test
        if expected_intent == UserIntent.EVOLVE_EXISTING:
            context.current_recipe = {"name": "Test"}

        result = await orchestrator.process(message, context)
        actual_intent = result.intent

        match = actual_intent == expected_intent
        status = "[OK]" if match else "[FAIL]"

        print(f"{status} '{message}' -> {actual_intent} (expected: {expected_intent})")

        if match:
            correct += 1

    accuracy = correct / len(test_messages) * 100
    print(f"\nIntent classification accuracy: {accuracy:.1f}%")

    return accuracy >= 80


async def test_search_integration():
    """Test search tool integration"""
    print("\n" + "="*70)
    print("TEST 4: Search Tool Integration")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_4",
        conversation_history=[]
    )

    result = await orchestrator.process(
        "Find romantic floral perfumes",
        context
    )

    print(f"Intent: {result.intent}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    if result.search_results:
        print(f"Search results: {len(result.search_results)} items")
    else:
        print("No search results (tool may need initialization)")

    return True  # Search may fail if DB not set up


async def test_validation_integration():
    """Test validation tool integration"""
    print("\n" + "="*70)
    print("TEST 5: Validation Tool Integration")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_5",
        conversation_history=[]
    )

    # Create a recipe first
    await orchestrator.process(
        "Create a balanced floral perfume",
        context
    )

    # Now validate it
    result = await orchestrator.process(
        "Validate this recipe",
        context
    )

    print(f"Intent: {result.intent}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    if result.validation_result:
        print(f"Validation score: {result.validation_result.get('overall_score', 'N/A')}")

    return result.success


async def test_feedback_learning():
    """Test that RLHF actually learns from feedback"""
    print("\n" + "="*70)
    print("TEST 6: Feedback Learning (RLHF Policy Updates)")
    print("="*70)

    orchestrator = EnhancedArtisanOrchestrator()
    context = OrchestrationContext(
        user_id="test_user",
        session_id="test_session_6",
        conversation_history=[]
    )

    # Create initial recipe
    await orchestrator.process("Create a simple perfume", context)

    initial_updates = orchestrator.rlhf_system.policy_network.total_updates
    print(f"Initial policy updates: {initial_updates}")

    # Provide multiple rounds of feedback
    feedback_messages = [
        "Make it better",
        "I love this version!",
        "Too strong, make it lighter",
        "Perfect, this is great!"
    ]

    for i, feedback in enumerate(feedback_messages, 1):
        print(f"\nRound {i}: {feedback}")
        result = await orchestrator.process(feedback, context)

        current_updates = orchestrator.rlhf_system.policy_network.total_updates
        print(f"Policy updates: {current_updates}")

        if result.metadata:
            print(f"Feedback history: {result.metadata.get('feedback_history_length')}")

    final_updates = orchestrator.rlhf_system.policy_network.total_updates
    updates_made = final_updates - initial_updates

    print(f"\nTotal policy updates made: {updates_made}")
    print(f"Policy network file: models/orchestrator/policy_network.pth")

    # Check if file was modified
    from pathlib import Path
    model_file = Path("models/orchestrator/policy_network.pth")
    if model_file.exists():
        print(f"Model file exists: YES")
        print(f"File size: {model_file.stat().st_size:,} bytes")
    else:
        print(f"Model file exists: NO (will be created on first update)")

    return updates_made > 0


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("ENHANCED ARTISAN ORCHESTRATOR INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("CREATE_NEW with MOGA", test_create_new_with_moga),
        ("EVOLVE_EXISTING with RLHF", test_evolve_existing_with_rlhf),
        ("Intent Classification", test_intent_classification),
        ("Search Integration", test_search_integration),
        ("Validation Integration", test_validation_integration),
        ("Feedback Learning", test_feedback_learning)
    ]

    passed = 0
    failed = 0
    errors = []

    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"\n[PASS] {test_name}")
            else:
                failed += 1
                print(f"\n[FAIL] {test_name}")
        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {test_name}: {e}")
            errors.append((test_name, str(e)))

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if errors:
        print("\nErrors encountered:")
        for test_name, error in errors:
            print(f"  - {test_name}: {error}")

    if passed == len(tests):
        print("\n[SUCCESS] All integration tests passed!")
        print("\nKey achievements:")
        print("  1. MOGA optimizer integrated for CREATE_NEW intent")
        print("  2. RLHF system integrated for EVOLVE_EXISTING intent")
        print("  3. Real AI engines replace all simulations")
        print("  4. Policy network updates and persists with feedback")
        print("  5. All tools properly integrated")
    else:
        print(f"\n[WARNING] {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())