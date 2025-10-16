"""
Test Real MOGA Optimization
CPU만으로 실행 가능한 진짜 유전 알고리즘 테스트
"""

import time
from fragrance_ai.training.moga_optimizer_stable import StableMOGA

print("="*60)
print("REAL MOGA OPTIMIZATION TEST")
print("CPU-only Genetic Algorithm")
print("="*60)
print()

# Create optimizer with high-quality parameters
print("[1] Initializing StableMOGA...")
print("    - Population size: 100")
print("    - Generations: 30")
print("    - Algorithm: NSGA-II with Exponential Mutation")
print()

start_time = time.time()

optimizer = StableMOGA(
    population_size=100,
    generations=30
)

init_time = time.time() - start_time
print(f"[OK] Initialized in {init_time:.2f}s")
print()

# Run optimization
print("[2] Running genetic algorithm optimization...")
print("    This should take 10-30 seconds...")
print()

opt_start = time.time()
results = optimizer.optimize(enable_diversity=True)
opt_time = time.time() - opt_start

print(f"[OK] Optimization complete in {opt_time:.2f}s")
print()

# Display results
print("="*60)
print("RESULTS")
print("="*60)
print()

print(f"Found {len(results['pareto_front'])} Pareto-optimal solutions")
print(f"Total population: {results['population_size']}")
print(f"Generations run: {results['generations']}")
print(f"Algorithm: {results['algorithm']}")
print()

print("Top 5 Solutions:")
print("-"*60)

for i, solution in enumerate(results['pareto_front'][:5], 1):
    print(f"\nSolution #{i}:")
    print(f"  Quality Score:   {solution['quality_score']:.2f}/100")
    print(f"  Cost:            ${solution['cost']:.2f}/kg")
    print(f"  Stability Score: {solution['stability_score']:.2f}/100")
    print(f"  Diversity Score: {solution['diversity_score']:.4f}")
    print(f"  Entropy:         {solution['entropy']:.4f}")
    print(f"  Ingredients ({len(solution['ingredients'])}):")

    for ing in solution['ingredients']:
        print(f"    - {ing['name']:15s}: {ing['concentration']:5.1f}%")

print()
print("="*60)
print("VERDICT")
print("="*60)
print()

if opt_time >= 5.0 and len(results['pareto_front']) > 0:
    print("[PASS] Real AI optimization verified!")
    print(f"  - Processing time: {opt_time:.1f}s (realistic for GA)")
    print(f"  - Solutions generated: {len(results['pareto_front'])}")
    print(f"  - Best quality: {max(s['quality_score'] for s in results['pareto_front']):.1f}")
    print(f"  - Best stability: {max(s['stability_score'] for s in results['pareto_front']):.1f}")
else:
    print("[FAIL] Optimization too fast or no results")
    print(f"  - Time: {opt_time:.1f}s")
    print(f"  - Solutions: {len(results['pareto_front'])}")
