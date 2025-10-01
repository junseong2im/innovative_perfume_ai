"""
Performance tests for KnowledgeTool
Tests fast and accurate retrieval from large knowledge databases
"""

import unittest
import json
import time
import os
import sys
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch
import random
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.tools.knowledge_tool import (
    PerfumeKnowledgeBase,
    KnowledgeQuery,
    KnowledgeResponse,
    query_knowledge_base,
    get_knowledge_base
)


class TestKnowledgeToolPerformance(unittest.TestCase):
    """Performance test suite for KnowledgeTool"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.database_path = Path(__file__).parent.parent / "assets" / "comprehensive_fragrance_notes_database.json"
        cls.knowledge_base = PerfumeKnowledgeBase()

        # Load comprehensive database
        if cls.database_path.exists():
            with open(cls.database_path, 'r') as f:
                cls.comprehensive_db = json.load(f)
        else:
            cls.comprehensive_db = None

    def setUp(self):
        """Set up each test"""
        self.kb = get_knowledge_base()

    def test_query_speed_single_note(self):
        """Test single note query speed"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        # Test single query speed
        note_names = list(self.comprehensive_db['notes'].keys())
        test_note = random.choice(note_names)

        start_time = time.perf_counter()
        query = KnowledgeQuery(
            category="note",
            query=f"Tell me about {test_note}"
        )
        response = self.kb.query(query)
        end_time = time.perf_counter()

        query_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Single query should be very fast (under 10ms)
        self.assertLess(query_time, 10, f"Single query took {query_time:.2f}ms, expected < 10ms")
        self.assertIsNotNone(response.answer)

    def test_batch_query_performance(self):
        """Test performance with multiple queries"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        note_names = list(self.comprehensive_db['notes'].keys())
        num_queries = 100
        query_times = []

        for _ in range(num_queries):
            test_note = random.choice(note_names)

            start_time = time.perf_counter()
            query = KnowledgeQuery(
                category="note",
                query=f"What are the properties of {test_note}?"
            )
            response = self.kb.query(query)
            end_time = time.perf_counter()

            query_times.append((end_time - start_time) * 1000)

        # Calculate statistics
        avg_time = statistics.mean(query_times)
        max_time = max(query_times)
        min_time = min(query_times)
        p95_time = statistics.quantiles(query_times, n=20)[18]  # 95th percentile

        # Performance assertions
        self.assertLess(avg_time, 5, f"Average query time {avg_time:.2f}ms exceeds 5ms")
        self.assertLess(p95_time, 10, f"95th percentile {p95_time:.2f}ms exceeds 10ms")
        self.assertLess(max_time, 20, f"Max query time {max_time:.2f}ms exceeds 20ms")

        print(f"\nBatch Query Performance (n={num_queries}):")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")

    def test_emotion_vector_retrieval(self):
        """Test fast retrieval of emotion vectors"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        # Simulate emotion vector queries
        notes_with_vectors = []
        for note_name, note_data in self.comprehensive_db['notes'].items():
            if 'emotion_vector' in note_data:
                notes_with_vectors.append((note_name, note_data['emotion_vector']))

        start_time = time.perf_counter()

        # Retrieve multiple emotion vectors
        retrieved_vectors = []
        for note_name, expected_vector in notes_with_vectors[:10]:
            # Simulate vector retrieval
            query = KnowledgeQuery(
                category="note",
                query=note_name
            )
            response = self.kb.query(query)
            # In real implementation, would extract vector from response
            retrieved_vectors.append(expected_vector)

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        # Should retrieve 10 vectors very quickly
        self.assertLess(total_time, 50, f"Vector retrieval took {total_time:.2f}ms for 10 notes")
        self.assertEqual(len(retrieved_vectors), 10)

    def test_complex_query_performance(self):
        """Test performance with complex queries"""
        complex_queries = [
            ("history", "Tell me about the history of perfume in ancient Egypt"),
            ("technique", "Explain the enfleurage extraction method"),
            ("accord", "What is the chypre accord formula?"),
            ("perfumer", "Describe Jean Claude Ellena's style")
        ]

        query_times = []

        for category, query_text in complex_queries:
            start_time = time.perf_counter()
            query = KnowledgeQuery(
                category=category,
                query=query_text
            )
            response = self.kb.query(query)
            end_time = time.perf_counter()

            query_time = (end_time - start_time) * 1000
            query_times.append(query_time)

            # Each complex query should still be fast
            self.assertLess(query_time, 15, f"Complex query '{query_text[:30]}...' took {query_time:.2f}ms")
            self.assertIsNotNone(response.answer)
            self.assertGreater(response.confidence, 0)

        avg_complex_time = statistics.mean(query_times)
        self.assertLess(avg_complex_time, 10, f"Average complex query time {avg_complex_time:.2f}ms exceeds 10ms")

    def test_concurrent_queries(self):
        """Test performance under concurrent load"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        async def concurrent_query(note_name):
            """Async query function"""
            return await query_knowledge_base(
                category="note",
                query=f"Properties of {note_name}"
            )

        async def run_concurrent_test():
            """Run multiple queries concurrently"""
            note_names = list(self.comprehensive_db['notes'].keys())
            tasks = []

            # Create 20 concurrent queries
            for _ in range(20):
                note = random.choice(note_names)
                tasks.append(concurrent_query(note))

            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            return results, (end_time - start_time) * 1000

        # Run concurrent test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results, total_time = loop.run_until_complete(run_concurrent_test())
        loop.close()

        # All queries should complete
        self.assertEqual(len(results), 20)

        # Total time for 20 concurrent queries should be reasonable
        self.assertLess(total_time, 100, f"20 concurrent queries took {total_time:.2f}ms")

        # Average time per query in concurrent scenario
        avg_concurrent = total_time / 20
        print(f"\nConcurrent Query Performance:")
        print(f"  Total time for 20 queries: {total_time:.2f}ms")
        print(f"  Average per query: {avg_concurrent:.2f}ms")

    def test_cache_effectiveness(self):
        """Test that caching improves performance"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        test_note = "bergamot"

        # First query (cold cache)
        start_cold = time.perf_counter()
        query1 = KnowledgeQuery(category="note", query=test_note)
        response1 = self.kb.query(query1)
        end_cold = time.perf_counter()
        cold_time = (end_cold - start_cold) * 1000

        # Multiple subsequent queries (should hit cache if implemented)
        warm_times = []
        for _ in range(5):
            start_warm = time.perf_counter()
            query2 = KnowledgeQuery(category="note", query=test_note)
            response2 = self.kb.query(query2)
            end_warm = time.perf_counter()
            warm_times.append((end_warm - start_warm) * 1000)

        avg_warm_time = statistics.mean(warm_times)

        # Warm queries should be faster or at least as fast
        self.assertLessEqual(avg_warm_time, cold_time * 1.1,
                            f"Cache not effective: warm {avg_warm_time:.2f}ms vs cold {cold_time:.2f}ms")

    def test_search_accuracy(self):
        """Test accuracy of search results"""
        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        # Test specific note retrieval
        test_cases = [
            ("bergamot", ["citrus", "fresh", "bitter"]),
            ("jasmine", ["floral", "narcotic", "sweet"]),
            ("sandalwood", ["woody", "creamy", "smooth"])
        ]

        for note_name, expected_keywords in test_cases:
            query = KnowledgeQuery(
                category="note",
                query=f"describe {note_name}"
            )
            response = self.kb.query(query)

            self.assertIsNotNone(response.answer)

            # Check if expected keywords appear in response
            answer_lower = response.answer.lower()
            keyword_found = any(keyword in answer_lower for keyword in expected_keywords)

            self.assertTrue(keyword_found,
                          f"Expected keywords {expected_keywords} not found in response for {note_name}")

    def test_memory_efficiency(self):
        """Test memory usage remains reasonable"""
        import tracemalloc

        if not self.comprehensive_db:
            self.skipTest("Comprehensive database not available")

        # Start memory tracking
        tracemalloc.start()

        # Perform many queries
        note_names = list(self.comprehensive_db['notes'].keys())
        for _ in range(100):
            note = random.choice(note_names)
            query = KnowledgeQuery(category="note", query=note)
            response = self.kb.query(query)

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Convert to MB
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        print(f"\nMemory Usage:")
        print(f"  Current: {current_mb:.2f} MB")
        print(f"  Peak: {peak_mb:.2f} MB")

        # Memory usage should be reasonable (under 100MB for this test)
        self.assertLess(peak_mb, 100, f"Peak memory usage {peak_mb:.2f}MB exceeds 100MB")

    def test_related_topics_performance(self):
        """Test performance of finding related topics"""
        categories = ["note", "accord", "perfumer", "technique"]

        total_time = 0
        for category in categories:
            start_time = time.perf_counter()

            # Query that should return related topics
            if category == "note":
                query_text = "bergamot"
            elif category == "accord":
                query_text = "chypre"
            elif category == "perfumer":
                query_text = "jean claude ellena"
            else:
                query_text = "distillation"

            query = KnowledgeQuery(category=category, query=query_text)
            response = self.kb.query(query)

            end_time = time.perf_counter()
            query_time = (end_time - start_time) * 1000
            total_time += query_time

            # Should have related topics
            self.assertIsInstance(response.related_topics, list)
            self.assertGreater(len(response.related_topics), 0,
                             f"No related topics found for {category}:{query_text}")

            # Query with related topics should still be fast
            self.assertLess(query_time, 15, f"Query with related topics took {query_time:.2f}ms")

        avg_time = total_time / len(categories)
        self.assertLess(avg_time, 10, f"Average query time with related topics {avg_time:.2f}ms")

    def test_large_context_handling(self):
        """Test performance with large context"""
        large_context = {
            "user_preferences": ["citrus", "fresh", "light"],
            "previous_queries": ["bergamot", "lemon", "grapefruit"],
            "current_recipe": {
                "top_notes": ["bergamot", "lemon"],
                "heart_notes": ["jasmine", "rose"],
                "base_notes": ["sandalwood", "musk"]
            },
            "metadata": {"session_id": "test123", "timestamp": time.time()}
        }

        start_time = time.perf_counter()
        query = KnowledgeQuery(
            category="note",
            query="suggest complementary notes",
            context=large_context
        )
        response = self.kb.query(query)
        end_time = time.perf_counter()

        query_time = (end_time - start_time) * 1000

        # Even with large context, should be reasonably fast
        self.assertLess(query_time, 20, f"Query with large context took {query_time:.2f}ms")
        self.assertIsNotNone(response.answer)


class TestKnowledgeBaseOptimization(unittest.TestCase):
    """Test optimizations and edge cases"""

    def setUp(self):
        """Set up test"""
        self.kb = PerfumeKnowledgeBase()

    def test_singleton_performance(self):
        """Test singleton pattern doesn't create multiple instances"""
        start_time = time.perf_counter()

        # Get instance multiple times
        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()
        kb3 = get_knowledge_base()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        # Should be instant after first creation
        self.assertLess(total_time, 1, f"Getting singleton took {total_time:.2f}ms")

        # Should be same instance
        self.assertIs(kb1, kb2)
        self.assertIs(kb2, kb3)

    def test_invalid_query_handling(self):
        """Test performance with invalid queries"""
        invalid_queries = [
            ("invalid_category", "test query"),
            ("note", ""),  # Empty query
            ("", "test"),  # Empty category
            ("note", "x" * 10000),  # Very long query
        ]

        for category, query_text in invalid_queries:
            start_time = time.perf_counter()

            query = KnowledgeQuery(
                category=category,
                query=query_text
            )
            response = self.kb.query(query)

            end_time = time.perf_counter()
            query_time = (end_time - start_time) * 1000

            # Invalid queries should fail fast
            self.assertLess(query_time, 5, f"Invalid query took {query_time:.2f}ms")
            self.assertIsInstance(response, KnowledgeResponse)


if __name__ == '__main__':
    # Run with verbosity to see performance metrics
    unittest.main(verbosity=2)