"""
Load database data into vector store for testing
"""

import asyncio
import sqlite3
import sys
import os
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fragrance_ai.services.search_service import SearchService

async def load_database_to_vector_store():
    """Load fragrance data from SQLite database to vector store"""
    print("=" * 60)
    print("Loading Database Data to Vector Store")
    print("=" * 60)

    try:
        # Initialize search service
        search_service = SearchService()
        await search_service.initialize()
        print("1. SearchService initialized successfully")

        # Connect to SQLite database
        conn = sqlite3.connect('fragrance_ai.db')

        # Get fragrance notes data
        fragrance_data = conn.execute("""
            SELECT id, name, name_korean, note_type, fragrance_family,
                   description, description_korean, intensity, longevity
            FROM fragrance_notes
            LIMIT 50
        """).fetchall()

        print(f"2. Found {len(fragrance_data)} fragrance notes in database")

        # Prepare documents for vector store (fix metadata format)
        documents = []
        for i, note in enumerate(fragrance_data):
            note_id, name, korean_name, note_type, family, desc, desc_korean, intensity, longevity = note

            # Create document text for embedding
            doc_text = f"{name}"
            if korean_name:
                doc_text += f" {korean_name}"
            if family:
                doc_text += f" {family}"
            if note_type:
                doc_text += f" {note_type}"
            if desc:
                doc_text += f" {desc}"
            if desc_korean:
                doc_text += f" {desc_korean}"

            # Convert nested data to JSON strings for ChromaDB compatibility
            documents.append({
                "id": f"note_{i}",
                "document": doc_text,
                "metadata": {
                    "name": name or "Unknown",
                    "korean_name": korean_name or "",
                    "note_type": note_type or "unknown",
                    "fragrance_family": family or "unknown",
                    "intensity": float(intensity) if intensity else 5.0,
                    "longevity": float(longevity) if longevity else 5.0,
                    "source": "database",
                    "notes_json": json.dumps({"note_type": note_type, "name": name})  # Flatten complex data
                }
            })

        conn.close()

        # Add to vector store
        print("3. Adding documents to vector store...")
        try:
            result = await search_service.add_fragrance_data("fragrance_notes", documents)
            print(f"SUCCESS: Added {result['items_added']} documents to vector store")
        except Exception as e:
            print(f"FAILED: {e}")
            # Try with simpler metadata
            simple_documents = []
            for doc in documents[:10]:  # Try with fewer documents first
                simple_doc = {
                    "id": doc["id"],
                    "document": doc["document"],
                    "metadata": {
                        "name": doc["metadata"]["name"],
                        "fragrance_family": doc["metadata"]["fragrance_family"],
                        "source": "database"
                    }
                }
                simple_documents.append(simple_doc)

            print("Trying with simplified metadata...")
            result = await search_service.add_fragrance_data("fragrance_notes", simple_documents)
            print(f"SUCCESS: Added {result['items_added']} documents with simple metadata")

        # Test search
        print("4. Testing search...")
        search_results = await search_service.semantic_search(
            query="로맨틱한 장미 향수",
            collection_names=["fragrance_notes"],
            top_k=5
        )

        print(f"SUCCESS: Search returned {len(search_results['results'])} results")
        for i, result in enumerate(search_results['results'][:3], 1):
            metadata = result.get('metadata', {})
            print(f"   {i}. {metadata.get('name', 'Unknown')} ({metadata.get('fragrance_family', 'Unknown')}) - Score: {result.get('similarity_score', 0):.3f}")

        print("\n" + "=" * 60)
        print("SUCCESS: Vector store loaded and tested!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(load_database_to_vector_store())
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")