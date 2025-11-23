#!/usr/bin/env python3
"""
Example Usage Script - Smart Document Search Engine

This script demonstrates how to use the search engine programmatically.
Run the API first: python main.py
Then run this: python example_usage.py
"""

import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"


def check_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print(" API is healthy!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(" API returned error")
            return False
    except requests.exceptions.ConnectionError:
        print(" Cannot connect to API. Make sure it's running!")
        print("   Run: python main.py")
        return False


def load_sample_documents():
    """Load sample documents from JSON file."""
    sample_path = Path(__file__).parent / "data" / "sample_docs" / "sample_documents.json"

    if not sample_path.exists():
        print(f" Sample documents not found at {sample_path}")
        return []

    with open(sample_path) as f:
        return json.load(f)


def add_documents(documents):
    """Add documents to the search engine."""
    print(f"\n Adding {len(documents)} documents...")

    response = requests.post(
        f"{API_URL}/documents/batch",
        json=[{
            "content": doc["content"],
            "title": doc["title"],
            "doc_type": "text",
            "metadata": doc.get("metadata", {})
        } for doc in documents]
    )

    if response.status_code == 200:
        result = response.json()
        print(f" Added {result['total_chunks']} chunks from {result['documents_processed']} documents")
        return True
    else:
        print(f" Error: {response.text}")
        return False


def search(query, top_k=5, use_hybrid=True, use_rerank=True):
    """Search for documents."""
    print(f"\n Searching: '{query}'")
    print("-" * 50)

    response = requests.get(
        f"{API_URL}/search",
        params={
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank
        }
    )

    if response.status_code == 200:
        result = response.json()

        print(f"Found {result['total_results']} results in {result['search_time_ms']:.2f}ms")
        print(f"Search type: {result['search_type']}")
        print()

        for i, r in enumerate(result['results'], 1):
            print(f"{i}. [{r['score']:.4f}] {r.get('title', 'Untitled')}")
            print(f"   {r['content'][:100]}...")
            print()

        return result
    else:
        print(f" Error: {response.text}")
        return None


def compare_search_methods(query):
    """Compare different search methods."""
    print(f"\n Comparing search methods for: '{query}'")
    print("=" * 60)

    methods = [
        {"name": "Vector Only", "use_hybrid": False, "use_rerank": False},
        {"name": "Hybrid", "use_hybrid": True, "use_rerank": False},
        {"name": "Hybrid + Rerank", "use_hybrid": True, "use_rerank": True},
    ]

    for method in methods:
        print(f"\n{method['name']}:")
        print("-" * 40)

        response = requests.get(
            f"{API_URL}/search",
            params={
                "query": query,
                "top_k": 3,
                "use_hybrid": method["use_hybrid"],
                "use_rerank": method["use_rerank"]
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Time: {result['search_time_ms']:.2f}ms")

            for i, r in enumerate(result['results'], 1):
                print(f"  {i}. [{r['score']:.4f}] {r.get('title', 'Untitled')[:40]}")


def calculate_similarity(text1, text2):
    """Calculate similarity between two texts."""
    print(f"\n Calculating similarity...")

    response = requests.post(
        f"{API_URL}/similarity",
        params={"text1": text1, "text2": text2}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Text 1: '{text1[:50]}...'")
        print(f"Text 2: '{text2[:50]}...'")
        print(f"Similarity: {result['similarity']:.4f}")
        return result
    else:
        print(f" Error: {response.text}")
        return None


def get_embedding(texts):
    """Get embeddings for texts."""
    print(f"\n Getting embeddings for {len(texts)} texts...")

    response = requests.post(
        f"{API_URL}/embed",
        json=texts
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Dimension: {result['dimension']}")
        print(f"First embedding (first 5 values): {result['embeddings'][0][:5]}")
        return result
    else:
        print(f" Error: {response.text}")
        return None


def main():
    """Main demonstration."""
    print("=" * 60)
    print("   SMART DOCUMENT SEARCH ENGINE - DEMO")
    print("=" * 60)

    # Check API health
    if not check_health():
        return

    # Get current stats
    print("\n Current Statistics:")
    response = requests.get(f"{API_URL}/stats")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))

    # Load and add sample documents
    documents = load_sample_documents()
    if documents:
        add_documents(documents)

    # Perform searches
    queries = [
        "How do I learn Python programming?",
        "What is machine learning and AI?",
        "Explain neural networks",
        "How to store vector embeddings?",
        "Best practices for API design",
    ]

    print("\n" + "=" * 60)
    print("   SEARCH DEMONSTRATIONS")
    print("=" * 60)

    for query in queries:
        search(query, top_k=3)
        print()

    # Compare search methods
    compare_search_methods("deep learning neural networks")

    # Calculate similarity
    print("\n" + "=" * 60)
    print("   SIMILARITY CALCULATIONS")
    print("=" * 60)

    calculate_similarity(
        "Machine learning is great for data analysis",
        "ML helps analyze large datasets"
    )

    calculate_similarity(
        "Machine learning is great for data analysis",
        "The weather is nice today"
    )

    # Get embeddings
    print("\n" + "=" * 60)
    print("   EMBEDDING GENERATION")
    print("=" * 60)

    get_embedding(["Hello world", "Machine learning is fascinating"])

    print("\n" + "=" * 60)
    print("   DEMO COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
