"""
Solution to Prerequisites Exercise
Simple Document Store Implementation
"""

from typing import List, Dict, Optional


class SimpleDocStore:
    """A simple in-memory document store with basic search"""

    def __init__(self):
        self.docs: Dict[str, str] = {}

    def add_document(self, doc_id: str, text: str):
        """Add a document to the store"""
        self.docs[doc_id] = text
        print(f"Added document '{doc_id}'")

    def simple_search(self, query: str) -> List[str]:
        """Find documents containing the query (case-insensitive)"""
        query_lower = query.lower()
        results = []

        for doc_id, text in self.docs.items():
            if query_lower in text.lower():
                results.append(doc_id)

        return results

    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve a document by ID"""
        return self.docs.get(doc_id)

    def __len__(self):
        """Return number of documents"""
        return len(self.docs)


def main():
    """Test the document store"""
    # Create store
    store = SimpleDocStore()

    # Add documents
    store.add_document("doc1", "Python is great for AI")
    store.add_document("doc2", "Machine learning uses data")
    store.add_document("doc3", "Python programming is fun")

    # Search for documents
    print("\n--- Search Results ---")
    results = store.simple_search("python")
    print(f"Search 'python': Found {len(results)} documents: {results}")

    results = store.simple_search("machine")
    print(f"Search 'machine': Found {len(results)} documents: {results}")

    results = store.simple_search("data")
    print(f"Search 'data': Found {len(results)} documents: {results}")

    # Retrieve specific document
    print("\n--- Get Specific Document ---")
    doc = store.get_document("doc1")
    print(f"doc1: {doc}")

    # Stats
    print(f"\nTotal documents in store: {len(store)}")


if __name__ == "__main__":
    main()
