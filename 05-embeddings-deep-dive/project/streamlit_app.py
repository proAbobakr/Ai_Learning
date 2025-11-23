"""
Smart Document Search Engine - Streamlit Web Interface

A user-friendly web interface for the search engine.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Smart Document Search",
    page_icon="",
    layout="wide"
)


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def search_documents(
    query: str,
    top_k: int = 10,
    use_hybrid: bool = True,
    use_rerank: bool = True,
    alpha: float = 0.7
) -> Optional[dict]:
    """Search documents via API."""
    try:
        params = {
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "alpha": alpha
        }
        response = requests.get(f"{API_URL}/search", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def add_document(content: str, title: str = "", source: str = "", metadata: dict = None) -> bool:
    """Add document via API."""
    try:
        data = {
            "content": content,
            "title": title or None,
            "source": source or None,
            "doc_type": "text",
            "metadata": metadata or {}
        }
        response = requests.post(f"{API_URL}/documents", json=data)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error adding document: {e}")
        return False


def get_stats() -> Optional[dict]:
    """Get statistics from API."""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def main():
    """Main Streamlit app."""

    # Header
    st.title(" Smart Document Search Engine")
    st.markdown("*Semantic search powered by embeddings*")

    # Check API status
    api_healthy = check_api_health()

    if not api_healthy:
        st.error("""
         **API is not running!**

        Please start the API server first:
        ```bash
        python main.py
        ```
        """)
        return

    # Sidebar
    with st.sidebar:
        st.header(" Settings")

        # Search settings
        st.subheader("Search Options")
        top_k = st.slider("Number of results", 1, 50, 10)
        use_hybrid = st.checkbox("Use hybrid search", value=True)
        use_rerank = st.checkbox("Use re-ranking", value=True)

        if use_hybrid:
            alpha = st.slider(
                "Hybrid alpha (0=keyword, 1=vector)",
                0.0, 1.0, 0.7
            )
        else:
            alpha = 1.0

        # Stats
        st.subheader(" Statistics")
        stats = get_stats()
        if stats:
            st.metric("Documents", stats.get("total_documents", 0))
            st.metric("Model", stats.get("embedding_model", "N/A"))
            st.metric("Dimensions", stats.get("embedding_dimension", 0))

    # Main content tabs
    tab1, tab2, tab3 = st.tabs([" Search", " Add Documents", " About"])

    # Search Tab
    with tab1:
        st.header("Search Documents")

        # Search input
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., What is machine learning?"
        )

        search_clicked = st.button(" Search", type="primary")

        if search_clicked and query:
            with st.spinner("Searching..."):
                results = search_documents(
                    query=query,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                    alpha=alpha
                )

            if results:
                # Search info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Results", results["total_results"])
                with col2:
                    st.metric("Search Time", f"{results['search_time_ms']:.2f} ms")
                with col3:
                    st.metric("Search Type", results["search_type"])

                st.divider()

                # Results
                if results["results"]:
                    for i, result in enumerate(results["results"], 1):
                        with st.container():
                            # Result header
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                title = result.get("title") or "Untitled"
                                st.markdown(f"**{i}. {title}**")
                            with col2:
                                st.markdown(f"**Score: {result['score']:.4f}**")

                            # Content
                            content = result["content"]
                            if len(content) > 500:
                                content = content[:500] + "..."
                            st.markdown(content)

                            # Scores breakdown
                            scores = []
                            if result.get("vector_score") is not None:
                                scores.append(f"Vector: {result['vector_score']:.4f}")
                            if result.get("keyword_score") is not None:
                                scores.append(f"Keyword: {result['keyword_score']:.4f}")
                            if result.get("rerank_score") is not None:
                                scores.append(f"Rerank: {result['rerank_score']:.4f}")

                            if scores:
                                st.caption(" | ".join(scores))

                            # Metadata
                            with st.expander("Metadata"):
                                st.json(result.get("metadata", {}))

                            st.divider()
                else:
                    st.info("No results found. Try a different query.")

    # Add Documents Tab
    with tab2:
        st.header("Add Documents")

        with st.form("add_document_form"):
            title = st.text_input("Title (optional)")
            source = st.text_input("Source (optional)", placeholder="e.g., tutorial.md")
            content = st.text_area(
                "Document Content",
                height=300,
                placeholder="Paste your document content here..."
            )

            # Custom metadata
            with st.expander("Custom Metadata (optional)"):
                meta_key = st.text_input("Key", key="meta_key")
                meta_value = st.text_input("Value", key="meta_value")

            submitted = st.form_submit_button(" Add Document", type="primary")

            if submitted:
                if not content:
                    st.error("Please enter document content")
                else:
                    metadata = {}
                    if meta_key and meta_value:
                        metadata[meta_key] = meta_value

                    with st.spinner("Adding document..."):
                        success = add_document(content, title, source, metadata)

                    if success:
                        st.success(" Document added successfully!")
                        st.balloons()
                    else:
                        st.error("Failed to add document")

        # Bulk add section
        st.subheader("Bulk Add")
        st.markdown("Add multiple documents by pasting JSON:")

        bulk_json = st.text_area(
            "JSON Array of Documents",
            height=200,
            placeholder='[{"content": "First document...", "title": "Doc 1"}, {"content": "Second document...", "title": "Doc 2"}]'
        )

        if st.button(" Add Bulk Documents"):
            try:
                docs = json.loads(bulk_json)
                if not isinstance(docs, list):
                    st.error("JSON must be an array of documents")
                else:
                    response = requests.post(f"{API_URL}/documents/batch", json=docs)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Added {result['total_chunks']} chunks from {result['documents_processed']} documents!")
                    else:
                        st.error(f"Failed: {response.text}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # About Tab
    with tab3:
        st.header("About")

        st.markdown("""
        ## Smart Document Search Engine

        This is a semantic search engine that understands the **meaning** of your queries,
        not just keywords.

        ### Features

        -  **Semantic Search**: Find documents by meaning
        -  **Hybrid Search**: Combines vector and keyword search
        -  **Re-ranking**: Uses cross-encoder for better results
        -  **Document Chunking**: Handles long documents

        ### How It Works

        1. **Documents are chunked** into smaller pieces
        2. **Embeddings are generated** for each chunk
        3. **Vector database** stores and indexes embeddings
        4. **Search** finds similar vectors to your query
        5. **Re-ranking** improves result order

        ### Technology Stack

        - **FastAPI**: Backend API
        - **Streamlit**: This web interface
        - **Sentence Transformers**: Embedding models
        - **ChromaDB**: Vector database
        - **BM25**: Keyword search
        - **Cross-Encoder**: Re-ranking

        ### API Endpoints

        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | `/search` | GET/POST | Search documents |
        | `/documents` | POST | Add document |
        | `/documents/{id}` | DELETE | Delete document |
        | `/health` | GET | Health check |
        | `/stats` | GET | Statistics |

        ### Source Code

        This project is part of the **LLM Embeddings: Zero to Hero** tutorial.
        """)

        # Sample documents to add
        st.subheader(" Quick Start: Add Sample Documents")

        sample_docs = [
            {
                "title": "Introduction to Python",
                "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."
            },
            {
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
            },
            {
                "title": "Deep Learning Neural Networks",
                "content": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain, allowing them to learn from large amounts of data."
            },
            {
                "title": "Natural Language Processing",
                "content": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with statistical, machine learning, and deep learning models."
            },
            {
                "title": "Vector Databases",
                "content": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for semantic search, recommendation systems, and other AI applications that rely on similarity search."
            }
        ]

        if st.button(" Add Sample Documents"):
            with st.spinner("Adding sample documents..."):
                docs_to_add = [
                    {"content": d["content"], "title": d["title"], "doc_type": "text", "metadata": {}}
                    for d in sample_docs
                ]
                response = requests.post(f"{API_URL}/documents/batch", json=docs_to_add)
                if response.status_code == 200:
                    st.success("Sample documents added!")
                    st.rerun()
                else:
                    st.error("Failed to add sample documents")


if __name__ == "__main__":
    main()
