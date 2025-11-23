"""
Document Processor - Handles document loading, parsing, and chunking.
"""

from typing import List, Dict, Any, Optional
import re
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for embedding."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def generate_id(self, content: str, index: int = 0) -> str:
        """Generate unique ID for a document chunk."""
        hash_input = f"{content[:100]}{index}{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Uses sentence boundaries when possible for more coherent chunks.
        """
        text = self.clean_text(text)

        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence is too long, split by words
            if sentence_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence by words
                words = sentence.split()
                word_chunk = []
                word_length = 0

                for word in words:
                    if word_length + len(word) + 1 > self.chunk_size and word_chunk:
                        chunks.append(' '.join(word_chunk))
                        # Keep some overlap
                        overlap_words = word_chunk[-3:] if len(word_chunk) > 3 else []
                        word_chunk = overlap_words
                        word_length = sum(len(w) + 1 for w in word_chunk)

                    word_chunk.append(word)
                    word_length += len(word) + 1

                if word_chunk:
                    current_chunk = word_chunk
                    current_length = word_length

            # If adding sentence exceeds chunk size
            elif current_length + sentence_length + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                    # Calculate overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s) + 1
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_length = overlap_length

                current_chunk.append(sentence)
                current_length += sentence_length + 1

            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def process_document(
        self,
        content: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a document into chunks with metadata.

        Args:
            content: Document content
            title: Document title
            source: Document source (filename, URL, etc.)
            doc_type: Type of document
            metadata: Additional metadata

        Returns:
            List of chunk dictionaries ready for indexing
        """
        chunks = self.chunk_text(content)
        total_chunks = len(chunks)

        if total_chunks == 0:
            logger.warning(f"Document produced no chunks: {title or 'untitled'}")
            return []

        processed_chunks = []
        base_id = self.generate_id(content)

        for idx, chunk_content in enumerate(chunks):
            chunk_id = f"{base_id}_{idx}"

            chunk_data = {
                "id": chunk_id,
                "content": chunk_content,
                "title": title,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }

            processed_chunks.append(chunk_data)

        logger.info(f"Processed document '{title}' into {total_chunks} chunks")
        return processed_chunks

    def process_pdf(self, file_path: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Process a PDF file."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            text_parts = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            content = '\n\n'.join(text_parts)

            return self.process_document(
                content=content,
                title=file_path.split('/')[-1],
                source=file_path,
                doc_type="pdf",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []

    def process_markdown(self, content: str, source: Optional[str] = None) -> List[Dict]:
        """Process markdown content."""
        import markdown
        from bs4 import BeautifulSoup

        # Convert markdown to HTML, then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        # Extract title from first heading
        title = None
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text()

        return self.process_document(
            content=text,
            title=title,
            source=source,
            doc_type="markdown"
        )
