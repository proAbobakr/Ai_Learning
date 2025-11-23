"""
Chunking utilities for document processing.
"""

from typing import List, Tuple
import re


def split_by_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_paragraphs(text: str) -> List[str]:
    """Split text by paragraph breaks."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def sliding_window_chunks(
    text: str,
    window_size: int = 500,
    step_size: int = 250
) -> List[Tuple[str, int, int]]:
    """
    Create overlapping chunks using sliding window.

    Args:
        text: Text to chunk
        window_size: Size of each chunk in characters
        step_size: Step between chunks (overlap = window_size - step_size)

    Returns:
        List of (chunk_text, start_pos, end_pos) tuples
    """
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        chunk = text[start:end]

        # Try to end at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_exclaim = chunk.rfind('!')
            last_boundary = max(last_period, last_question, last_exclaim)

            if last_boundary > window_size * 0.5:  # Only if past halfway
                end = start + last_boundary + 1
                chunk = text[start:end]

        chunks.append((chunk.strip(), start, end))
        start += step_size

    return chunks


def semantic_chunks(
    text: str,
    max_chunk_size: int = 500,
    min_chunk_size: int = 100
) -> List[str]:
    """
    Create semantically coherent chunks.

    Tries to keep related content together by respecting:
    - Paragraph boundaries
    - Section headers
    - List items
    """
    # Split by major boundaries
    sections = re.split(r'\n#{1,3}\s+', text)  # Markdown headers

    chunks = []
    current_chunk = ""

    for section in sections:
        paragraphs = split_by_paragraphs(section)

        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                current_chunk += " " + para if current_chunk else para
            else:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Start new chunk
                if len(para) > max_chunk_size:
                    # Split long paragraph
                    sentences = split_by_sentences(para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                            current_chunk += " " + sent if current_chunk else sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                else:
                    current_chunk = para

    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())

    return chunks


def recursive_character_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[str]:
    """
    Recursively split text trying different separators.

    Similar to LangChain's RecursiveCharacterTextSplitter.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = []

    def split_recursive(text: str, sep_idx: int = 0) -> List[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        if sep_idx >= len(separators):
            # Last resort: hard split
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

        separator = separators[sep_idx]
        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        result = []
        current = ""

        for part in parts:
            candidate = current + separator + part if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(split_recursive(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    return [c.strip() for c in split_recursive(text) if c.strip()]
