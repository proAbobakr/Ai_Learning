"""
LLM-Ready Formatter
Creates formatted content optimized for local LLM consumption
"""

import logging
from typing import List, Dict
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFormatter:
    """Formats content for optimal LLM consumption"""

    def __init__(self, output_dir: str = "output/llm_format", max_chunk_size: int = 4000):
        """Initialize LLM formatter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_chunk_size = max_chunk_size

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for LLM"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

    def format_page_for_llm(self, content: Dict) -> str:
        """Format a single page optimally for LLM"""
        sections = []

        # Header with metadata
        sections.append("=" * 80)
        sections.append(f"DOCUMENT: {content.get('title', 'Untitled')}")
        sections.append(f"SOURCE: {content.get('url', '')}")
        sections.append(f"CATEGORY: {content.get('category', 'Unknown')}")
        sections.append("=" * 80)
        sections.append("")

        # Content structure
        sections.append("CONTENT STRUCTURE:")
        headings = content.get('headings', [])
        for heading in headings:
            level = int(heading['level'][1])
            indent = "  " * (level - 1)
            sections.append(f"{indent}- {heading['text']}")
        sections.append("")

        # Main content
        sections.append("MAIN CONTENT:")
        paragraphs = content.get('paragraphs', [])
        for paragraph in paragraphs:
            cleaned = self.clean_text(paragraph)
            if cleaned:
                sections.append(cleaned)
                sections.append("")

        # Code examples
        code_examples = content.get('code_examples', [])
        if code_examples:
            sections.append("CODE EXAMPLES:")
            for i, example in enumerate(code_examples, 1):
                language = example.get('language', 'dart')
                code = example.get('code', '')
                sections.append(f"\nExample {i} ({language}):")
                sections.append("```")
                sections.append(code)
                sections.append("```")
                sections.append("")

        # Tables
        tables = content.get('tables', [])
        if tables:
            sections.append("TABLES:")
            for i, table in enumerate(tables, 1):
                sections.append(f"\nTable {i}:")
                for row in table:
                    sections.append(" | ".join(row))
                sections.append("")

        # Key concepts extraction
        sections.append("KEY CONCEPTS:")
        # Extract important terms from title and headings
        key_terms = set()
        if content.get('title'):
            key_terms.update(content['title'].split())
        for heading in headings[:5]:
            key_terms.update(heading['text'].split())

        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        key_terms = {term for term in key_terms if term.lower() not in stop_words and len(term) > 3}

        sections.append(", ".join(sorted(key_terms)[:20]))
        sections.append("")

        sections.append("=" * 80)
        sections.append("")

        return "\n".join(sections)

    def chunk_content(self, text: str, max_size: int) -> List[str]:
        """Split content into chunks suitable for LLM context window"""
        chunks = []
        current_chunk = []
        current_size = 0

        lines = text.split('\n')

        for line in lines:
            line_size = len(line)

            if current_size + line_size > max_size and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def generate_category_file(self, category: str, items: List[Dict], output_path: Path):
        """Generate LLM-optimized file for a category"""
        all_content = []

        # Category header
        all_content.append("#" * 80)
        all_content.append(f"# FLUTTER DOCUMENTATION CATEGORY: {category.upper()}")
        all_content.append(f"# Total Documents: {len(items)}")
        all_content.append("#" * 80)
        all_content.append("")

        # Process each item
        for item in items:
            formatted = self.format_page_for_llm(item)
            all_content.append(formatted)

        # Combine all content
        full_text = "\n".join(all_content)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        # Also create chunked version if content is large
        if len(full_text) > self.max_chunk_size:
            chunks = self.chunk_content(full_text, self.max_chunk_size)
            chunk_dir = output_path.parent / f"{output_path.stem}_chunks"
            chunk_dir.mkdir(exist_ok=True)

            for i, chunk in enumerate(chunks, 1):
                chunk_path = chunk_dir / f"chunk_{i:03d}.txt"
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)

            logger.info(f"Created {len(chunks)} chunks for {category}")

        logger.info(f"Generated LLM format for {category}: {output_path}")

    def generate_metadata(self, categorized_content: Dict[str, List[Dict]]):
        """Generate metadata file for LLM context"""
        metadata = {
            "source": "Flutter Documentation (docs.flutter.dev)",
            "total_pages": sum(len(items) for items in categorized_content.values()),
            "categories": {},
            "recommended_usage": {
                "context_window": f"Each category file is optimized for context windows. Large categories are chunked into {self.max_chunk_size} character segments.",
                "format": "Plain text with clear section markers for easy parsing",
                "metadata_markers": "Documents use = and # markers for easy identification"
            }
        }

        for category, items in categorized_content.items():
            metadata["categories"][category] = {
                "document_count": len(items),
                "total_code_examples": sum(len(item.get('code_examples', [])) for item in items),
                "file": f"{self._sanitize_filename(category)}.txt"
            }

        # Write metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Generated metadata: {metadata_path}")

    def generate_combined_file(self, categorized_content: Dict[str, List[Dict]]):
        """Generate a single combined file with all content"""
        combined_path = self.output_dir / "flutter_docs_complete.txt"

        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("#" * 80 + "\n")
            f.write("# COMPLETE FLUTTER DOCUMENTATION\n")
            f.write("# Optimized for Local LLM Training and Context\n")
            f.write("#" * 80 + "\n\n")

            for category, items in sorted(categorized_content.items()):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"CATEGORY: {category.upper()}\n")
                f.write(f"{'=' * 80}\n\n")

                for item in items:
                    formatted = self.format_page_for_llm(item)
                    f.write(formatted)
                    f.write("\n\n")

        logger.info(f"Generated combined file: {combined_path}")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize category name for filename"""
        return name.lower().replace(' ', '_').replace('/', '_')

    def format_all(self, categorized_content: Dict[str, List[Dict]]):
        """Generate all LLM-optimized output files"""
        logger.info("Generating LLM-optimized documentation...")

        # Generate category files
        for category, items in categorized_content.items():
            if items:
                filename = self._sanitize_filename(category)
                output_path = self.output_dir / f"{filename}.txt"
                self.generate_category_file(category, items, output_path)

        # Generate metadata
        self.generate_metadata(categorized_content)

        # Generate combined file
        self.generate_combined_file(categorized_content)

        logger.info("LLM-optimized documentation complete!")
