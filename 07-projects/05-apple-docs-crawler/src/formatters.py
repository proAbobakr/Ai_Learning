"""
Output Formatters
Creates human-readable and LLM-optimized outputs
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HumanReadableFormatter:
    """Creates human-readable markdown documentation"""

    def __init__(self, output_dir: Path, include_diagrams: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_diagrams = include_diagrams

    def format_page(self, page, related_pages: List = None, complexity: float = 0) -> str:
        """Format a single documentation page as markdown"""
        md = []

        # Title
        md.append(f"# {page.title}\n")

        # Metadata
        md.append("## Metadata\n")
        md.append(f"- **Category**: {page.category}")
        md.append(f"- **URL**: [{page.url}]({page.url})")
        md.append(f"- **Complexity Score**: {complexity:.1f}/100")
        if page.metadata.get('framework'):
            md.append(f"- **Framework**: {page.metadata['framework']}")
        md.append(f"- **Code Examples**: {len(page.code_examples)}")
        md.append("")

        # Table of Contents
        if page.sections:
            md.append("## Table of Contents\n")
            for i, section in enumerate(page.sections, 1):
                md.append(f"{i}. [{section['title']}](#{section['title'].lower().replace(' ', '-')})")
            md.append("")

        # Main Content
        md.append("## Overview\n")
        # Split content into paragraphs
        paragraphs = page.content.split('\n\n')
        for para in paragraphs[:5]:  # First 5 paragraphs
            if para.strip():
                md.append(para.strip())
                md.append("")

        # Sections
        if page.sections:
            md.append("## Detailed Documentation\n")
            for section in page.sections:
                md.append(f"### {section['title']}\n")
                md.append(section['content'])
                md.append("")

        # Code Examples
        if page.code_examples:
            md.append("## Code Examples\n")
            for i, example in enumerate(page.code_examples, 1):
                md.append(f"### Example {i}\n")
                md.append("```swift")
                md.append(example)
                md.append("```\n")

        # Related Documentation
        if related_pages:
            md.append("## Related Documentation\n")
            for related in related_pages:
                md.append(f"- [{related.title}]({related.url})")
            md.append("")

        # Related Links
        if page.related_links:
            md.append("## See Also\n")
            for link in page.related_links[:10]:
                md.append(f"- [{link}]({link})")
            md.append("")

        # Footer
        md.append("---")
        md.append(f"\n*Last updated: {page.metadata.get('scraped_at', 'Unknown')}*")

        return "\n".join(md)

    def create_category_index(self, category_name: str, pages: List, stats) -> str:
        """Create an index page for a category"""
        md = []

        md.append(f"# {category_name} Documentation\n")

        # Statistics
        md.append("## Statistics\n")
        md.append(f"- **Total Pages**: {stats.total_pages}")
        md.append(f"- **Code Examples**: {stats.total_code_examples}")
        md.append(f"- **Average Content Length**: {stats.avg_content_length:.0f} characters")
        md.append(f"- **Complexity Score**: {stats.complexity_score:.1f}/100")
        md.append("")

        # Top Topics
        if stats.top_topics:
            md.append("## Top Topics\n")
            for topic, count in stats.top_topics:
                md.append(f"- **{topic}**: {count} documents")
            md.append("")

        # Frameworks
        if stats.frameworks:
            md.append("## Related Frameworks\n")
            for framework in sorted(stats.frameworks):
                md.append(f"- {framework}")
            md.append("")

        # Documentation List
        md.append("## Documentation Pages\n")
        for page in pages:
            filename = self._get_filename(page)
            md.append(f"- [{page.title}](./{filename})")

        return "\n".join(md)

    def create_master_index(self, taxonomy: Dict, diagram_files: List[str] = None) -> str:
        """Create master index for all documentation"""
        md = []

        md.append("# Apple Developer Documentation - Complete Index\n")
        md.append(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Global Statistics
        global_stats = taxonomy['global_stats']
        md.append("## Global Statistics\n")
        md.append(f"- **Total Pages**: {global_stats['total_pages']}")
        md.append(f"- **Total Code Examples**: {global_stats['total_code_examples']}")
        md.append(f"- **Unique Frameworks**: {len(global_stats['frameworks'])}")
        md.append("")

        # Visualizations
        if self.include_diagrams and diagram_files:
            md.append("## Visualizations\n")
            for diagram in diagram_files:
                md.append(f"![{diagram}](./diagrams/{diagram})")
            md.append("")

        # Top Topics
        if global_stats.get('topics'):
            md.append("## Top Topics Across All Documentation\n")
            for topic, count in global_stats['topics'][:10]:
                md.append(f"- **{topic}**: {count} documents")
            md.append("")

        # Categories
        md.append("## Categories\n")
        for cat_name, stats in taxonomy['categories'].items():
            filename = f"{cat_name.lower().replace(' ', '_')}_index.md"
            md.append(f"### [{cat_name}](./{cat_name}/{filename})")
            md.append(f"- Pages: {stats.total_pages}")
            md.append(f"- Code Examples: {stats.total_code_examples}")
            md.append(f"- Complexity: {stats.complexity_score:.1f}/100")
            md.append("")

        return "\n".join(md)

    def _get_filename(self, page) -> str:
        """Generate a filename for a page"""
        # Extract last part of URL as filename
        title = page.title.lower().replace(' ', '_').replace('/', '_')
        # Clean up any special characters
        title = ''.join(c for c in title if c.isalnum() or c in ('_', '-'))
        return f"{title[:50]}.md"

    def save_category(self, category_name: str, pages: List, stats, related_map: Dict = None):
        """Save all pages for a category"""
        category_dir = self.output_dir / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        index_content = self.create_category_index(category_name, pages, stats)
        index_path = category_dir / f"{category_name.lower().replace(' ', '_')}_index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

        # Save individual pages
        for page in pages:
            related = related_map.get(page.url, []) if related_map else None
            complexity = getattr(stats, 'complexity_score', 0) / len(pages) if pages else 0

            content = self.format_page(page, related, complexity)
            filename = self._get_filename(page)
            filepath = category_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        logger.info(f"Saved {len(pages)} pages to {category_dir}")


class LLMFormatter:
    """Creates LLM-optimized format for feeding to local LLMs"""

    def __init__(self, output_dir: Path, chunk_size: int = 2000):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

    def format_for_llm(self, page, include_metadata: bool = True) -> Dict:
        """Format a page for LLM consumption"""
        doc = {
            'id': self._generate_id(page.url),
            'type': 'documentation',
            'source': 'Apple Developer Documentation',
            'category': page.category,
            'title': page.title,
            'url': page.url,
        }

        if include_metadata:
            doc['metadata'] = {
                'framework': page.metadata.get('framework', ''),
                'scraped_at': page.metadata.get('scraped_at', ''),
                'num_code_examples': len(page.code_examples),
                'num_sections': len(page.sections),
                'content_length': len(page.content)
            }

        # Main content
        doc['content'] = page.content

        # Structured sections
        doc['sections'] = page.sections

        # Code examples as separate field
        doc['code_examples'] = [
            {
                'id': i,
                'code': example,
                'language': 'swift'
            }
            for i, example in enumerate(page.code_examples)
        ]

        # Related links
        doc['related_urls'] = page.related_links

        return doc

    def chunk_content(self, content: str) -> List[str]:
        """Split content into chunks for LLM processing"""
        chunks = []
        current_chunk = ""

        # Split by paragraphs
        paragraphs = content.split('\n\n')

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def create_llm_dataset(self, category_name: str, pages: List) -> List[Dict]:
        """Create a complete LLM dataset for a category"""
        dataset = []

        for page in pages:
            # Format the page
            doc = self.format_for_llm(page)

            # Create chunks if content is too long
            if len(doc['content']) > self.chunk_size:
                chunks = self.chunk_content(doc['content'])

                for i, chunk in enumerate(chunks):
                    chunk_doc = doc.copy()
                    chunk_doc['id'] = f"{doc['id']}_chunk_{i}"
                    chunk_doc['content'] = chunk
                    chunk_doc['metadata']['chunk_id'] = i
                    chunk_doc['metadata']['total_chunks'] = len(chunks)
                    dataset.append(chunk_doc)
            else:
                dataset.append(doc)

        return dataset

    def save_category(self, category_name: str, pages: List):
        """Save category in LLM-optimized format"""
        dataset = self.create_llm_dataset(category_name, pages)

        # Save as JSONL (one JSON object per line)
        filename = self.output_dir / f"{category_name.lower().replace(' ', '_')}.jsonl"

        with open(filename, 'w', encoding='utf-8') as f:
            for doc in dataset:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(dataset)} LLM-ready documents to {filename}")

        # Also save as regular JSON for easier inspection
        json_filename = self.output_dir / f"{category_name.lower().replace(' ', '_')}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    def _generate_id(self, url: str) -> str:
        """Generate a unique ID from URL"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def create_metadata_file(self, taxonomy: Dict):
        """Create metadata file for the entire dataset"""
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'source': 'Apple Developer Documentation',
            'total_categories': len(taxonomy['categories']),
            'total_documents': taxonomy['global_stats']['total_pages'],
            'total_code_examples': taxonomy['global_stats']['total_code_examples'],
            'categories': {
                name: {
                    'total_pages': stats.total_pages,
                    'total_code_examples': stats.total_code_examples,
                    'avg_content_length': stats.avg_content_length,
                    'complexity_score': stats.complexity_score,
                    'frameworks': list(stats.frameworks)
                }
                for name, stats in taxonomy['categories'].items()
            },
            'usage': {
                'format': 'JSONL (JSON Lines)',
                'encoding': 'UTF-8',
                'recommended_chunk_size': self.chunk_size,
                'fields': {
                    'id': 'Unique identifier',
                    'title': 'Document title',
                    'category': 'Documentation category',
                    'content': 'Main text content',
                    'code_examples': 'List of code examples',
                    'sections': 'Structured sections',
                    'metadata': 'Additional metadata'
                }
            }
        }

        filename = self.output_dir / 'dataset_metadata.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved dataset metadata to {filename}")


if __name__ == "__main__":
    # Test formatters
    from crawler import DocumentationPage

    sample_page = DocumentationPage(
        url="https://developer.apple.com/documentation/swiftui/view",
        title="View Protocol",
        category="SwiftUI",
        content="The View protocol represents a piece of user interface." * 50,
        code_examples=["struct ContentView: View { var body: some View { Text('Hello') } }"],
        sections=[{"title": "Overview", "content": "Views are the building blocks"}],
        related_links=["https://developer.apple.com/documentation/swiftui/text"],
        metadata={'scraped_at': '2024-01-01'}
    )

    # Test human-readable
    hr_formatter = HumanReadableFormatter(Path('./test_output/human'))
    content = hr_formatter.format_page(sample_page, complexity=45.5)
    print("Human-readable preview:")
    print(content[:500])

    # Test LLM format
    llm_formatter = LLMFormatter(Path('./test_output/llm'))
    llm_doc = llm_formatter.format_for_llm(sample_page)
    print("\n\nLLM format preview:")
    print(json.dumps(llm_doc, indent=2)[:500])
