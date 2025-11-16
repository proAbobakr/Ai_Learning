#!/usr/bin/env python3
"""
Example usage of Apple Documentation Crawler components
Demonstrates how to use individual components programmatically
"""

import asyncio
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crawler import AppleDocsCrawler, DocumentationPage
from categorizer import DocumentationCategorizer
from visualizer import DocumentationVisualizer
from formatters import HumanReadableFormatter, LLMFormatter


async def example_1_basic_crawl():
    """Example 1: Basic crawling of a single category"""
    print("=" * 60)
    print("Example 1: Basic Crawling")
    print("=" * 60)

    # Load config
    config_path = Path(__file__).parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create crawler
    async with AppleDocsCrawler(config) as crawler:
        # Crawl SwiftUI category (limited to 3 pages for demo)
        category = {
            'name': 'SwiftUI',
            'url_pattern': 'swiftui',
            'description': 'SwiftUI framework'
        }

        print(f"\nCrawling {category['name']} (max 3 pages)...")
        pages = await crawler.crawl_category(category, max_pages=3)

        print(f"\n✓ Crawled {len(pages)} pages:")
        for i, page in enumerate(pages, 1):
            print(f"  {i}. {page.title}")
            print(f"     - Code examples: {len(page.code_examples)}")
            print(f"     - Sections: {len(page.sections)}")

        return pages


def example_2_categorization(pages):
    """Example 2: Categorize and analyze documentation"""
    print("\n" + "=" * 60)
    print("Example 2: Categorization & Analysis")
    print("=" * 60)

    categorizer = DocumentationCategorizer()

    for page in pages[:2]:  # Analyze first 2 pages
        print(f"\nAnalyzing: {page.title}")

        # Get topics
        topics = categorizer.categorize_by_topic(page)
        print(f"  Topics: {', '.join(topics)}")

        # Get complexity
        complexity = categorizer.calculate_complexity(page)
        print(f"  Complexity: {complexity:.1f}/100")

        # Get frameworks
        frameworks = categorizer.extract_frameworks(page)
        print(f"  Frameworks: {', '.join(frameworks) if frameworks else 'None detected'}")

        # Get keywords
        keywords = categorizer.extract_keywords(page, top_n=5)
        print(f"  Top Keywords: {', '.join([word for word, _ in keywords])}")


def example_3_human_readable_output(pages):
    """Example 3: Generate human-readable markdown"""
    print("\n" + "=" * 60)
    print("Example 3: Human-Readable Output")
    print("=" * 60)

    output_dir = Path(__file__).parent / 'output' / 'examples' / 'human'
    formatter = HumanReadableFormatter(output_dir)

    # Format first page
    page = pages[0]
    markdown = formatter.format_page(page, complexity=42.5)

    print(f"\nGenerated markdown for: {page.title}")
    print("\nPreview (first 500 chars):")
    print("-" * 60)
    print(markdown[:500] + "...")
    print("-" * 60)

    # Save to file
    output_file = output_dir / 'example.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"\n✓ Saved to: {output_file}")


def example_4_llm_format(pages):
    """Example 4: Generate LLM-optimized format"""
    print("\n" + "=" * 60)
    print("Example 4: LLM-Optimized Format")
    print("=" * 60)

    output_dir = Path(__file__).parent / 'output' / 'examples' / 'llm'
    formatter = LLMFormatter(output_dir, chunk_size=1000)

    # Format pages for LLM
    llm_docs = formatter.create_llm_dataset('Example', pages)

    print(f"\n✓ Created {len(llm_docs)} LLM-ready documents")
    print(f"  (including chunks for long content)")

    # Show first document
    import json
    print("\nExample document (first 500 chars):")
    print("-" * 60)
    print(json.dumps(llm_docs[0], indent=2)[:500] + "...")
    print("-" * 60)

    # Save
    formatter.save_category('Example', pages)
    print(f"\n✓ Saved to: {output_dir}/example.jsonl")


def example_5_visualization(pages):
    """Example 5: Create visualizations"""
    print("\n" + "=" * 60)
    print("Example 5: Visualizations")
    print("=" * 60)

    from categorizer import CategoryStats

    # Create sample taxonomy
    categorizer = DocumentationCategorizer()
    stats = categorizer.analyze_category(pages, 'SwiftUI')

    taxonomy = {
        'categories': {
            'SwiftUI': stats
        },
        'global_stats': {
            'total_pages': len(pages),
            'total_code_examples': sum(len(p.code_examples) for p in pages),
            'topics': [('UI', 10), ('Animation', 5)],
            'frameworks': ['SwiftUI', 'Combine']
        }
    }

    # Create visualizer
    output_dir = Path(__file__).parent / 'output' / 'examples' / 'diagrams'
    visualizer = DocumentationVisualizer(output_dir)

    print("\nGenerating visualizations...")

    # Generate Mermaid diagram
    mermaid = visualizer.generate_mermaid_diagram(taxonomy)
    print("\nMermaid Diagram:")
    print("-" * 60)
    print(mermaid)
    print("-" * 60)

    print(f"\n✓ Visualizations would be saved to: {output_dir}")
    print("  (Skipping chart generation in example)")


async def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Apple Documentation Crawler Examples" + " " * 11 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        # Example 1: Crawl
        pages = await example_1_basic_crawl()

        if not pages:
            print("\n⚠️  No pages crawled. Check your internet connection.")
            print("   The remaining examples will use dummy data.")
            # Create dummy page for demonstration
            pages = [
                DocumentationPage(
                    url="https://example.com",
                    title="Example View",
                    category="SwiftUI",
                    content="This is example content. " * 50,
                    code_examples=["struct Example: View { }"],
                    sections=[{"title": "Overview", "content": "Example section"}],
                    related_links=[],
                    metadata={'scraped_at': '2024-01-01'}
                )
            ]

        # Example 2: Categorization
        example_2_categorization(pages)

        # Example 3: Human-readable output
        example_3_human_readable_output(pages)

        # Example 4: LLM format
        example_4_llm_format(pages)

        # Example 5: Visualizations
        example_5_visualization(pages)

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Check the output/examples/ directory")
        print("  2. Run: python src/main.py --test")
        print("  3. Read the README.md for more details")
        print()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
