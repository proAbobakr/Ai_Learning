#!/usr/bin/env python3
"""
Example usage of the Flutter Documentation Crawler
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from crawler import FlutterDocsCrawler
from categorizer import ContentCategorizer
from human_readable_formatter import HumanReadableFormatter
from llm_formatter import LLMFormatter


def example_basic_crawl():
    """Example: Basic crawling"""
    print("Example 1: Basic Crawling")
    print("-" * 50)

    crawler = FlutterDocsCrawler()

    # Crawl just 10 pages for quick testing
    content = crawler.crawl(max_pages=10)

    print(f"Crawled {len(content)} pages")
    print(f"First page title: {content[0].get('title', 'N/A')}")


def example_with_categorization():
    """Example: Crawling with categorization"""
    print("\nExample 2: Crawling with Categorization")
    print("-" * 50)

    # Crawl
    crawler = FlutterDocsCrawler()
    content = crawler.crawl(max_pages=20)

    # Categorize
    categorizer = ContentCategorizer()
    categorized = categorizer.categorize_content(content)

    # Show results
    for category, items in categorized.items():
        print(f"{category}: {len(items)} pages")


def example_full_pipeline():
    """Example: Full pipeline with all formatters"""
    print("\nExample 3: Full Pipeline")
    print("-" * 50)

    # Step 1: Crawl
    print("Step 1: Crawling...")
    crawler = FlutterDocsCrawler()
    content = crawler.crawl(max_pages=15)

    # Step 2: Categorize
    print("Step 2: Categorizing...")
    categorizer = ContentCategorizer()
    categorized = categorizer.categorize_content(content)

    # Step 3: Format for humans
    print("Step 3: Generating human-readable docs...")
    human_formatter = HumanReadableFormatter(output_dir="output/example_human")
    human_formatter.format_all(categorized)

    # Step 4: Format for LLMs
    print("Step 4: Generating LLM-optimized docs...")
    llm_formatter = LLMFormatter(output_dir="output/example_llm")
    llm_formatter.format_all(categorized)

    print("\nComplete! Check output/example_human and output/example_llm")


def example_custom_categories():
    """Example: Using custom categories"""
    print("\nExample 4: Custom Categories")
    print("-" * 50)

    crawler = FlutterDocsCrawler()
    content = crawler.crawl(max_pages=10)

    # Create custom categorizer
    categorizer = ContentCategorizer()

    # Override categories
    categorizer.categories = [
        {"name": "UI Components", "keywords": ["widget", "ui", "component"]},
        {"name": "Data Handling", "keywords": ["state", "data", "provider"]},
    ]

    categorized = categorizer.categorize_content(content)

    for category, items in categorized.items():
        print(f"{category}: {len(items)} pages")


if __name__ == "__main__":
    print("Flutter Documentation Crawler - Examples")
    print("=" * 50)

    # Run examples
    try:
        example_basic_crawl()
        example_with_categorization()
        # example_full_pipeline()  # Uncomment to run full pipeline
        # example_custom_categories()  # Uncomment for custom categories

        print("\n" + "=" * 50)
        print("Examples complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
