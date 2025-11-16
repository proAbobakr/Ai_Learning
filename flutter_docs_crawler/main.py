#!/usr/bin/env python3
"""
Flutter Documentation Crawler - Main Orchestrator
Coordinates crawling, categorization, and formatting of Flutter documentation
"""

import argparse
import logging
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from crawler import FlutterDocsCrawler
from categorizer import ContentCategorizer
from human_readable_formatter import HumanReadableFormatter
from llm_formatter import LLMFormatter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Crawl and process Flutter documentation'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=100,
        help='Maximum number of pages to crawl (default: 100)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/crawler_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-crawl',
        action='store_true',
        help='Skip crawling and use existing raw data'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default='output/raw_data.json',
        help='Path to raw data file'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Flutter Documentation Crawler")
    logger.info("=" * 80)

    # Step 1: Crawl documentation
    if not args.skip_crawl:
        logger.info("\n[Step 1/4] Crawling Flutter documentation...")
        crawler = FlutterDocsCrawler(config_path=args.config)
        scraped_content = crawler.crawl(max_pages=args.max_pages)
        crawler.save_raw_data(args.raw_data)
    else:
        logger.info("\n[Step 1/4] Loading existing raw data...")
        try:
            with open(args.raw_data, 'r', encoding='utf-8') as f:
                scraped_content = json.load(f)
            logger.info(f"Loaded {len(scraped_content)} pages from {args.raw_data}")
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {args.raw_data}")
            logger.error("Please run without --skip-crawl first")
            return

    if not scraped_content:
        logger.error("No content was scraped. Exiting.")
        return

    # Step 2: Categorize content
    logger.info("\n[Step 2/4] Categorizing content...")
    categorizer = ContentCategorizer(config_path=args.config)
    categorized_content = categorizer.categorize_content(scraped_content)

    # Print statistics
    stats = categorizer.get_category_statistics()
    logger.info("\nCategorization Statistics:")
    logger.info(f"Total Pages: {stats['total_pages']}")
    for category, cat_stats in stats['categories'].items():
        logger.info(f"  {category}: {cat_stats['page_count']} pages, "
                   f"{cat_stats['total_code_examples']} code examples")

    # Step 3: Generate human-readable output
    logger.info("\n[Step 3/4] Generating human-readable documentation...")
    human_formatter = HumanReadableFormatter()
    human_formatter.format_all(categorized_content)

    # Step 4: Generate LLM-optimized output
    logger.info("\n[Step 4/4] Generating LLM-optimized documentation...")
    llm_formatter = LLMFormatter()
    llm_formatter.format_all(categorized_content)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nOutput locations:")
    logger.info(f"  Human-readable: output/human_readable/")
    logger.info(f"  LLM-optimized:  output/llm_format/")
    logger.info(f"  Raw data:       {args.raw_data}")
    logger.info("\nQuick links:")
    logger.info(f"  Index:          output/human_readable/INDEX.md")
    logger.info(f"  Metadata:       output/llm_format/metadata.json")
    logger.info(f"  Combined LLM:   output/llm_format/flutter_docs_complete.txt")


if __name__ == "__main__":
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nCrawling interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
