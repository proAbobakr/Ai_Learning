#!/usr/bin/env python3
"""
Apple Documentation Crawler - Main Entry Point
Complete system for crawling, categorizing, and formatting Apple developer documentation
"""

import asyncio
import yaml
import logging
from pathlib import Path
import sys
from typing import Dict, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from crawler import AppleDocsCrawler, DocumentationPage
from categorizer import DocumentationCategorizer
from visualizer import DocumentationVisualizer
from formatters import HumanReadableFormatter, LLMFormatter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AppleDocsSystem:
    """Main orchestrator for the documentation crawler system"""

    def __init__(self, config_path: Path):
        """Initialize the system"""
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent

        # Initialize components
        self.crawler = None
        self.categorizer = DocumentationCategorizer()
        self.visualizer = DocumentationVisualizer(
            self.project_root / 'output' / 'human_readable' / 'diagrams'
        )
        self.hr_formatter = HumanReadableFormatter(
            self.project_root / 'output' / 'human_readable',
            include_diagrams=self.config['visualization']['enable_diagrams']
        )
        self.llm_formatter = LLMFormatter(
            self.project_root / 'output' / 'llm_format',
            chunk_size=self.config['output']['llm_format']['chunk_size']
        )

    def _load_config(self) -> Dict:
        """Load configuration file"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    async def crawl(self, test_mode: bool = False) -> Dict[str, List[DocumentationPage]]:
        """Crawl all categories"""
        logger.info("Starting documentation crawl...")

        categories = self.config['categories']

        if test_mode:
            logger.info("TEST MODE: Crawling limited data")
            categories = categories[:2]  # Only first 2 categories
            max_pages = 5
        else:
            max_pages = self.config['crawler'].get('max_pages_per_category', 100)

        async with AppleDocsCrawler(self.config) as crawler:
            self.crawler = crawler
            all_data = {}

            for category in categories:
                try:
                    logger.info(f"Crawling category: {category['name']}")
                    pages = await crawler.crawl_category(category, max_pages=max_pages)
                    all_data[category['name']] = pages

                    # Save raw data incrementally
                    data_dir = self.project_root / 'data'
                    crawler.save_raw_data({category['name']: pages}, data_dir)

                except Exception as e:
                    logger.error(f"Error crawling {category['name']}: {e}")
                    all_data[category['name']] = []

        logger.info(f"Crawling complete. Total categories: {len(all_data)}")
        return all_data

    def categorize_and_analyze(self, all_data: Dict[str, List[DocumentationPage]]) -> Dict:
        """Categorize and analyze all documentation"""
        logger.info("Categorizing and analyzing documentation...")

        taxonomy = self.categorizer.create_taxonomy(all_data)

        # Save taxonomy
        import json
        taxonomy_file = self.project_root / 'data' / 'taxonomy.json'
        taxonomy_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        taxonomy_copy = taxonomy.copy()
        for cat_stats in taxonomy_copy['categories'].values():
            cat_stats.frameworks = list(cat_stats.frameworks)

        with open(taxonomy_file, 'w', encoding='utf-8') as f:
            json.dump(taxonomy_copy, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Taxonomy saved to {taxonomy_file}")
        return taxonomy

    def create_visualizations(self, taxonomy: Dict):
        """Create all visualizations"""
        logger.info("Creating visualizations...")

        if self.config['visualization']['enable_charts']:
            self.visualizer.create_all_visualizations(taxonomy)

            # Generate Mermaid diagram
            mermaid = self.visualizer.generate_mermaid_diagram(taxonomy)
            mermaid_file = self.project_root / 'output' / 'human_readable' / 'diagrams' / 'structure.mmd'
            mermaid_file.parent.mkdir(parents=True, exist_ok=True)

            with open(mermaid_file, 'w', encoding='utf-8') as f:
                f.write(mermaid)

            logger.info("Visualizations created successfully")

    def generate_human_readable_output(self, all_data: Dict[str, List[DocumentationPage]], taxonomy: Dict):
        """Generate human-readable markdown files"""
        logger.info("Generating human-readable output...")

        # Find related documents for each page
        all_pages = [page for pages in all_data.values() for page in pages]

        for category_name, pages in all_data.items():
            if not pages:
                continue

            stats = taxonomy['categories'][category_name]

            # Find related docs for each page
            related_map = {}
            for page in pages:
                related = self.categorizer.find_related_docs(page, all_pages)
                related_map[page.url] = related

            # Save category
            self.hr_formatter.save_category(category_name, pages, stats, related_map)

        # Create master index
        diagram_files = [
            'category_distribution.png',
            'complexity_analysis.png',
            'code_examples.png',
            'topic_distribution.png',
            'framework_network.png'
        ] if self.config['visualization']['enable_diagrams'] else None

        master_index = self.hr_formatter.create_master_index(taxonomy, diagram_files)
        index_path = self.project_root / 'output' / 'human_readable' / 'INDEX.md'

        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(master_index)

        logger.info(f"Human-readable output saved. Main index: {index_path}")

    def generate_llm_output(self, all_data: Dict[str, List[DocumentationPage]], taxonomy: Dict):
        """Generate LLM-optimized output"""
        logger.info("Generating LLM-optimized output...")

        for category_name, pages in all_data.items():
            if pages:
                self.llm_formatter.save_category(category_name, pages)

        # Create metadata file
        self.llm_formatter.create_metadata_file(taxonomy)

        logger.info("LLM-optimized output saved")

    async def run(self, test_mode: bool = False, skip_crawl: bool = False):
        """Run the complete pipeline"""
        logger.info("=" * 80)
        logger.info("Apple Documentation Crawler System")
        logger.info("=" * 80)

        try:
            # Step 1: Crawl (or load existing data)
            if skip_crawl:
                logger.info("Skipping crawl, loading existing data...")
                all_data = self._load_existing_data()
            else:
                all_data = await self.crawl(test_mode=test_mode)

            if not all_data or all(not pages for pages in all_data.values()):
                logger.error("No data collected. Exiting.")
                return

            # Step 2: Categorize and analyze
            taxonomy = self.categorize_and_analyze(all_data)

            # Step 3: Create visualizations
            self.create_visualizations(taxonomy)

            # Step 4: Generate human-readable output
            self.generate_human_readable_output(all_data, taxonomy)

            # Step 5: Generate LLM-optimized output
            self.generate_llm_output(all_data, taxonomy)

            logger.info("=" * 80)
            logger.info("Pipeline complete!")
            logger.info(f"Human-readable output: {self.project_root / 'output' / 'human_readable'}")
            logger.info(f"LLM-optimized output: {self.project_root / 'output' / 'llm_format'}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    def _load_existing_data(self) -> Dict[str, List[DocumentationPage]]:
        """Load existing raw data from disk"""
        import json

        data_dir = self.project_root / 'data'
        all_data = {}

        for json_file in data_dir.glob('*_raw.json'):
            category_name = json_file.stem.replace('_raw', '').replace('_', ' ').title()

            with open(json_file, 'r', encoding='utf-8') as f:
                pages_data = json.load(f)

            pages = [DocumentationPage(**page_data) for page_data in pages_data]
            all_data[category_name] = pages

            logger.info(f"Loaded {len(pages)} pages for {category_name}")

        return all_data


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Apple Documentation Crawler')
    parser.add_argument('--test', action='store_true', help='Run in test mode (limited crawl)')
    parser.add_argument('--skip-crawl', action='store_true', help='Skip crawling, use existing data')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')

    args = parser.parse_args()

    # Setup project paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / args.config

    # Create logs directory
    (project_root / 'logs').mkdir(exist_ok=True)

    # Create and run system
    system = AppleDocsSystem(config_path)
    await system.run(test_mode=args.test, skip_crawl=args.skip_crawl)


if __name__ == "__main__":
    asyncio.run(main())
