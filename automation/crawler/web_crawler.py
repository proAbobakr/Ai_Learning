"""
Web Crawler for RAG Document Updates
Fetches documents from configured URLs and detects changes
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentCrawler:
    """Crawls web documents and tracks changes"""

    def __init__(self, config_path: str = "config/crawler_config.json"):
        """
        Initialize the crawler

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.state_file = Path(self.config.get('state_file', 'automation/crawler/crawler_state.json'))
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_config(self) -> Dict:
        """Load crawler configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return {
                "urls": [],
                "output_dir": "data/crawled_docs",
                "state_file": "automation/crawler/crawler_state.json"
            }

    def _load_state(self) -> Dict:
        """Load previous crawl state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"last_crawl": None, "documents": {}}

    def _save_state(self):
        """Save current crawl state"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate hash of content for change detection"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _fetch_url(self, url: str) -> Optional[Dict]:
        """
        Fetch content from URL

        Args:
            url: URL to fetch

        Returns:
            Dictionary with content and metadata or None if failed
        """
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Extract title
            title = soup.find('title')
            title_text = title.string if title else url.split('/')[-1]

            # Extract main sections
            sections = []
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                section_text = heading.get_text(strip=True)
                # Get content after heading until next heading
                content = []
                for sibling in heading.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3']:
                        break
                    content.append(sibling.get_text(strip=True))

                sections.append({
                    'heading': section_text,
                    'level': heading.name,
                    'content': '\n'.join(content)
                })

            return {
                'url': url,
                'title': title_text,
                'content': text,
                'sections': sections,
                'hash': self._calculate_hash(text),
                'fetched_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def crawl(self) -> Dict:
        """
        Execute crawl of all configured URLs

        Returns:
            Dictionary with crawl results and changes detected
        """
        logger.info("Starting crawl...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'new_documents': [],
            'updated_documents': [],
            'unchanged_documents': [],
            'failed_urls': []
        }

        urls = self.config.get('urls', [])
        output_dir = Path(self.config.get('output_dir', 'data/crawled_docs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            doc = self._fetch_url(url)

            if not doc:
                results['failed_urls'].append(url)
                continue

            url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

            # Check if document changed
            if url in self.state['documents']:
                old_hash = self.state['documents'][url].get('hash')
                if old_hash == doc['hash']:
                    results['unchanged_documents'].append(url)
                    logger.info(f"No changes: {url}")
                else:
                    results['updated_documents'].append({
                        'url': url,
                        'title': doc['title'],
                        'sections': doc['sections']
                    })
                    logger.info(f"Updated: {url}")
            else:
                results['new_documents'].append({
                    'url': url,
                    'title': doc['title'],
                    'sections': doc['sections']
                })
                logger.info(f"New document: {url}")

            # Save document
            filename = f"{url_hash}.txt"
            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {doc['title']}\n")
                f.write(f"URL: {doc['url']}\n")
                f.write(f"Fetched: {doc['fetched_at']}\n")
                f.write(f"\n{'='*80}\n\n")
                f.write(doc['content'])

            # Update state
            self.state['documents'][url] = {
                'hash': doc['hash'],
                'title': doc['title'],
                'last_updated': doc['fetched_at'],
                'filepath': str(filepath)
            }

        # Update state
        self.state['last_crawl'] = datetime.now().isoformat()
        self._save_state()

        logger.info(f"Crawl complete. New: {len(results['new_documents'])}, "
                   f"Updated: {len(results['updated_documents'])}, "
                   f"Unchanged: {len(results['unchanged_documents'])}, "
                   f"Failed: {len(results['failed_urls'])}")

        return results

    def get_summary(self, results: Dict) -> str:
        """
        Generate human-readable summary of crawl results

        Args:
            results: Crawl results dictionary

        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append("WEEKLY DOCUMENT CRAWLER REPORT")
        summary.append("=" * 80)
        summary.append(f"\nCrawl Time: {results['timestamp']}\n")

        # New documents
        if results['new_documents']:
            summary.append(f"\n📄 NEW DOCUMENTS ({len(results['new_documents'])})")
            summary.append("-" * 80)
            for doc in results['new_documents']:
                summary.append(f"\nTitle: {doc['title']}")
                summary.append(f"URL: {doc['url']}")
                if doc['sections']:
                    summary.append("\nSections:")
                    for section in doc['sections'][:5]:  # Show first 5 sections
                        summary.append(f"  • {section['heading']}")
                summary.append("")

        # Updated documents
        if results['updated_documents']:
            summary.append(f"\n🔄 UPDATED DOCUMENTS ({len(results['updated_documents'])})")
            summary.append("-" * 80)
            for doc in results['updated_documents']:
                summary.append(f"\nTitle: {doc['title']}")
                summary.append(f"URL: {doc['url']}")
                if doc['sections']:
                    summary.append("\nSections:")
                    for section in doc['sections'][:5]:
                        summary.append(f"  • {section['heading']}")
                summary.append("")

        # Stats
        summary.append("\n" + "=" * 80)
        summary.append("SUMMARY")
        summary.append("=" * 80)
        summary.append(f"New Documents: {len(results['new_documents'])}")
        summary.append(f"Updated Documents: {len(results['updated_documents'])}")
        summary.append(f"Unchanged Documents: {len(results['unchanged_documents'])}")
        summary.append(f"Failed URLs: {len(results['failed_urls'])}")

        if results['failed_urls']:
            summary.append("\n⚠️  Failed URLs:")
            for url in results['failed_urls']:
                summary.append(f"  • {url}")

        return "\n".join(summary)


def main():
    """Main execution function for standalone use"""
    import sys

    config_file = sys.argv[1] if len(sys.argv) > 1 else "automation/crawler/config/crawler_config.json"

    crawler = DocumentCrawler(config_file)
    results = crawler.crawl()

    # Print summary
    summary = crawler.get_summary(results)
    print(summary)

    # Save results for n8n
    results_file = Path("automation/crawler/last_crawl_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Exit with status code
    if results['failed_urls']:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
