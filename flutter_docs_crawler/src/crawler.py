"""
Flutter Documentation Crawler
Crawls and extracts content from Flutter documentation website
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import logging
from typing import Set, Dict, List, Optional
import yaml
import json
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlutterDocsCrawler:
    """Crawls Flutter documentation and extracts structured content"""

    def __init__(self, config_path: str = "config/crawler_config.yaml"):
        """Initialize the crawler with configuration"""
        self.config = self._load_config(config_path)
        self.base_url = self.config['crawler']['base_url']
        self.max_depth = self.config['crawler']['max_depth']
        self.delay = self.config['crawler']['delay_between_requests']
        self.timeout = self.config['crawler']['timeout']
        self.user_agent = self.config['crawler']['user_agent']

        self.visited_urls: Set[str] = set()
        self.scraped_content: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'crawler': {
                'base_url': 'https://docs.flutter.dev/',
                'max_depth': 5,
                'delay_between_requests': 1.0,
                'timeout': 30,
                'user_agent': 'Mozilla/5.0'
            },
            'categories': []
        }

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and within Flutter docs domain"""
        parsed = urlparse(url)
        base_parsed = urlparse(self.base_url)

        # Must be same domain
        if parsed.netloc != base_parsed.netloc:
            return False

        # Skip non-documentation pages
        skip_patterns = [
            '/search', '/api/', 'javascript:', 'mailto:',
            '.pdf', '.zip', '.tar', '#'
        ]

        for pattern in skip_patterns:
            if pattern in url:
                return False

        return True

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page"""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            return BeautifulSoup(response.content, 'lxml')

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content from a page"""
        content = {
            'url': url,
            'title': '',
            'headings': [],
            'paragraphs': [],
            'code_examples': [],
            'links': [],
            'images': [],
            'tables': []
        }

        # Extract title
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            content['title'] = title_tag.get_text(strip=True)

        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            content['headings'].append({
                'level': heading.name,
                'text': heading.get_text(strip=True)
            })

        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                content['paragraphs'].append(text)

        # Extract code examples
        for code in soup.find_all(['pre', 'code']):
            code_text = code.get_text(strip=True)
            if code_text and len(code_text) > 10:  # Skip very short snippets
                language = code.get('class', [''])[0].replace('language-', '')
                content['code_examples'].append({
                    'language': language or 'dart',
                    'code': code_text
                })

        # Extract links
        for link in soup.find_all('a', href=True):
            href = urljoin(url, link['href'])
            if self.is_valid_url(href):
                content['links'].append(href)

        # Extract images
        for img in soup.find_all('img', src=True):
            content['images'].append({
                'src': urljoin(url, img['src']),
                'alt': img.get('alt', '')
            })

        # Extract tables
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                content['tables'].append(rows)

        return content

    def crawl(self, start_url: Optional[str] = None, max_pages: int = 100) -> List[Dict]:
        """Crawl Flutter documentation starting from start_url"""
        if start_url is None:
            start_url = self.base_url

        urls_to_visit = [(start_url, 0)]  # (url, depth)
        pages_crawled = 0

        with tqdm(total=max_pages, desc="Crawling pages") as pbar:
            while urls_to_visit and pages_crawled < max_pages:
                current_url, depth = urls_to_visit.pop(0)

                # Skip if already visited or too deep
                if current_url in self.visited_urls or depth > self.max_depth:
                    continue

                self.visited_urls.add(current_url)

                # Fetch and parse page
                soup = self.fetch_page(current_url)
                if not soup:
                    continue

                # Extract content
                content = self.extract_content(soup, current_url)
                self.scraped_content.append(content)

                # Add new links to visit
                for link in content['links']:
                    if link not in self.visited_urls:
                        urls_to_visit.append((link, depth + 1))

                pages_crawled += 1
                pbar.update(1)

                # Respectful crawling delay
                time.sleep(self.delay)

        logger.info(f"Crawling complete. Scraped {len(self.scraped_content)} pages.")
        return self.scraped_content

    def save_raw_data(self, output_path: str = "output/raw_data.json"):
        """Save raw scraped data to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_content, f, indent=2, ensure_ascii=False)

        logger.info(f"Raw data saved to {output_path}")


if __name__ == "__main__":
    crawler = FlutterDocsCrawler()
    crawler.crawl(max_pages=50)  # Limit for testing
    crawler.save_raw_data()
