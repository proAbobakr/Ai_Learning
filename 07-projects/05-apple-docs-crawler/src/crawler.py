"""
Apple Documentation Crawler
Fetches and processes documentation from developer.apple.com
"""

import asyncio
import aiohttp
import time
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DocumentationPage:
    """Represents a single documentation page"""
    url: str
    title: str
    category: str
    content: str
    code_examples: List[str]
    sections: List[Dict[str, str]]
    related_links: List[str]
    metadata: Dict[str, str]

    def to_dict(self):
        return asdict(self)


class AppleDocsCrawler:
    """Crawls Apple Developer Documentation"""

    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['crawler']['base_url']
        self.delay = config['crawler']['delay_between_requests']
        self.max_retries = config['crawler']['max_retries']
        self.timeout = config['crawler']['timeout']
        self.user_agent = config['crawler']['user_agent']
        self.max_depth = config['crawler']['max_depth']
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page with retry logic"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching: {url} (attempt {attempt + 1})")

                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Rate limited
                        wait_time = self.delay * (attempt + 1) * 2
                        logger.warning(f"Rate limited. Waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}")
                await asyncio.sleep(self.delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                await asyncio.sleep(self.delay)

        return None

    def parse_documentation_page(self, html: str, url: str, category: str) -> Optional[DocumentationPage]:
        """Parse an Apple documentation page"""
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Untitled"

            # Extract main content
            content_areas = []

            # Look for common Apple docs content containers
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', {'role': 'main'})
            )

            if main_content:
                # Remove script and style elements
                for script in main_content(["script", "style"]):
                    script.decompose()

                content = main_content.get_text(separator='\n', strip=True)
            else:
                content = soup.get_text(separator='\n', strip=True)

            # Extract code examples
            code_examples = []
            code_blocks = soup.find_all(['code', 'pre'])
            for code in code_blocks:
                code_text = code.get_text(strip=True)
                if len(code_text) > 10:  # Ignore very short snippets
                    code_examples.append(code_text)

            # Extract sections
            sections = []
            headers = soup.find_all(['h2', 'h3', 'h4'])
            for header in headers:
                section_title = header.get_text(strip=True)
                section_content = ""

                # Get content until next header
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h2', 'h3', 'h4']:
                    section_content += next_sibling.get_text(strip=True) + "\n"
                    next_sibling = next_sibling.find_next_sibling()

                if section_content:
                    sections.append({
                        'title': section_title,
                        'content': section_content.strip()
                    })

            # Extract related links
            related_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/documentation/' in href:
                    full_url = urljoin(url, href)
                    if full_url not in related_links:
                        related_links.append(full_url)

            # Extract metadata
            metadata = {
                'url': url,
                'category': category,
                'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            # Look for framework/SDK info
            framework_elem = soup.find('meta', {'name': 'framework'})
            if framework_elem:
                metadata['framework'] = framework_elem.get('content', '')

            return DocumentationPage(
                url=url,
                title=title,
                category=category,
                content=content,
                code_examples=code_examples,
                sections=sections,
                related_links=related_links[:20],  # Limit related links
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None

    async def crawl_category(self, category: Dict, max_pages: int = 100) -> List[DocumentationPage]:
        """Crawl a specific category"""
        logger.info(f"Crawling category: {category['name']}")

        category_url = urljoin(self.base_url, category['url_pattern'])
        pages: List[DocumentationPage] = []
        to_visit = [category_url]
        visited = set()

        while to_visit and len(pages) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            visited.add(url)

            # Respect rate limiting
            await asyncio.sleep(self.delay)

            html = await self.fetch_page(url)
            if not html:
                continue

            doc_page = self.parse_documentation_page(html, url, category['name'])
            if doc_page:
                pages.append(doc_page)
                logger.info(f"Parsed: {doc_page.title} ({len(pages)}/{max_pages})")

                # Add related links to queue (breadth-first)
                for link in doc_page.related_links:
                    if (link not in visited and
                        link not in to_visit and
                        category['url_pattern'].lower() in link.lower()):
                        to_visit.append(link)

            # Safety limit
            if len(visited) > max_pages * 2:
                break

        logger.info(f"Completed {category['name']}: {len(pages)} pages")
        return pages

    async def crawl_all_categories(self, categories: List[Dict]) -> Dict[str, List[DocumentationPage]]:
        """Crawl all categories"""
        results = {}

        for category in categories:
            try:
                max_pages = self.config['crawler'].get('max_pages_per_category', 100)
                pages = await self.crawl_category(category, max_pages)
                results[category['name']] = pages
            except Exception as e:
                logger.error(f"Error crawling category {category['name']}: {e}")
                results[category['name']] = []

        return results

    def save_raw_data(self, data: Dict[str, List[DocumentationPage]], output_dir: Path):
        """Save raw crawled data to JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for category, pages in data.items():
            filename = output_dir / f"{category.lower().replace(' ', '_')}_raw.json"

            pages_dict = [page.to_dict() for page in pages]

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pages_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(pages)} pages to {filename}")


async def main():
    """Main entry point for testing"""
    import yaml

    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create crawler
    async with AppleDocsCrawler(config) as crawler:
        # Test with one category
        test_category = config['categories'][0]
        pages = await crawler.crawl_category(test_category, max_pages=5)

        # Save results
        output_dir = Path(__file__).parent.parent / 'data'
        crawler.save_raw_data({test_category['name']: pages}, output_dir)

        print(f"\nCrawled {len(pages)} pages from {test_category['name']}")
        for page in pages:
            print(f"  - {page.title}")


if __name__ == "__main__":
    asyncio.run(main())
