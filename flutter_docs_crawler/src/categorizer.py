"""
Content Categorizer
Categorizes Flutter documentation content based on keywords and context
"""

import yaml
import logging
from typing import List, Dict, Set
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentCategorizer:
    """Categorizes documentation content into logical groups"""

    def __init__(self, config_path: str = "config/crawler_config.yaml"):
        """Initialize categorizer with configuration"""
        self.config = self._load_config(config_path)
        self.categories = self.config.get('categories', [])
        self.categorized_content: Dict[str, List[Dict]] = defaultdict(list)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {'categories': []}

    def calculate_category_score(self, content: Dict, category: Dict) -> float:
        """Calculate relevance score for a category"""
        score = 0.0
        keywords = category.get('keywords', [])

        # Combine all text content
        text_content = ' '.join([
            content.get('title', ''),
            ' '.join([h['text'] for h in content.get('headings', [])]),
            ' '.join(content.get('paragraphs', []))[:1000]  # Limit paragraph text
        ]).lower()

        url = content.get('url', '').lower()

        # Score based on keyword matches
        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Higher score for title matches
            if keyword_lower in content.get('title', '').lower():
                score += 5.0

            # Medium score for URL matches
            if keyword_lower in url:
                score += 3.0

            # Lower score for content matches
            matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_content))
            score += matches * 0.5

        return score

    def categorize_content(self, scraped_content: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize all scraped content"""
        logger.info(f"Categorizing {len(scraped_content)} pages...")

        for content in scraped_content:
            best_category = None
            best_score = 0.0

            # Find best matching category
            for category in self.categories:
                score = self.calculate_category_score(content, category)

                if score > best_score:
                    best_score = score
                    best_category = category['name']

            # Assign to category (or 'Other' if no good match)
            category_name = best_category if best_score > 2.0 else 'Other'

            # Add category metadata to content
            content['category'] = category_name
            content['category_score'] = best_score

            self.categorized_content[category_name].append(content)

        # Log categorization results
        for category, items in self.categorized_content.items():
            logger.info(f"Category '{category}': {len(items)} pages")

        return dict(self.categorized_content)

    def get_category_statistics(self) -> Dict:
        """Get statistics about categorized content"""
        stats = {
            'total_pages': sum(len(items) for items in self.categorized_content.values()),
            'categories': {}
        }

        for category, items in self.categorized_content.items():
            stats['categories'][category] = {
                'page_count': len(items),
                'total_code_examples': sum(len(item.get('code_examples', [])) for item in items),
                'total_images': sum(len(item.get('images', [])) for item in items),
            }

        return stats
