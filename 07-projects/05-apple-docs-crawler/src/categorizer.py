"""
Documentation Categorizer and Analyzer
Categorizes and analyzes Apple documentation
"""

import re
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class CategoryStats:
    """Statistics for a documentation category"""
    name: str
    total_pages: int
    total_code_examples: int
    avg_content_length: float
    top_topics: List[tuple]
    frameworks: Set[str]
    complexity_score: float


class DocumentationCategorizer:
    """Categorizes and analyzes documentation"""

    # Keywords for different subcategories
    SUBCATEGORY_KEYWORDS = {
        'UI': ['view', 'button', 'label', 'image', 'text', 'layout', 'interface'],
        'Data': ['data', 'model', 'storage', 'database', 'persistence', 'core data'],
        'Networking': ['network', 'http', 'url', 'request', 'response', 'api'],
        'Animation': ['animation', 'transition', 'animate', 'motion', 'gesture'],
        'Graphics': ['graphics', 'drawing', 'render', 'image', 'color', 'shape'],
        'Audio': ['audio', 'sound', 'music', 'player', 'recording'],
        'Video': ['video', 'camera', 'avkit', 'playback'],
        'AR/VR': ['arkit', 'augmented', 'reality', 'ar', 'scene'],
        'ML': ['machine learning', 'ml', 'coreml', 'model', 'prediction'],
        'Security': ['security', 'authentication', 'encryption', 'keychain', 'biometric'],
        'Performance': ['performance', 'optimization', 'memory', 'cpu', 'battery'],
        'Testing': ['test', 'xctest', 'unit test', 'ui test', 'mock'],
    }

    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        'high': ['protocol', 'generic', 'async', 'concurrent', 'thread', 'delegate'],
        'medium': ['class', 'struct', 'enum', 'function', 'method'],
        'low': ['property', 'constant', 'variable', 'parameter']
    }

    def __init__(self):
        self.categories = defaultdict(list)

    def categorize_by_topic(self, doc_page) -> List[str]:
        """Categorize a page by its topics"""
        topics = []
        content_lower = (doc_page.title + " " + doc_page.content).lower()

        for topic, keywords in self.SUBCATEGORY_KEYWORDS.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)

        return topics if topics else ['General']

    def calculate_complexity(self, doc_page) -> float:
        """Calculate complexity score (0-100)"""
        content = doc_page.title + " " + doc_page.content
        content_lower = content.lower()

        score = 0

        # High complexity indicators (+10 each)
        for indicator in self.COMPLEXITY_INDICATORS['high']:
            score += content_lower.count(indicator) * 10

        # Medium complexity indicators (+5 each)
        for indicator in self.COMPLEXITY_INDICATORS['medium']:
            score += content_lower.count(indicator) * 5

        # Code examples increase complexity
        score += len(doc_page.code_examples) * 8

        # Long content increases complexity
        if len(doc_page.content) > 5000:
            score += 15
        elif len(doc_page.content) > 2000:
            score += 5

        # Number of sections
        score += len(doc_page.sections) * 3

        # Normalize to 0-100 scale
        return min(score, 100)

    def extract_frameworks(self, doc_page) -> Set[str]:
        """Extract mentioned frameworks"""
        frameworks = set()

        # Common iOS/macOS frameworks
        framework_list = [
            'SwiftUI', 'UIKit', 'AppKit', 'Foundation', 'Combine',
            'CoreData', 'CoreGraphics', 'CoreAnimation', 'AVFoundation',
            'ARKit', 'RealityKit', 'SceneKit', 'SpriteKit', 'GameplayKit',
            'MapKit', 'CoreLocation', 'CoreML', 'Vision', 'CreateML',
            'StoreKit', 'CloudKit', 'HealthKit', 'HomeKit', 'WatchKit',
            'WidgetKit', 'CoreBluetooth', 'CoreNFC', 'PassKit',
            'NetworkFramework', 'Combine', 'RxSwift'
        ]

        content = doc_page.title + " " + doc_page.content

        for framework in framework_list:
            if framework.lower() in content.lower():
                frameworks.add(framework)

        return frameworks

    def extract_keywords(self, doc_page, top_n: int = 10) -> List[tuple]:
        """Extract top keywords from documentation"""
        # Common words to ignore
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'you', 'your', 'we', 'our', 'they', 'their'
        }

        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', doc_page.content.lower())

        # Filter and count
        filtered_words = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered_words)

        return word_counts.most_common(top_n)

    def analyze_category(self, pages: List, category_name: str) -> CategoryStats:
        """Analyze a category of documentation pages"""
        if not pages:
            return CategoryStats(
                name=category_name,
                total_pages=0,
                total_code_examples=0,
                avg_content_length=0,
                top_topics=[],
                frameworks=set(),
                complexity_score=0
            )

        total_code_examples = sum(len(p.code_examples) for p in pages)
        avg_content_length = sum(len(p.content) for p in pages) / len(pages)

        # Collect all topics
        all_topics = []
        for page in pages:
            all_topics.extend(self.categorize_by_topic(page))

        top_topics = Counter(all_topics).most_common(5)

        # Collect all frameworks
        all_frameworks = set()
        for page in pages:
            all_frameworks.update(self.extract_frameworks(page))

        # Average complexity
        avg_complexity = sum(self.calculate_complexity(p) for p in pages) / len(pages)

        return CategoryStats(
            name=category_name,
            total_pages=len(pages),
            total_code_examples=total_code_examples,
            avg_content_length=avg_content_length,
            top_topics=top_topics,
            frameworks=all_frameworks,
            complexity_score=avg_complexity
        )

    def create_taxonomy(self, all_data: Dict[str, List]) -> Dict:
        """Create a taxonomy of all documentation"""
        taxonomy = {
            'categories': {},
            'global_stats': {
                'total_pages': 0,
                'total_code_examples': 0,
                'frameworks': set(),
                'topics': Counter()
            }
        }

        for category_name, pages in all_data.items():
            stats = self.analyze_category(pages, category_name)
            taxonomy['categories'][category_name] = stats

            # Update global stats
            taxonomy['global_stats']['total_pages'] += stats.total_pages
            taxonomy['global_stats']['total_code_examples'] += stats.total_code_examples
            taxonomy['global_stats']['frameworks'].update(stats.frameworks)

            for topic, count in stats.top_topics:
                taxonomy['global_stats']['topics'][topic] += count

        # Convert Counter to list of tuples
        taxonomy['global_stats']['topics'] = taxonomy['global_stats']['topics'].most_common(20)
        taxonomy['global_stats']['frameworks'] = list(taxonomy['global_stats']['frameworks'])

        return taxonomy

    def group_by_subcategory(self, pages: List) -> Dict[str, List]:
        """Group pages by subcategories"""
        grouped = defaultdict(list)

        for page in pages:
            topics = self.categorize_by_topic(page)
            for topic in topics:
                grouped[topic].append(page)

        return dict(grouped)

    def find_related_docs(self, page, all_pages: List, top_n: int = 5) -> List:
        """Find related documentation pages"""
        # Simple similarity based on common keywords
        page_keywords = set(word for word, _ in self.extract_keywords(page, 20))

        similarities = []
        for other_page in all_pages:
            if other_page.url == page.url:
                continue

            other_keywords = set(word for word, _ in self.extract_keywords(other_page, 20))
            common_keywords = page_keywords & other_keywords

            if common_keywords:
                similarity = len(common_keywords) / max(len(page_keywords), len(other_keywords))
                similarities.append((similarity, other_page))

        # Sort by similarity and return top N
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [page for _, page in similarities[:top_n]]


if __name__ == "__main__":
    # Test the categorizer
    from crawler import DocumentationPage

    # Create sample page
    sample_page = DocumentationPage(
        url="https://developer.apple.com/documentation/swiftui/view",
        title="View - SwiftUI",
        category="SwiftUI",
        content="A view represents a portion of your app's user interface. SwiftUI provides built-in views for common interface elements.",
        code_examples=["struct ContentView: View { var body: some View { Text('Hello') } }"],
        sections=[{"title": "Overview", "content": "Views are the building blocks"}],
        related_links=[],
        metadata={}
    )

    categorizer = DocumentationCategorizer()
    topics = categorizer.categorize_by_topic(sample_page)
    complexity = categorizer.calculate_complexity(sample_page)
    frameworks = categorizer.extract_frameworks(sample_page)

    print(f"Topics: {topics}")
    print(f"Complexity: {complexity}")
    print(f"Frameworks: {frameworks}")
