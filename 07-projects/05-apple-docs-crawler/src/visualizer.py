"""
Visualization and Diagram Generator
Creates charts and diagrams for documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import networkx as nx
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DocumentationVisualizer:
    """Creates visualizations for documentation analysis"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for better-looking charts
        plt.style.use('seaborn-v0_8-darkgrid')

    def create_category_distribution_chart(self, taxonomy: Dict, filename: str = "category_distribution.png"):
        """Create a bar chart showing pages per category"""
        categories = taxonomy['categories']

        if not categories:
            logger.warning("No categories to visualize")
            return

        names = list(categories.keys())
        values = [stats.total_pages for stats in categories.values()]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, values, color=cm.viridis([i/len(names) for i in range(len(names))]))

        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Pages', fontsize=12, fontweight='bold')
        ax.set_title('Documentation Pages per Category', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created category distribution chart: {output_path}")

    def create_complexity_chart(self, taxonomy: Dict, filename: str = "complexity_analysis.png"):
        """Create a chart showing complexity scores"""
        categories = taxonomy['categories']

        if not categories:
            return

        names = list(categories.keys())
        complexities = [stats.complexity_score for stats in categories.values()]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars based on complexity level
        colors = ['green' if c < 30 else 'orange' if c < 60 else 'red' for c in complexities]
        bars = ax.bar(names, complexities, color=colors, alpha=0.7)

        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Complexity Score', fontsize=12, fontweight='bold')
        ax.set_title('Documentation Complexity by Category', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 100)

        # Add legend
        low_patch = mpatches.Patch(color='green', alpha=0.7, label='Low (0-30)')
        med_patch = mpatches.Patch(color='orange', alpha=0.7, label='Medium (30-60)')
        high_patch = mpatches.Patch(color='red', alpha=0.7, label='High (60-100)')
        ax.legend(handles=[low_patch, med_patch, high_patch])

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created complexity chart: {output_path}")

    def create_code_examples_chart(self, taxonomy: Dict, filename: str = "code_examples.png"):
        """Create a chart showing code examples per category"""
        categories = taxonomy['categories']

        if not categories:
            return

        names = list(categories.keys())
        examples = [stats.total_code_examples for stats in categories.values()]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, examples, color='skyblue', edgecolor='navy', alpha=0.7)

        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Code Examples', fontsize=12, fontweight='bold')
        ax.set_title('Code Examples per Category', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created code examples chart: {output_path}")

    def create_topic_distribution_pie(self, taxonomy: Dict, filename: str = "topic_distribution.png"):
        """Create a pie chart of top topics"""
        topics = taxonomy['global_stats'].get('topics', [])

        if not topics:
            return

        # Take top 10 topics
        top_topics = topics[:10]
        labels = [topic for topic, _ in top_topics]
        sizes = [count for _, count in top_topics]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = cm.Set3(range(len(labels)))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Top Documentation Topics', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created topic distribution pie chart: {output_path}")

    def create_framework_network_diagram(self, taxonomy: Dict, filename: str = "framework_network.png"):
        """Create a network diagram showing framework relationships"""
        categories = taxonomy['categories']

        # Create graph
        G = nx.Graph()

        # Add nodes for each framework
        all_frameworks = set()
        for stats in categories.values():
            all_frameworks.update(stats.frameworks)

        if not all_frameworks:
            logger.warning("No frameworks found for network diagram")
            return

        for framework in all_frameworks:
            G.add_node(framework)

        # Add edges between frameworks that appear in same categories
        framework_categories = {}
        for cat_name, stats in categories.items():
            for framework in stats.frameworks:
                if framework not in framework_categories:
                    framework_categories[framework] = set()
                framework_categories[framework].add(cat_name)

        # Connect frameworks that share categories
        frameworks_list = list(all_frameworks)
        for i, fw1 in enumerate(frameworks_list):
            for fw2 in frameworks_list[i+1:]:
                shared = framework_categories[fw1] & framework_categories[fw2]
                if shared:
                    G.add_edge(fw1, fw2, weight=len(shared))

        # Draw graph
        fig, ax = plt.subplots(figsize=(14, 10))

        # Use spring layout for better positioning
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.7,
            ax=ax
        )

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos,
            width=[w/max_weight * 3 for w in weights],
            alpha=0.5,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Framework Relationships', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Created framework network diagram: {output_path}")

    def create_all_visualizations(self, taxonomy: Dict):
        """Create all visualizations"""
        logger.info("Creating all visualizations...")

        self.create_category_distribution_chart(taxonomy)
        self.create_complexity_chart(taxonomy)
        self.create_code_examples_chart(taxonomy)
        self.create_topic_distribution_pie(taxonomy)
        self.create_framework_network_diagram(taxonomy)

        logger.info(f"All visualizations saved to {self.output_dir}")

    def generate_mermaid_diagram(self, taxonomy: Dict) -> str:
        """Generate a Mermaid diagram for documentation structure"""
        mermaid = ["graph TD"]
        mermaid.append("    Root[Apple Documentation]")

        categories = taxonomy['categories']
        for i, (cat_name, stats) in enumerate(categories.items()):
            cat_id = f"Cat{i}"
            mermaid.append(f"    Root --> {cat_id}[{cat_name}<br/>{stats.total_pages} pages]")

            # Add top topics as sub-nodes
            for j, (topic, count) in enumerate(stats.top_topics[:3]):
                topic_id = f"Topic{i}_{j}"
                mermaid.append(f"    {cat_id} --> {topic_id}[{topic}<br/>{count} docs]")

        return "\n".join(mermaid)


if __name__ == "__main__":
    # Test visualization
    from categorizer import CategoryStats

    # Create sample taxonomy
    sample_taxonomy = {
        'categories': {
            'SwiftUI': CategoryStats(
                name='SwiftUI',
                total_pages=50,
                total_code_examples=120,
                avg_content_length=2500,
                top_topics=[('UI', 30), ('Animation', 15)],
                frameworks={'SwiftUI', 'Combine'},
                complexity_score=45
            ),
            'UIKit': CategoryStats(
                name='UIKit',
                total_pages=80,
                total_code_examples=200,
                avg_content_length=3000,
                top_topics=[('UI', 50), ('Graphics', 20)],
                frameworks={'UIKit', 'Foundation'},
                complexity_score=65
            )
        },
        'global_stats': {
            'total_pages': 130,
            'total_code_examples': 320,
            'topics': [('UI', 80), ('Animation', 15), ('Graphics', 20)],
            'frameworks': ['SwiftUI', 'UIKit', 'Combine', 'Foundation']
        }
    }

    viz = DocumentationVisualizer(Path('./test_output'))
    viz.create_all_visualizations(sample_taxonomy)
    print("Test visualizations created!")
