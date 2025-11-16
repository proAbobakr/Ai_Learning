"""
Human-Readable Formatter
Creates formatted markdown documentation with diagrams and charts
"""

import logging
from typing import List, Dict
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HumanReadableFormatter:
    """Formats content into human-readable markdown with visualizations"""

    def __init__(self, output_dir: str = "output/human_readable"):
        """Initialize formatter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_category_chart(self, categorized_content: Dict[str, List[Dict]], output_path: str):
        """Create pie chart showing content distribution by category"""
        categories = list(categorized_content.keys())
        counts = [len(items) for items in categorized_content.values()]

        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])

        fig.update_layout(
            title="Flutter Documentation Distribution by Category",
            showlegend=True,
            height=500
        )

        fig.write_html(output_path)
        logger.info(f"Category chart saved to {output_path}")

    def create_content_stats_chart(self, categorized_content: Dict[str, List[Dict]], output_path: str):
        """Create bar chart showing content statistics per category"""
        categories = []
        code_examples = []
        images = []
        pages = []

        for category, items in categorized_content.items():
            categories.append(category)
            pages.append(len(items))
            code_examples.append(sum(len(item.get('code_examples', [])) for item in items))
            images.append(sum(len(item.get('images', [])) for item in items))

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Pages per Category", "Code Examples", "Images"),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=categories, y=pages, name="Pages", marker_color='lightblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=categories, y=code_examples, name="Code Examples", marker_color='lightgreen'),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(x=categories, y=images, name="Images", marker_color='lightcoral'),
            row=1, col=3
        )

        fig.update_layout(
            title_text="Flutter Documentation Content Statistics",
            showlegend=False,
            height=500
        )

        fig.update_xaxes(tickangle=45)

        fig.write_html(output_path)
        logger.info(f"Content stats chart saved to {output_path}")

    def format_code_example(self, code_example: Dict) -> str:
        """Format a code example in markdown"""
        language = code_example.get('language', 'dart')
        code = code_example.get('code', '')

        return f"```{language}\n{code}\n```\n"

    def format_page_content(self, content: Dict) -> str:
        """Format a single page's content in markdown"""
        md = []

        # Title
        title = content.get('title', 'Untitled')
        md.append(f"## {title}\n")

        # Source URL
        md.append(f"**Source:** [{content.get('url', '')}]({content.get('url', '')})\n")

        # Headings and structure
        md.append("### Content Overview\n")
        headings = content.get('headings', [])
        if headings:
            for heading in headings[:10]:  # Limit to first 10 headings
                level = int(heading['level'][1])  # Extract number from h1, h2, etc.
                indent = "  " * (level - 1)
                md.append(f"{indent}- {heading['text']}\n")
            md.append("\n")

        # Main content (first few paragraphs)
        md.append("### Description\n")
        paragraphs = content.get('paragraphs', [])
        for paragraph in paragraphs[:5]:  # First 5 paragraphs
            md.append(f"{paragraph}\n\n")

        # Code examples
        code_examples = content.get('code_examples', [])
        if code_examples:
            md.append("### Code Examples\n")
            for i, example in enumerate(code_examples[:5], 1):  # First 5 examples
                md.append(f"**Example {i}:**\n\n")
                md.append(self.format_code_example(example))
                md.append("\n")

        # Images
        images = content.get('images', [])
        if images:
            md.append("### Diagrams and Images\n")
            for img in images[:5]:  # First 5 images
                alt = img.get('alt', 'Image')
                src = img.get('src', '')
                md.append(f"![{alt}]({src})\n\n")

        # Tables
        tables = content.get('tables', [])
        if tables:
            md.append("### Tables\n")
            for table in tables[:3]:  # First 3 tables
                if table:
                    # Header
                    md.append("| " + " | ".join(table[0]) + " |\n")
                    md.append("| " + " | ".join(["---"] * len(table[0])) + " |\n")
                    # Rows
                    for row in table[1:]:
                        md.append("| " + " | ".join(row) + " |\n")
                    md.append("\n")

        md.append("---\n\n")
        return "".join(md)

    def generate_category_file(self, category: str, items: List[Dict], output_path: Path):
        """Generate a markdown file for a category"""
        md = []

        # Header
        md.append(f"# Flutter Documentation: {category}\n\n")
        md.append(f"*Total Pages: {len(items)}*\n\n")
        md.append(f"---\n\n")

        # Table of Contents
        md.append("## Table of Contents\n\n")
        for i, item in enumerate(items, 1):
            title = item.get('title', 'Untitled')
            md.append(f"{i}. [{title}](#{self._anchor_link(title)})\n")
        md.append("\n---\n\n")

        # Content
        for item in items:
            md.append(self.format_page_content(item))

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(md))

        logger.info(f"Generated {category} documentation: {output_path}")

    def _anchor_link(self, text: str) -> str:
        """Convert text to markdown anchor link"""
        return text.lower().replace(' ', '-').replace('/', '-')

    def generate_index(self, categorized_content: Dict[str, List[Dict]]):
        """Generate main index file"""
        md = []

        md.append("# Flutter Documentation Index\n\n")
        md.append("*Automatically generated documentation from docs.flutter.dev*\n\n")

        # Statistics
        total_pages = sum(len(items) for items in categorized_content.values())
        total_examples = sum(
            len(item.get('code_examples', []))
            for items in categorized_content.values()
            for item in items
        )

        md.append("## Overview\n\n")
        md.append(f"- **Total Pages:** {total_pages}\n")
        md.append(f"- **Total Categories:** {len(categorized_content)}\n")
        md.append(f"- **Total Code Examples:** {total_examples}\n\n")

        # Visualizations
        md.append("## Visualizations\n\n")
        md.append("- [Category Distribution Chart](charts/category_distribution.html)\n")
        md.append("- [Content Statistics Chart](charts/content_stats.html)\n\n")

        # Categories
        md.append("## Categories\n\n")
        for category, items in sorted(categorized_content.items()):
            filename = self._sanitize_filename(category)
            md.append(f"### {category}\n\n")
            md.append(f"- **Pages:** {len(items)}\n")
            md.append(f"- **File:** [{filename}.md]({filename}.md)\n\n")

        # Write index
        index_path = self.output_dir / "INDEX.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("".join(md))

        logger.info(f"Generated index: {index_path}")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize category name for filename"""
        return name.lower().replace(' ', '_').replace('/', '_')

    def format_all(self, categorized_content: Dict[str, List[Dict]]):
        """Generate all formatted output files"""
        logger.info("Generating human-readable documentation...")

        # Create charts directory
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Generate charts
        self.create_category_chart(
            categorized_content,
            str(charts_dir / "category_distribution.html")
        )
        self.create_content_stats_chart(
            categorized_content,
            str(charts_dir / "content_stats.html")
        )

        # Generate category files
        for category, items in categorized_content.items():
            if items:  # Only generate if there are items
                filename = self._sanitize_filename(category)
                output_path = self.output_dir / f"{filename}.md"
                self.generate_category_file(category, items, output_path)

        # Generate index
        self.generate_index(categorized_content)

        logger.info("Human-readable documentation complete!")
