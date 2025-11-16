# Flutter Documentation Crawler

A comprehensive system to crawl, categorize, and format Flutter documentation from [docs.flutter.dev](https://docs.flutter.dev/) for both human consumption and local LLM training/usage.

## Features

- 🕷️ **Web Crawler**: Systematically crawls Flutter documentation
- 🏷️ **Auto-Categorization**: Intelligently categorizes content into logical groups
- 📊 **Visualizations**: Generates interactive charts and diagrams
- 📖 **Human-Readable Output**: Beautiful markdown documentation with code examples
- 🤖 **LLM-Optimized Format**: Structured text optimized for local LLM consumption
- 📦 **Organized Structure**: Separate files for each category

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd flutter_docs_crawler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Crawl 100 pages (default) and generate all outputs:
```bash
python main.py
```

### Advanced Options

Crawl specific number of pages:
```bash
python main.py --max-pages 200
```

Use existing crawled data (skip crawling):
```bash
python main.py --skip-crawl
```

Custom configuration file:
```bash
python main.py --config my_config.yaml
```

### Full Options

```bash
python main.py --help
```

Options:
- `--max-pages N`: Maximum number of pages to crawl (default: 100)
- `--config PATH`: Path to configuration file (default: config/crawler_config.yaml)
- `--skip-crawl`: Skip crawling and use existing raw data
- `--raw-data PATH`: Path to raw data file (default: output/raw_data.json)

## Output Structure

### Human-Readable Format (`output/human_readable/`)

- **INDEX.md**: Main index with overview and links to all categories
- **Category files**: One markdown file per category (e.g., `widgets.md`, `state_management.md`)
- **charts/**: Interactive HTML charts showing content distribution

#### Features:
- Clean markdown formatting
- Syntax-highlighted code examples
- Embedded images and diagrams
- Tables of contents
- Cross-references

### LLM-Optimized Format (`output/llm_format/`)

- **Category files**: Plain text files optimized for LLM ingestion (e.g., `widgets.txt`)
- **metadata.json**: Metadata about the documentation
- **flutter_docs_complete.txt**: Single combined file with all documentation
- **Chunked files**: Large categories split into chunks (for context window limits)

#### Features:
- Clear section markers for parsing
- Optimized formatting for context windows
- Metadata headers for each document
- Key concepts extraction
- No unnecessary formatting characters

## Configuration

Edit `config/crawler_config.yaml` to customize:

### Crawler Settings
```yaml
crawler:
  base_url: "https://docs.flutter.dev/"
  max_depth: 5
  delay_between_requests: 1.0
  timeout: 30
```

### Categories
Add or modify categories and their keywords:
```yaml
categories:
  - name: "Widgets"
    keywords: ["widget", "stateless", "stateful", "layout"]
  - name: "State Management"
    keywords: ["state", "provider", "bloc", "riverpod"]
```

## Project Structure

```
flutter_docs_crawler/
├── main.py                 # Main orchestrator script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/
│   └── crawler_config.yaml   # Configuration file
├── src/
│   ├── crawler.py            # Web crawler implementation
│   ├── categorizer.py        # Content categorization
│   ├── human_readable_formatter.py  # Human-readable output
│   └── llm_formatter.py      # LLM-optimized output
├── output/
│   ├── human_readable/       # Markdown documentation
│   ├── llm_format/           # LLM-optimized text files
│   └── raw_data.json         # Raw scraped data
└── logs/
    └── crawler.log           # Execution logs
```

## Categories

Default categories include:

- Getting Started
- Widgets
- State Management
- Navigation
- Networking
- Persistence
- Animation
- Testing
- Deployment
- Platform Integration
- Performance
- Architecture
- Other (uncategorized content)

## Examples

### Use Case 1: Training a Local LLM

```bash
# Crawl extensive documentation
python main.py --max-pages 500

# Use the combined file for training
cat output/llm_format/flutter_docs_complete.txt
```

### Use Case 2: Creating a Reference Guide

```bash
# Crawl and generate markdown docs
python main.py --max-pages 200

# View the index
cat output/human_readable/INDEX.md
```

### Use Case 3: Updating Existing Data

```bash
# Re-process existing crawled data with new categories
python main.py --skip-crawl
```

## LLM Integration Tips

### For RAG (Retrieval-Augmented Generation)

1. Use individual category files from `output/llm_format/`
2. Load relevant categories based on user queries
3. Chunk files are pre-split for optimal context window usage

### For Fine-tuning

1. Use `flutter_docs_complete.txt` as training data
2. Documents are clearly marked with separators
3. Metadata includes source URLs and categories

### Example: Loading into LLM Context

```python
# Load specific category
with open('output/llm_format/widgets.txt', 'r') as f:
    widgets_docs = f.read()

# Feed to your local LLM
response = llm.generate(
    prompt=f"Based on this documentation:\n\n{widgets_docs}\n\nHow do I create a custom widget?",
    max_tokens=1000
)
```

## Visualizations

The system generates interactive charts:

1. **Category Distribution**: Pie chart showing content spread across categories
2. **Content Statistics**: Bar charts showing pages, code examples, and images per category

View these in your browser:
```bash
open output/human_readable/charts/category_distribution.html
open output/human_readable/charts/content_stats.html
```

## Logging

Logs are written to:
- Console (INFO level)
- `logs/crawler.log` (detailed logs)

## Troubleshooting

### Issue: Crawler is slow
**Solution**: Adjust `delay_between_requests` in config (min 0.5s recommended)

### Issue: Pages not categorizing correctly
**Solution**: Add more keywords to categories in `crawler_config.yaml`

### Issue: Out of memory
**Solution**: Reduce `--max-pages` or process categories separately

### Issue: Network errors
**Solution**: Check internet connection, increase `timeout` in config

## Performance

- **Crawling Speed**: ~1 page per second (respects server limits)
- **Memory Usage**: ~100MB for 100 pages
- **Disk Usage**: ~10MB per 100 pages

## Ethics & Best Practices

This crawler:
- ✅ Respects robots.txt
- ✅ Includes delays between requests
- ✅ Uses polite user agent string
- ✅ Only crawls public documentation
- ✅ Preserves attribution and source links

## License

This tool is for educational and research purposes. Flutter documentation is property of Google LLC.

## Contributing

To add new features:
1. Add new formatters in `src/`
2. Update `main.py` to call new formatters
3. Document in this README

## Support

For issues or questions:
1. Check the logs: `logs/crawler.log`
2. Review configuration: `config/crawler_config.yaml`
3. Validate output structure in `output/` directory

## Roadmap

Future enhancements:
- [ ] Async crawling for better performance
- [ ] API endpoint for on-demand crawling
- [ ] Version tracking for documentation updates
- [ ] Search functionality for crawled content
- [ ] Export to additional formats (PDF, EPUB)
- [ ] Integration with popular LLM frameworks

---

**Happy Crawling!** 🚀
