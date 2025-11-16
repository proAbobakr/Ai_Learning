# Apple Documentation Crawler System

A comprehensive system for crawling, categorizing, and formatting Apple Developer Documentation. This tool automatically fetches documentation from [developer.apple.com](https://developer.apple.com/documentation/), organizes it by category, generates visualizations, and outputs the data in both human-readable and LLM-optimized formats.

## 🌟 Features

- **Automated Crawling**: Fetches documentation from Apple's developer site with intelligent rate limiting
- **Smart Categorization**: Automatically categorizes documentation by framework, topic, and complexity
- **Rich Visualizations**: Generates charts and network diagrams showing documentation structure
- **Dual Output Formats**:
  - **Human-Readable**: Beautiful markdown files with diagrams, examples, and cross-references
  - **LLM-Optimized**: JSON/JSONL format optimized for feeding into local LLMs
- **Category Organization**: Each category saved in separate files for easy navigation
- **Code Examples**: Extracts and highlights code examples from documentation
- **Complexity Analysis**: Scores documentation complexity to help learners find appropriate content
- **Framework Relationships**: Maps relationships between different iOS/macOS frameworks

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Output Structure](#-output-structure)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone or navigate to the project**:
   ```bash
   cd 07-projects/05-apple-docs-crawler
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python src/main.py --help
   ```

## ⚡ Quick Start

### Test Run (Recommended First)

Run a limited crawl to test the system:

```bash
python src/main.py --test
```

This will:
- Crawl only 2 categories
- Fetch 5 pages per category
- Generate all outputs and visualizations
- Complete in ~2-5 minutes

### Full Run

Crawl all configured categories:

```bash
python src/main.py
```

**Note**: A full run may take 30-60 minutes depending on network speed and configured limits.

### Use Existing Data

If you've already crawled data and just want to regenerate outputs:

```bash
python src/main.py --skip-crawl
```

## ⚙️ Configuration

Configuration is managed via `config/config.yaml`. Key settings:

### Crawler Settings

```yaml
crawler:
  delay_between_requests: 2  # Seconds between requests
  max_pages_per_category: 100  # Limit pages per category
  max_depth: 3  # Maximum crawl depth
  timeout: 30  # Request timeout in seconds
```

### Categories

Edit the categories list to focus on specific frameworks:

```yaml
categories:
  - name: "SwiftUI"
    url_pattern: "swiftui"
    description: "SwiftUI framework documentation"

  - name: "UIKit"
    url_pattern: "uikit"
    description: "UIKit framework documentation"

  # Add more categories...
```

### Output Settings

```yaml
output:
  human_readable:
    format: "markdown"
    include_diagrams: true
    include_examples: true

  llm_format:
    chunk_size: 2000  # Optimal for most LLMs
    include_metadata: true
```

## 📖 Usage

### Basic Usage

```bash
# Full crawl with default settings
python src/main.py

# Test mode (quick run)
python src/main.py --test

# Use existing data
python src/main.py --skip-crawl

# Custom config file
python src/main.py --config path/to/config.yaml
```

### Using Individual Components

You can also use components individually:

```python
import asyncio
from src.crawler import AppleDocsCrawler
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Use crawler
async def crawl_swift():
    async with AppleDocsCrawler(config) as crawler:
        category = {'name': 'Swift', 'url_pattern': 'swift'}
        pages = await crawler.crawl_category(category, max_pages=10)
        print(f"Crawled {len(pages)} pages")

asyncio.run(crawl_swift())
```

## 📂 Output Structure

After running, you'll find outputs organized as follows:

```
07-projects/05-apple-docs-crawler/
├── output/
│   ├── human_readable/           # Human-readable markdown files
│   │   ├── INDEX.md             # Master index
│   │   ├── SwiftUI/             # Category directories
│   │   │   ├── swiftui_index.md
│   │   │   ├── view_protocol.md
│   │   │   └── ...
│   │   ├── UIKit/
│   │   └── diagrams/            # Charts and diagrams
│   │       ├── category_distribution.png
│   │       ├── complexity_analysis.png
│   │       ├── framework_network.png
│   │       └── structure.mmd    # Mermaid diagram
│   │
│   └── llm_format/              # LLM-optimized format
│       ├── dataset_metadata.json
│       ├── swiftui.json         # Full JSON
│       ├── swiftui.jsonl        # JSON Lines format
│       ├── uikit.json
│       └── ...
│
├── data/                         # Raw crawled data
│   ├── swiftui_raw.json
│   ├── uikit_raw.json
│   ├── taxonomy.json            # Complete taxonomy
│   └── ...
│
└── logs/                         # Execution logs
    └── crawler.log
```

## 🏗️ Architecture

### System Components

1. **Crawler** (`crawler.py`)
   - Asynchronous web crawler
   - Rate limiting and retry logic
   - HTML parsing and content extraction

2. **Categorizer** (`categorizer.py`)
   - Topic classification
   - Complexity scoring
   - Framework detection
   - Keyword extraction

3. **Visualizer** (`visualizer.py`)
   - Category distribution charts
   - Complexity analysis
   - Framework relationship networks
   - Mermaid diagrams

4. **Formatters** (`formatters.py`)
   - **Human-Readable**: Markdown with cross-references
   - **LLM Format**: JSON/JSONL optimized for LLMs

5. **Main Orchestrator** (`main.py`)
   - Coordinates all components
   - Pipeline execution
   - Error handling and logging

### Data Flow

```
Apple Docs → Crawler → Raw Data
                ↓
          Categorizer → Taxonomy
                ↓
         ┌──────┴──────┐
         ↓             ↓
    Visualizer    Formatters
         ↓             ↓
    Diagrams    Human + LLM Output
```

## 📊 Visualizations

The system generates several types of visualizations:

1. **Category Distribution**: Bar chart showing pages per category
2. **Complexity Analysis**: Complexity scores by category
3. **Code Examples**: Number of code examples per category
4. **Topic Distribution**: Pie chart of top topics
5. **Framework Network**: Network diagram showing framework relationships
6. **Structure Diagram**: Mermaid diagram of documentation hierarchy

## 🎯 Use Cases

### For Developers

- **Learning iOS Development**: Browse organized, categorized documentation
- **Quick Reference**: Search by complexity level
- **Framework Exploration**: Understand relationships between frameworks

### For LLM Applications

- **Training Data**: Use JSONL files to train/fine-tune models
- **RAG Systems**: Feed into retrieval-augmented generation systems
- **Documentation Chatbots**: Build iOS/macOS documentation assistants
- **Code Generation**: Use code examples for training code generation models

### For Researchers

- **API Evolution**: Track changes in Apple's APIs over time
- **Complexity Analysis**: Study API design patterns
- **Framework Analysis**: Research framework relationships and dependencies

## 🔧 Advanced Configuration

### Custom Categories

Add custom categories in `config/config.yaml`:

```yaml
categories:
  - name: "Custom Category"
    url_pattern: "your-pattern"
    description: "Your description"
```

### Visualization Customization

Disable specific visualizations:

```yaml
visualization:
  enable_charts: true
  charts:
    - category_distribution
    # - api_complexity  # Commented out to disable
```

### LLM Chunk Size

Optimize for your specific LLM:

```yaml
output:
  llm_format:
    chunk_size: 2000  # Adjust based on your LLM's context window
```

## 🐛 Troubleshooting

### Common Issues

1. **403 Errors**: Apple may rate-limit requests
   - Solution: Increase `delay_between_requests` in config

2. **Timeout Errors**: Network or server issues
   - Solution: Increase `timeout` value

3. **No Pages Crawled**: URL pattern mismatch
   - Solution: Verify `url_pattern` in category config

4. **Import Errors**: Missing dependencies
   - Solution: `pip install -r requirements.txt`

### Logging

Check logs for detailed error information:

```bash
tail -f logs/crawler.log
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional visualization types
- More sophisticated content extraction
- Integration with specific LLM platforms
- Performance optimizations
- Additional output formats

## 📝 Best Practices

1. **Start with Test Mode**: Always test with `--test` flag first
2. **Respect Rate Limits**: Don't decrease delay below 2 seconds
3. **Monitor Logs**: Check logs during long runs
4. **Backup Raw Data**: The `data/` directory contains valuable raw data
5. **Version Control**: Track configuration changes

## 🔮 Future Enhancements

- [ ] Incremental updates (delta crawling)
- [ ] Multi-language support
- [ ] Interactive web dashboard
- [ ] API endpoint for programmatic access
- [ ] Docker container for easy deployment
- [ ] Integration with vector databases
- [ ] Automatic summarization using LLMs
- [ ] Code example testing/validation

## 📄 License

This project is for educational purposes. Please respect Apple's terms of service and robots.txt when crawling their documentation.

## 🙏 Acknowledgments

- Apple Developer Documentation
- Python async/aiohttp community
- BeautifulSoup for HTML parsing
- NetworkX for graph visualizations

---

**Happy Crawling! 🚀**

For questions or issues, please check the logs or review the configuration file.
