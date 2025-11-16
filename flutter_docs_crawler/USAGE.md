# Flutter Documentation Crawler - Usage Guide

## Quick Start

### 1. Basic Crawl (Recommended for First Run)

```bash
# Crawl 50 pages and generate all outputs
python main.py --max-pages 50
```

This will:
- Crawl 50 pages from docs.flutter.dev
- Categorize the content automatically
- Generate human-readable markdown files
- Generate LLM-optimized text files
- Create visualization charts

### 2. View Results

After crawling:

```bash
# View the main index
cat output/human_readable/INDEX.md

# View a specific category (e.g., Widgets)
cat output/human_readable/widgets.md

# View LLM metadata
cat output/llm_format/metadata.json
```

### 3. Use with Local LLM

```bash
# For RAG applications - load specific categories
cat output/llm_format/widgets.txt

# For training - use the complete file
cat output/llm_format/flutter_docs_complete.txt
```

## Common Workflows

### Workflow 1: Building a Flutter Reference Guide

```bash
# Crawl comprehensive documentation
python main.py --max-pages 300

# Open the index in your browser/editor
# The INDEX.md has links to all categories
```

### Workflow 2: Preparing Data for LLM Training

```bash
# Crawl with maximum coverage
python main.py --max-pages 500

# Combine all LLM format files
cat output/llm_format/*.txt > my_training_data.txt

# Or use the pre-combined file
cp output/llm_format/flutter_docs_complete.txt my_training_data.txt
```

### Workflow 3: Category-Specific Crawling

```bash
# 1. Crawl broadly
python main.py --max-pages 200

# 2. Edit config to focus on specific categories
nano config/crawler_config.yaml

# 3. Re-process with new categories
python main.py --skip-crawl
```

### Workflow 4: Incremental Updates

```bash
# First crawl
python main.py --max-pages 100

# Later, crawl more pages
python main.py --max-pages 200

# Re-process everything
python main.py --skip-crawl
```

## Integration Examples

### Example 1: Use with Local LLM (Python)

```python
import json

# Load metadata
with open('output/llm_format/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load specific category
category = 'widgets'
with open(f'output/llm_format/{category}.txt', 'r') as f:
    docs = f.read()

# Feed to your LLM
prompt = f"""Based on this Flutter documentation:

{docs[:4000]}  # Use chunk for context window

Question: How do I create a StatefulWidget?
"""

# Use with your LLM API
# response = your_llm.generate(prompt)
```

### Example 2: RAG System

```python
from pathlib import Path
import json

class FlutterDocsRAG:
    def __init__(self, docs_path='output/llm_format'):
        self.docs_path = Path(docs_path)
        self.load_metadata()

    def load_metadata(self):
        with open(self.docs_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

    def get_relevant_docs(self, query):
        # Simple keyword matching
        relevant_categories = []

        for category, info in self.metadata['categories'].items():
            if any(keyword in query.lower() for keyword in category.lower().split()):
                relevant_categories.append(category)

        # Load relevant category files
        docs = []
        for category in relevant_categories:
            filename = info['file']
            with open(self.docs_path / filename, 'r') as f:
                docs.append(f.read())

        return '\n\n'.join(docs)

# Usage
rag = FlutterDocsRAG()
context = rag.get_relevant_docs("How do I handle state in widgets?")
```

### Example 3: Search Functionality

```python
import json
from pathlib import Path

def search_docs(query, docs_dir='output/llm_format'):
    """Search through all documentation"""
    results = []

    for file_path in Path(docs_dir).glob('*.txt'):
        if file_path.name == 'flutter_docs_complete.txt':
            continue  # Skip combined file

        with open(file_path, 'r') as f:
            content = f.read()

        # Simple search
        if query.lower() in content.lower():
            # Find context around match
            idx = content.lower().find(query.lower())
            context = content[max(0, idx-200):idx+200]

            results.append({
                'file': file_path.name,
                'context': context
            })

    return results

# Usage
results = search_docs("setState")
for result in results:
    print(f"\nFound in: {result['file']}")
    print(f"Context: ...{result['context']}...")
```

## Configuration Tips

### Customize Categories

Edit `config/crawler_config.yaml`:

```yaml
categories:
  - name: "My Custom Category"
    keywords: ["custom", "special", "unique"]
```

### Adjust Crawling Behavior

```yaml
crawler:
  max_depth: 3              # Less deep = faster
  delay_between_requests: 2.0  # Higher = slower but more polite
  timeout: 60               # Increase for slow connections
```

### Optimize for LLM

```yaml
output:
  llm_format:
    max_chunk_size: 8000    # Adjust for your LLM's context window
    include_metadata: true   # Include source URLs and categories
```

## Performance Tuning

### For Speed

```bash
# Reduce depth and increase speed
# Edit config: max_depth: 2, delay: 0.5
python main.py --max-pages 100
```

### For Comprehensiveness

```bash
# Increase depth and coverage
# Edit config: max_depth: 7
python main.py --max-pages 1000
```

### For Memory Efficiency

```bash
# Process in batches
python main.py --max-pages 100
python main.py --max-pages 200
# etc.
```

## Output Formats Explained

### Human-Readable Output

```
output/human_readable/
├── INDEX.md                 # Start here!
├── widgets.md              # One file per category
├── state_management.md
└── charts/
    ├── category_distribution.html
    └── content_stats.html
```

**Best for:** Reading, reference, documentation websites

### LLM-Optimized Output

```
output/llm_format/
├── metadata.json           # Overview and stats
├── widgets.txt             # Category files
├── widgets_chunks/         # Large categories split
│   ├── chunk_001.txt
│   └── chunk_002.txt
└── flutter_docs_complete.txt  # Everything combined
```

**Best for:** LLM training, RAG systems, embeddings

## Troubleshooting Common Issues

### Issue: Not enough content in category

**Solution:**
```yaml
# Add more keywords to the category in config
categories:
  - name: "Widgets"
    keywords: ["widget", "ui", "component", "element", "view"]
```

### Issue: Too many pages in "Other" category

**Solution:** Add more specific categories or refine keywords

### Issue: Slow crawling

**Solution:**
```bash
# Reduce max-pages for testing
python main.py --max-pages 20

# Or adjust delay in config
delay_between_requests: 0.5  # Minimum recommended
```

### Issue: Want to re-categorize without re-crawling

**Solution:**
```bash
# Edit categories in config, then:
python main.py --skip-crawl
```

## Advanced Usage

### Custom Output Directory

```python
# In main.py or custom script
human_formatter = HumanReadableFormatter(output_dir="my_custom_output/human")
llm_formatter = LLMFormatter(output_dir="my_custom_output/llm")
```

### Filtering by URL Pattern

```python
# In crawler.py, modify is_valid_url()
def is_valid_url(self, url: str) -> bool:
    # Only crawl cookbook section
    if '/cookbook/' not in url:
        return False
    return super().is_valid_url(url)
```

### Export to Other Formats

```python
# Add to main.py
import markdown
from weasyprint import HTML

# Convert markdown to PDF
for md_file in Path('output/human_readable').glob('*.md'):
    html = markdown.markdown(md_file.read_text())
    HTML(string=html).write_pdf(md_file.with_suffix('.pdf'))
```

## Tips for Best Results

1. **Start Small**: Test with `--max-pages 20` first
2. **Review Categories**: Check output and adjust config as needed
3. **Incremental Crawling**: Crawl more pages over time
4. **Regular Updates**: Re-crawl periodically as Flutter docs update
5. **Backup Raw Data**: Keep `raw_data.json` for re-processing

## Next Steps

- Explore the example scripts in `examples/`
- Customize categories for your needs
- Integrate with your LLM workflow
- Set up automated crawling (cron job)
- Build search/indexing on top of the data

Happy crawling! 🚀
