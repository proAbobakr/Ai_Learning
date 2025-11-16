# Quick Start Guide

Get up and running with the Apple Documentation Crawler in 5 minutes!

## Step 1: Installation

```bash
# Navigate to project directory
cd 07-projects/05-apple-docs-crawler

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test Run

Start with a quick test to make sure everything works:

```bash
python src/main.py --test
```

This will:
- ✅ Crawl 2 categories (iOS and SwiftUI)
- ✅ Fetch 5 pages from each
- ✅ Generate all visualizations
- ✅ Create both human-readable and LLM-optimized outputs
- ⏱️ Complete in ~2-5 minutes

## Step 3: Check the Output

After the test completes, explore your results:

```bash
# View the master index
cat output/human_readable/INDEX.md

# Browse a category
ls output/human_readable/SwiftUI/

# Check LLM format
ls output/llm_format/
```

## Step 4: View Visualizations

Open the generated charts:

```bash
# On macOS
open output/human_readable/diagrams/category_distribution.png

# On Linux
xdg-open output/human_readable/diagrams/category_distribution.png

# Or browse the directory
ls output/human_readable/diagrams/
```

## Step 5: Run Full Crawl (Optional)

Once you're happy with the test results, run a full crawl:

```bash
python src/main.py
```

**Note**: This will take 30-60 minutes and crawl up to 100 pages per category.

## What You Get

### Human-Readable Output

Perfect for browsing and learning:

```
output/human_readable/
├── INDEX.md                    # Start here!
├── SwiftUI/
│   ├── swiftui_index.md       # Category overview
│   ├── view_protocol.md        # Individual docs
│   └── ...
└── diagrams/
    ├── category_distribution.png
    ├── complexity_analysis.png
    └── framework_network.png
```

### LLM-Optimized Output

Feed directly into your LLM:

```
output/llm_format/
├── dataset_metadata.json       # Dataset info
├── swiftui.json               # Full JSON
├── swiftui.jsonl              # JSON Lines (one doc per line)
├── uikit.json
└── ...
```

## Using LLM Format

### With Local LLM

```python
import json

# Load a category
with open('output/llm_format/swiftui.json') as f:
    docs = json.load(f)

# Feed to your LLM
for doc in docs:
    prompt = f"Explain this Apple API: {doc['title']}\n\n{doc['content']}"
    # response = your_llm.generate(prompt)
```

### With RAG System

```python
# JSONL format is perfect for vector databases
from langchain.document_loaders import JSONLoader

loader = JSONLoader('output/llm_format/swiftui.jsonl')
documents = loader.load()

# Add to your vector store
# vector_store.add_documents(documents)
```

## Customization

### Change Categories

Edit `config/config.yaml`:

```yaml
categories:
  - name: "Combine"
    url_pattern: "combine"
    description: "Reactive programming framework"
```

Then run:

```bash
python src/main.py --test
```

### Adjust Crawl Limits

In `config/config.yaml`:

```yaml
crawler:
  max_pages_per_category: 50  # Reduced from 100
  delay_between_requests: 3   # Slower, more respectful
```

## Troubleshooting

### Problem: No pages crawled

**Solution**: Check your internet connection and Apple's site status.

```bash
# Test connectivity
curl https://developer.apple.com/documentation/
```

### Problem: Import errors

**Solution**: Reinstall dependencies

```bash
pip install -r requirements.txt --force-reinstall
```

### Problem: 403 errors

**Solution**: Increase delay in `config/config.yaml`

```yaml
crawler:
  delay_between_requests: 5  # Slower crawling
```

## Next Steps

1. ✅ Read the full [README.md](README.md)
2. ✅ Explore the generated documentation
3. ✅ Customize categories for your needs
4. ✅ Integrate with your LLM/RAG system
5. ✅ Set up automated crawls for updates

## Tips

💡 **Tip 1**: Start small with `--test` mode before doing full crawls

💡 **Tip 2**: The raw data is saved in `data/` - you can regenerate outputs without re-crawling using `--skip-crawl`

💡 **Tip 3**: Check `logs/crawler.log` if something goes wrong

💡 **Tip 4**: Use the LLM format for training custom models on Apple documentation

💡 **Tip 5**: Visualizations help understand the documentation structure at a glance

---

**Happy crawling! 🚀**

Need help? Check the [README.md](README.md) or review the logs.
