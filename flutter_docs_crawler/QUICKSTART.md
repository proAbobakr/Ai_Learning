# Quick Start Guide

Get started with the Flutter Documentation Crawler in 5 minutes!

## Step 1: Install Dependencies

### Option A: Using the setup script (Recommended)

```bash
cd flutter_docs_crawler
chmod +x setup.sh
./setup.sh
```

### Option B: Manual installation

```bash
cd flutter_docs_crawler

# Optional: Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Run Your First Crawl

Start with a small test crawl:

```bash
python main.py --max-pages 10
```

This will:
- Crawl 10 pages from Flutter documentation
- Automatically categorize them
- Generate human-readable markdown files
- Create LLM-optimized text files
- Generate visualization charts

**Expected time:** 10-30 seconds

## Step 3: View the Results

### View the index:
```bash
cat output/human_readable/INDEX.md
```

### View a specific category:
```bash
# List available categories
ls output/human_readable/*.md

# View one
cat output/human_readable/widgets.md
```

### View visualizations:
Open in your browser:
- `output/human_readable/charts/category_distribution.html`
- `output/human_readable/charts/content_stats.html`

## Step 4: Use with LLM

### For local LLM (RAG):
```bash
# View LLM metadata
cat output/llm_format/metadata.json

# Use specific category
cat output/llm_format/widgets.txt
```

### For training:
```bash
# Get complete documentation in one file
cat output/llm_format/flutter_docs_complete.txt
```

## Next Steps

### Crawl More Pages

```bash
python main.py --max-pages 100
```

### Customize Categories

Edit `config/crawler_config.yaml` to add or modify categories:

```yaml
categories:
  - name: "Your Custom Category"
    keywords: ["keyword1", "keyword2"]
```

### Re-process Existing Data

```bash
# Update categories without re-crawling
python main.py --skip-crawl
```

## Common Commands

```bash
# Small test crawl
python main.py --max-pages 10

# Medium crawl (recommended)
python main.py --max-pages 100

# Large comprehensive crawl
python main.py --max-pages 500

# Re-process with new categories
python main.py --skip-crawl

# Show help
python main.py --help
```

## Directory Structure After First Run

```
flutter_docs_crawler/
├── output/
│   ├── human_readable/
│   │   ├── INDEX.md              ← Start here!
│   │   ├── widgets.md
│   │   ├── state_management.md
│   │   └── charts/
│   │       ├── category_distribution.html
│   │       └── content_stats.html
│   ├── llm_format/
│   │   ├── metadata.json
│   │   ├── widgets.txt
│   │   └── flutter_docs_complete.txt
│   └── raw_data.json
└── logs/
    └── crawler.log
```

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Network errors
**Solution:** Check your internet connection and try again

### Issue: Slow crawling
**Solution:** This is normal. The crawler respects rate limits. For testing, use `--max-pages 10`

## Tips

1. **Start small** - Use `--max-pages 10` for testing
2. **Check logs** - Useful info in `logs/crawler.log`
3. **Be patient** - Crawling respects server rate limits
4. **Customize** - Edit `config/crawler_config.yaml` for your needs

## Examples

See `examples/example_usage.py` for Python integration examples.

## Documentation

- **README.md** - Full documentation
- **USAGE.md** - Detailed usage guide
- **This file** - Quick start

---

**Ready to explore?** Run your first crawl now! 🚀

```bash
python main.py --max-pages 10
```
