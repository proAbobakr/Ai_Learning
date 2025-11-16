# YouTube Software Engineer Video Crawler

A comprehensive tool to search YouTube for videos about software engineering, extract transcripts, and generate AI-powered summaries.

## Features

- 🔍 **Search YouTube** - Find videos about software engineers with top views
- 📝 **Extract Transcripts** - Automatically extract video transcripts (supports auto-generated and manual captions)
- 🤖 **AI Summaries** - Generate detailed summaries using OpenAI's GPT models
- 📊 **Key Points Extraction** - Identify main topics and key takeaways
- 💾 **Multiple Output Formats** - Save results as JSON, Markdown, and text files
- 🎯 **Flexible Search** - Custom queries or trending video discovery

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (optional, for AI-powered summaries)

### Setup

1. **Install dependencies:**
   ```bash
   cd youtube_crawler
   pip install -r ../requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Note: The crawler works without an API key, but summaries will be basic (first few sentences only).

## Usage

### Basic Usage

Run the crawler with default settings (searches for "software engineer" videos):

```bash
python main.py
```

### Custom Search Query

Search for specific topics:

```bash
python main.py --query "software engineer interview tips"
python main.py --query "day in the life software developer"
python main.py --query "how to become a software engineer"
```

### Adjust Number of Videos

Process more or fewer videos:

```bash
python main.py --max-videos 20
```

### Summary Types

Choose different summary detail levels:

```bash
# Brief summary (2-3 sentences)
python main.py --summary-type brief

# Detailed summary (default, includes key points)
python main.py --summary-type detailed

# Comprehensive analysis (includes everything)
python main.py --summary-type comprehensive
```

### Get Trending Videos

Find trending software engineering videos across multiple topics:

```bash
python main.py --trending --max-videos 15
```

### Custom Output Directory

Specify where to save results:

```bash
python main.py --output-dir ./my_results
```

### Advanced Examples

```bash
# Search for 15 videos about coding interviews with comprehensive summaries
python main.py --query "coding interview preparation" --max-videos 15 --summary-type comprehensive

# Get top 20 trending videos with brief summaries
python main.py --trending --max-videos 20 --summary-type brief

# Search without saving individual transcript files
python main.py --query "python programming" --no-save-transcripts
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | string | "software engineer" | Search query for YouTube |
| `--max-videos` | integer | 10 | Maximum number of videos to process |
| `--summary-type` | choice | "detailed" | Summary type: brief, detailed, or comprehensive |
| `--output-dir` | string | "./output" | Directory to save output files |
| `--trending` | flag | False | Get trending software engineering videos |
| `--no-save-transcripts` | flag | False | Don't save individual transcript files |

## Output Structure

The crawler creates the following directory structure:

```
output/
├── transcripts/
│   ├── VIDEO_ID_1.json          # Transcript data as JSON
│   ├── VIDEO_ID_1.txt           # Transcript as readable text
│   ├── VIDEO_ID_2.json
│   └── VIDEO_ID_2.txt
├── summaries/
│   └── summaries_20231116_143022.md  # All summaries in markdown
└── results_20231116_143022.json      # Complete results as JSON
```

### Output Files

1. **Transcript Files (`transcripts/`):**
   - JSON files contain full transcript data with metadata
   - TXT files contain readable transcript text

2. **Summary Markdown (`summaries/`):**
   - Formatted markdown file with all video summaries
   - Includes title, channel, URL, summary, key points, and topics

3. **Results JSON:**
   - Complete crawl results including:
     - Search metadata
     - Video information (title, channel, views, duration, etc.)
     - Full transcripts
     - Generated summaries

## Module Overview

### `youtube_search.py`
Handles YouTube video search functionality:
- Search by query with various sorting options
- Extract video metadata (views, duration, channel, etc.)
- Support for trending video discovery

### `transcript_extractor.py`
Extracts transcripts from YouTube videos:
- Supports multiple languages
- Handles auto-generated and manual captions
- Batch processing for multiple videos
- Error handling for unavailable transcripts

### `summarizer.py`
Generates AI-powered summaries:
- Three summary types: brief, detailed, comprehensive
- Extracts key points and topics
- Falls back to basic summarization without API key
- Supports batch processing

### `main.py`
Main crawler orchestrator:
- Coordinates all modules
- Handles command-line arguments
- Manages output file creation
- Provides progress feedback

## Examples

### Example 1: Research Software Engineering Careers

```bash
python main.py --query "software engineer career advice" \
               --max-videos 10 \
               --summary-type comprehensive \
               --output-dir ./career_research
```

### Example 2: Learn About Technical Interviews

```bash
python main.py --query "software engineer technical interview" \
               --max-videos 15 \
               --summary-type detailed
```

### Example 3: Quick Trending Overview

```bash
python main.py --trending \
               --max-videos 5 \
               --summary-type brief
```

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test YouTube search
python youtube_search.py

# Test transcript extraction
python transcript_extractor.py

# Test summarization
python summarizer.py
```

## Limitations

1. **Transcript Availability:**
   - Not all videos have transcripts
   - Some creators disable transcripts
   - Language support varies by video

2. **API Rate Limits:**
   - Web scraping has inherent rate limits
   - Add delays between requests for large batches
   - OpenAI API has usage limits based on your plan

3. **Summary Quality:**
   - Quality depends on transcript accuracy
   - Auto-generated transcripts may have errors
   - Without OpenAI API key, summaries are basic

## Troubleshooting

### No transcripts found
- Verify the video has captions enabled
- Try videos from different channels
- Some videos only have non-English transcripts

### OpenAI API errors
- Check your API key is correct in `.env`
- Verify you have API credits available
- Ensure you have access to the required models

### Search returns no results
- Try different search queries
- Check your internet connection
- YouTube's search results may vary

## Future Enhancements

Potential improvements:
- [ ] Support for YouTube playlists
- [ ] Video category filtering
- [ ] Multiple language support
- [ ] Export to additional formats (CSV, PDF)
- [ ] Sentiment analysis of videos
- [ ] Video comparison features
- [ ] Integration with other video platforms

## Contributing

This is part of an AI Learning project. Feel free to:
- Report issues
- Suggest improvements
- Add new features
- Improve documentation

## License

Part of the AI Learning repository.

## Acknowledgments

- Uses `scrapetube` for YouTube search
- Uses `youtube-transcript-api` for transcript extraction
- Uses OpenAI's GPT models for summarization
- Built with LangChain framework

---

**Happy Crawling! 🚀**
