"""
Demo script showing how to use the YouTube crawler
This demonstrates the API without requiring actual YouTube access
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_search():
    """Demonstrate YouTube search"""
    print("\n" + "="*80)
    print("DEMO: YouTube Video Search")
    print("="*80)
    print("\nThis shows how to search for software engineer videos:\n")

    print("```python")
    print("from youtube_search import YouTubeSearcher")
    print("")
    print("searcher = YouTubeSearcher()")
    print("videos = searcher.search_videos('software engineer', max_results=10)")
    print("")
    print("for video in videos:")
    print("    print(f\"Title: {video['title']}\")")
    print("    print(f\"Channel: {video['channel']}\")")
    print("    print(f\"Views: {video['view_count']:,}\")")
    print("    print(f\"URL: {video['url']}\")")
    print("```")


def demo_transcript():
    """Demonstrate transcript extraction"""
    print("\n" + "="*80)
    print("DEMO: Transcript Extraction")
    print("="*80)
    print("\nThis shows how to extract transcripts from videos:\n")

    print("```python")
    print("from transcript_extractor import TranscriptExtractor")
    print("")
    print("extractor = TranscriptExtractor()")
    print("transcript = extractor.get_transcript('VIDEO_ID_HERE')")
    print("")
    print("if transcript:")
    print("    print(f\"Transcript: {transcript['full_text'][:500]}...\")")
    print("    print(f\"Word count: {transcript['word_count']}\")")
    print("```")


def demo_summary():
    """Demonstrate summarization"""
    print("\n" + "="*80)
    print("DEMO: AI Summarization")
    print("="*80)
    print("\nThis shows how to generate summaries:\n")

    print("```python")
    print("from summarizer import TranscriptSummarizer")
    print("")
    print("summarizer = TranscriptSummarizer()")
    print("summary = summarizer.summarize_transcript(")
    print("    transcript_text=transcript['full_text'],")
    print("    video_info=video,")
    print("    summary_type='detailed'")
    print(")")
    print("")
    print("print(summary['summary'])")
    print("print('Key Points:', summary['key_points'])")
    print("```")


def demo_full_workflow():
    """Demonstrate complete workflow"""
    print("\n" + "="*80)
    print("DEMO: Complete Workflow")
    print("="*80)
    print("\nThis shows the complete crawling workflow:\n")

    print("```python")
    print("from main import YouTubeCrawler")
    print("")
    print("# Create crawler")
    print("crawler = YouTubeCrawler(output_dir='./output')")
    print("")
    print("# Run complete crawl")
    print("results = crawler.crawl(")
    print("    query='software engineer interview tips',")
    print("    max_videos=10,")
    print("    summary_type='detailed'")
    print(")")
    print("")
    print("# Results are saved to:")
    print("# - ./output/transcripts/  (individual transcript files)")
    print("# - ./output/summaries/    (markdown summary file)")
    print("# - ./output/results_*.json (complete JSON results)")
    print("```")


def demo_cli():
    """Demonstrate CLI usage"""
    print("\n" + "="*80)
    print("DEMO: Command-Line Usage")
    print("="*80)
    print("\nYou can run the crawler from the command line:\n")

    print("# Basic search")
    print("python main.py --query 'software engineer' --max-videos 10")
    print("")

    print("# Get trending videos")
    print("python main.py --trending --max-videos 15")
    print("")

    print("# Comprehensive summaries")
    print("python main.py --query 'coding interview' --summary-type comprehensive")
    print("")

    print("# Custom output directory")
    print("python main.py --output-dir ./my_results")


def show_example_output():
    """Show example output structure"""
    print("\n" + "="*80)
    print("EXAMPLE: Output Structure")
    print("="*80)
    print("\nThe crawler generates the following files:\n")

    print("output/")
    print("├── transcripts/")
    print("│   ├── VIDEO_ID_1.json    # Transcript data")
    print("│   ├── VIDEO_ID_1.txt     # Readable transcript")
    print("│   └── ...")
    print("├── summaries/")
    print("│   └── summaries_TIMESTAMP.md  # All summaries")
    print("└── results_TIMESTAMP.json      # Complete results")
    print("")

    print("\nExample summary output:\n")
    print("---")
    print("## 1. Day in the Life of a Software Engineer at Google")
    print("")
    print("**Channel:** Tech Career Tips")
    print("**URL:** https://youtube.com/watch?v=...")
    print("**Word Count:** 2,543 words")
    print("")
    print("### Summary")
    print("This video provides an inside look at the daily routine of a")
    print("software engineer working at Google...")
    print("")
    print("### Key Points")
    print("- Start day with standup meetings at 9 AM")
    print("- Spend most time on coding and code reviews")
    print("- Work-life balance is important")
    print("- Use cutting-edge technologies")
    print("")
    print("### Topics")
    print("- Daily routine")
    print("- Tech stack")
    print("- Team collaboration")
    print("---")


def main():
    """Run demos"""
    print("\n" + "="*80)
    print("🎬 YOUTUBE CRAWLER - DEMO & USAGE EXAMPLES")
    print("="*80)
    print("\nThis demonstrates how to use the YouTube Software Engineer Crawler")
    print("to search for videos, extract transcripts, and generate summaries.")

    demo_search()
    demo_transcript()
    demo_summary()
    demo_full_workflow()
    demo_cli()
    show_example_output()

    print("\n" + "="*80)
    print("📚 GETTING STARTED")
    print("="*80)
    print("\n1. Set up your OpenAI API key in .env file")
    print("2. Run: python main.py --query 'your search query'")
    print("3. Check the ./output directory for results")
    print("\nFor more information, see README.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
