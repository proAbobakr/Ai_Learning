"""
YouTube Software Engineer Video Crawler
Main script to search, extract transcripts, and summarize videos about software engineering
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

from youtube_search import YouTubeSearcher
from transcript_extractor import TranscriptExtractor
from summarizer import TranscriptSummarizer

# Load environment variables
load_dotenv()


class YouTubeCrawler:
    """Main crawler class that orchestrates the entire process"""

    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the YouTube crawler

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        self.searcher = YouTubeSearcher()
        self.extractor = TranscriptExtractor()
        self.summarizer = TranscriptSummarizer()

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "summaries"), exist_ok=True)

    def crawl(
        self,
        query: str = "software engineer",
        max_videos: int = 10,
        summary_type: str = "detailed",
        save_transcripts: bool = True
    ) -> Dict:
        """
        Crawl YouTube for videos and generate summaries

        Args:
            query: Search query
            max_videos: Maximum number of videos to process
            summary_type: Type of summary ('brief', 'detailed', or 'comprehensive')
            save_transcripts: Whether to save transcript files

        Returns:
            Dictionary containing all results
        """
        print("\n" + "="*80)
        print("🎬 YOUTUBE SOFTWARE ENGINEER VIDEO CRAWLER")
        print("="*80)
        print(f"Query: {query}")
        print(f"Max videos: {max_videos}")
        print(f"Summary type: {summary_type}")
        print("="*80 + "\n")

        # Step 1: Search for videos
        print("STEP 1: SEARCHING FOR VIDEOS")
        print("-"*80)
        videos = self.searcher.search_videos(query, max_results=max_videos, sort_by="view_count")

        if not videos:
            print("❌ No videos found. Exiting.")
            return {'videos': [], 'transcripts': {}, 'summaries': []}

        # Step 2: Extract transcripts
        print("\nSTEP 2: EXTRACTING TRANSCRIPTS")
        print("-"*80)
        video_ids = [v['video_id'] for v in videos]
        transcripts = self.extractor.get_batch_transcripts(video_ids)

        # Save transcripts if requested
        if save_transcripts:
            print("\n💾 Saving transcripts...")
            for video_id, transcript_data in transcripts.items():
                if transcript_data:
                    self.extractor.save_transcript(
                        video_id,
                        transcript_data,
                        os.path.join(self.output_dir, "transcripts")
                    )

        # Step 3: Generate summaries
        print("\nSTEP 3: GENERATING SUMMARIES")
        print("-"*80)
        summaries = self.summarizer.summarize_batch(transcripts, videos, summary_type)

        # Step 4: Save results
        print("\nSTEP 4: SAVING RESULTS")
        print("-"*80)
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'total_videos_found': len(videos),
            'videos_with_transcripts': len([t for t in transcripts.values() if t]),
            'summaries_generated': len(summaries),
            'videos': videos,
            'transcripts': {k: v for k, v in transcripts.items() if v},
            'summaries': summaries
        }

        # Save complete results as JSON
        results_file = os.path.join(
            self.output_dir,
            f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved complete results to: {results_file}")

        # Save summaries as markdown
        md_file = os.path.join(
            self.output_dir,
            "summaries",
            f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        self._save_summaries_markdown(summaries, md_file, query)
        print(f"✓ Saved summaries to: {md_file}")

        # Print summary statistics
        print("\n" + "="*80)
        print("📊 CRAWL STATISTICS")
        print("="*80)
        print(f"Total videos found: {len(videos)}")
        print(f"Transcripts extracted: {len([t for t in transcripts.values() if t])}")
        print(f"Summaries generated: {len(summaries)}")
        print(f"Output directory: {self.output_dir}")
        print("="*80 + "\n")

        return results

    def _save_summaries_markdown(self, summaries: List[Dict], output_file: str, query: str):
        """
        Save summaries as a formatted markdown file

        Args:
            summaries: List of summary dictionaries
            output_file: Output file path
            query: Search query used
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# YouTube Video Summaries: {query}\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"**Total Videos:** {len(summaries)}\n\n")
            f.write("---\n\n")

            for i, summary in enumerate(summaries, 1):
                f.write(f"## {i}. {summary['title']}\n\n")
                f.write(f"**Channel:** {summary['channel']}\n\n")
                f.write(f"**URL:** {summary['url']}\n\n")
                f.write(f"**Word Count:** {summary['word_count']:,} words\n\n")

                f.write("### Summary\n\n")
                f.write(f"{summary['summary']}\n\n")

                if summary.get('key_points'):
                    f.write("### Key Points\n\n")
                    for point in summary['key_points']:
                        f.write(f"- {point}\n")
                    f.write("\n")

                if summary.get('topics'):
                    f.write("### Topics\n\n")
                    for topic in summary['topics']:
                        f.write(f"- {topic}\n")
                    f.write("\n")

                f.write("---\n\n")

    def get_trending_videos(self, max_videos: int = 10) -> Dict:
        """
        Get trending software engineering videos

        Args:
            max_videos: Maximum number of videos

        Returns:
            Dictionary containing results
        """
        print("\n" + "="*80)
        print("🔥 GETTING TRENDING SOFTWARE ENGINEER VIDEOS")
        print("="*80 + "\n")

        videos = self.searcher.get_trending_software_videos(max_videos)
        return self.crawl(
            query="trending software engineer videos",
            max_videos=max_videos,
            summary_type="detailed"
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YouTube Software Engineer Video Crawler"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="software engineer",
        help="Search query (default: 'software engineer')"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=10,
        help="Maximum number of videos to process (default: 10)"
    )
    parser.add_argument(
        "--summary-type",
        type=str,
        choices=["brief", "detailed", "comprehensive"],
        default="detailed",
        help="Type of summary to generate (default: detailed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--trending",
        action="store_true",
        help="Get trending software engineering videos"
    )
    parser.add_argument(
        "--no-save-transcripts",
        action="store_true",
        help="Don't save individual transcript files"
    )

    args = parser.parse_args()

    # Create crawler
    crawler = YouTubeCrawler(output_dir=args.output_dir)

    # Run crawler
    if args.trending:
        results = crawler.get_trending_videos(max_videos=args.max_videos)
    else:
        results = crawler.crawl(
            query=args.query,
            max_videos=args.max_videos,
            summary_type=args.summary_type,
            save_transcripts=not args.no_save_transcripts
        )

    print("\n✅ Crawling complete! Check the output directory for results.\n")


if __name__ == "__main__":
    main()
