"""
Quick test script for the YouTube crawler
Tests basic functionality without requiring API keys
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from youtube_search import YouTubeSearcher
from transcript_extractor import TranscriptExtractor


def test_search():
    """Test YouTube search functionality"""
    print("\n" + "="*80)
    print("TEST 1: YouTube Search")
    print("="*80)

    searcher = YouTubeSearcher()
    videos = searcher.search_videos("software engineer", max_results=3)

    if videos:
        print(f"\n✅ Search test PASSED - Found {len(videos)} videos")
        print("\nSample video:")
        print(f"  Title: {videos[0]['title']}")
        print(f"  Channel: {videos[0]['channel']}")
        print(f"  Views: {videos[0]['view_count']:,}")
        return True
    else:
        print("\n❌ Search test FAILED - No videos found")
        return False


def test_transcript():
    """Test transcript extraction"""
    print("\n" + "="*80)
    print("TEST 2: Transcript Extraction")
    print("="*80)

    # First get a video
    searcher = YouTubeSearcher()
    videos = searcher.search_videos("software engineer", max_results=5)

    if not videos:
        print("❌ Cannot test transcript - no videos found")
        return False

    # Try to get transcript for first few videos
    extractor = TranscriptExtractor()

    for video in videos[:5]:
        print(f"\nTrying video: {video['title'][:50]}...")
        transcript = extractor.get_transcript(video['video_id'])

        if transcript:
            print(f"✅ Transcript test PASSED")
            print(f"  Video ID: {transcript['video_id']}")
            print(f"  Language: {transcript['language']}")
            print(f"  Word count: {transcript['word_count']}")
            print(f"  Preview: {transcript['full_text'][:200]}...")
            return True

    print("\n❌ Transcript test FAILED - No transcripts available for any video")
    return False


def test_basic_summary():
    """Test basic summarization (without API key)"""
    print("\n" + "="*80)
    print("TEST 3: Basic Summarization")
    print("="*80)

    from summarizer import TranscriptSummarizer

    summarizer = TranscriptSummarizer()

    sample_text = """
    Hello everyone! Today I'm going to talk about what it's like to work as a
    software engineer at a major tech company. I'll cover the daily routine,
    the technologies we use, team collaboration, and some tips for those
    interested in this career path. Let's start with a typical day.

    Most software engineers start their day around 9 AM with a standup meeting.
    This is where the team discusses what they worked on yesterday, what they
    plan to work on today, and any blockers they're facing. These meetings
    typically last 15 minutes.

    After standup, engineers typically spend the bulk of their day coding,
    reviewing code, and attending meetings. Code reviews are an important part
    of maintaining code quality and sharing knowledge across the team.
    """

    video_info = {
        'video_id': 'test',
        'title': 'Day in the Life of a Software Engineer',
        'channel': 'Tech Career',
        'url': 'https://youtube.com/watch?v=test'
    }

    summary = summarizer.summarize_transcript(sample_text, video_info, "brief")

    if summary and summary['summary']:
        print(f"✅ Summarization test PASSED")
        print(f"\nSummary:")
        print(f"  {summary['summary']}")
        return True
    else:
        print("❌ Summarization test FAILED")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 YOUTUBE CRAWLER TEST SUITE")
    print("="*80)
    print("\nThis will test the basic functionality of the crawler.")
    print("Note: OpenAI API key not required for these tests.\n")

    results = {
        'search': False,
        'transcript': False,
        'summary': False
    }

    try:
        results['search'] = test_search()
    except Exception as e:
        print(f"\n❌ Search test ERROR: {e}")

    try:
        results['transcript'] = test_transcript()
    except Exception as e:
        print(f"\n❌ Transcript test ERROR: {e}")

    try:
        results['summary'] = test_basic_summary()
    except Exception as e:
        print(f"\n❌ Summary test ERROR: {e}")

    # Print final results
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    passed = sum(results.values())
    total = len(results)

    for test_name, passed_status in results.items():
        status = "✅ PASSED" if passed_status else "❌ FAILED"
        print(f"{test_name.title()}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80 + "\n")

    if passed == total:
        print("🎉 All tests passed! The crawler is working correctly.\n")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
