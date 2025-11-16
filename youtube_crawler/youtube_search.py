"""
YouTube Video Search Module
Searches for videos about software engineers with top views
"""

import scrapetube
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class YouTubeSearcher:
    """Search YouTube for videos about software engineers"""

    def __init__(self):
        """Initialize the YouTube searcher"""
        pass

    def search_videos(
        self,
        query: str = "software engineer",
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> List[Dict]:
        """
        Search for YouTube videos

        Args:
            query: Search query (default: "software engineer")
            max_results: Maximum number of results to return
            sort_by: Sort order - 'relevance', 'upload_date', 'view_count', 'rating'

        Returns:
            List of video information dictionaries
        """
        print(f"🔍 Searching YouTube for: '{query}'")
        print(f"   Max results: {max_results}, Sort by: {sort_by}")

        videos = []

        try:
            # Use scrapetube to search YouTube
            video_generator = scrapetube.get_search(
                query=query,
                limit=max_results,
                sort_by=sort_by
            )

            for video in video_generator:
                video_data = self._extract_video_info(video)
                if video_data:
                    videos.append(video_data)
                    print(f"   ✓ Found: {video_data['title'][:60]}...")

            # Sort by view count if we have that information
            videos.sort(key=lambda x: x.get('view_count', 0), reverse=True)

            print(f"\n✅ Found {len(videos)} videos\n")
            return videos

        except Exception as e:
            print(f"❌ Error searching YouTube: {e}")
            return []

    def _extract_video_info(self, video: Dict) -> Optional[Dict]:
        """
        Extract relevant information from video data

        Args:
            video: Raw video data from scrapetube

        Returns:
            Cleaned video information dictionary
        """
        try:
            video_id = video.get('videoId')
            if not video_id:
                return None

            # Extract view count from viewCountText if available
            view_count = 0
            view_count_text = video.get('viewCountText', {})
            if isinstance(view_count_text, dict):
                simple_text = view_count_text.get('simpleText', '0')
                # Extract number from text like "1.2M views" or "10,000 views"
                view_count = self._parse_view_count(simple_text)

            # Extract duration
            duration = video.get('lengthText', {}).get('simpleText', 'Unknown')

            # Extract title
            title_runs = video.get('title', {}).get('runs', [])
            title = title_runs[0].get('text', 'Unknown Title') if title_runs else 'Unknown Title'

            # Extract channel name
            owner_text = video.get('ownerText', {}).get('runs', [])
            channel = owner_text[0].get('text', 'Unknown Channel') if owner_text else 'Unknown Channel'

            # Extract published time
            published_time = video.get('publishedTimeText', {}).get('simpleText', 'Unknown')

            return {
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'duration': duration,
                'view_count': view_count,
                'published_time': published_time,
            }

        except Exception as e:
            print(f"⚠️  Error extracting video info: {e}")
            return None

    def _parse_view_count(self, view_text: str) -> int:
        """
        Parse view count from text like '1.2M views' or '10,000 views'

        Args:
            view_text: View count text

        Returns:
            Integer view count
        """
        try:
            # Remove 'views' and clean
            text = view_text.lower().replace('views', '').replace('view', '').strip()
            text = text.replace(',', '')

            # Handle K, M, B suffixes
            multiplier = 1
            if 'k' in text:
                multiplier = 1000
                text = text.replace('k', '')
            elif 'm' in text:
                multiplier = 1000000
                text = text.replace('m', '')
            elif 'b' in text:
                multiplier = 1000000000
                text = text.replace('b', '')

            # Convert to number
            number = float(text)
            return int(number * multiplier)

        except Exception:
            return 0

    def get_trending_software_videos(self, max_results: int = 10) -> List[Dict]:
        """
        Get trending videos about software engineering

        Args:
            max_results: Maximum number of results

        Returns:
            List of trending video information
        """
        queries = [
            "software engineer day in the life",
            "software engineering career",
            "how to become software engineer",
            "software developer interview",
            "coding tutorial"
        ]

        all_videos = []

        for query in queries:
            videos = self.search_videos(query, max_results=max_results // len(queries) + 1)
            all_videos.extend(videos)

        # Remove duplicates based on video_id
        seen = set()
        unique_videos = []
        for video in all_videos:
            if video['video_id'] not in seen:
                seen.add(video['video_id'])
                unique_videos.append(video)

        # Sort by view count and return top results
        unique_videos.sort(key=lambda x: x.get('view_count', 0), reverse=True)
        return unique_videos[:max_results]


if __name__ == "__main__":
    # Test the searcher
    searcher = YouTubeSearcher()
    videos = searcher.search_videos("software engineer", max_results=5)

    print("\n" + "="*80)
    print("TOP VIDEOS:")
    print("="*80)

    for i, video in enumerate(videos, 1):
        print(f"\n{i}. {video['title']}")
        print(f"   Channel: {video['channel']}")
        print(f"   URL: {video['url']}")
        print(f"   Views: {video['view_count']:,}")
        print(f"   Duration: {video['duration']}")
        print(f"   Published: {video['published_time']}")
