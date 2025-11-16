"""
YouTube Transcript Extraction Module
Extracts transcripts from YouTube videos
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
from typing import List, Dict, Optional
import json


class TranscriptExtractor:
    """Extract transcripts from YouTube videos"""

    def __init__(self):
        """Initialize the transcript extractor"""
        self.api = YouTubeTranscriptApi

    def get_transcript(self, video_id: str, languages: List[str] = None) -> Optional[Dict]:
        """
        Get transcript for a YouTube video

        Args:
            video_id: YouTube video ID
            languages: List of preferred languages (default: ['en'])

        Returns:
            Dictionary containing transcript data or None if unavailable
        """
        if languages is None:
            languages = ['en']

        try:
            print(f"📝 Extracting transcript for video: {video_id}")

            # Try to get transcript in preferred languages
            transcript_list = self.api.list_transcripts(video_id)

            # Try to find transcript in preferred languages
            transcript = None
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except NoTranscriptFound:
                    continue

            # If no transcript found in preferred languages, try auto-generated
            if transcript is None:
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except NoTranscriptFound:
                    print(f"   ⚠️  No transcript available")
                    return None

            # Fetch the actual transcript
            transcript_data = transcript.fetch()

            # Combine all text
            full_text = ' '.join([entry['text'] for entry in transcript_data])

            # Clean up the text
            full_text = self._clean_transcript(full_text)

            result = {
                'video_id': video_id,
                'language': transcript.language_code,
                'is_generated': transcript.is_generated,
                'full_text': full_text,
                'segments': transcript_data[:10],  # Store first 10 segments as sample
                'word_count': len(full_text.split()),
                'char_count': len(full_text)
            }

            print(f"   ✓ Transcript extracted ({result['word_count']} words)")
            return result

        except TranscriptsDisabled:
            print(f"   ⚠️  Transcripts are disabled for this video")
            return None
        except NoTranscriptFound:
            print(f"   ⚠️  No transcript found")
            return None
        except VideoUnavailable:
            print(f"   ⚠️  Video is unavailable")
            return None
        except Exception as e:
            print(f"   ❌ Error extracting transcript: {e}")
            return None

    def get_batch_transcripts(self, video_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Get transcripts for multiple videos

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            Dictionary mapping video IDs to transcript data
        """
        print(f"\n📚 Extracting transcripts for {len(video_ids)} videos...")

        results = {}
        success_count = 0

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}]", end=" ")
            transcript = self.get_transcript(video_id)

            results[video_id] = transcript
            if transcript:
                success_count += 1

        print(f"\n✅ Successfully extracted {success_count}/{len(video_ids)} transcripts\n")
        return results

    def _clean_transcript(self, text: str) -> str:
        """
        Clean up transcript text

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())

        # Remove common artifacts
        text = text.replace('[Music]', '')
        text = text.replace('[Applause]', '')
        text = text.replace('[Laughter]', '')

        return text.strip()

    def save_transcript(self, video_id: str, transcript_data: Dict, output_dir: str = "./transcripts"):
        """
        Save transcript to a file

        Args:
            video_id: YouTube video ID
            transcript_data: Transcript data dictionary
            output_dir: Output directory for transcript files
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(output_dir, f"{video_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        # Save as text
        txt_path = os.path.join(output_dir, f"{video_id}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Video ID: {video_id}\n")
            f.write(f"Language: {transcript_data.get('language', 'unknown')}\n")
            f.write(f"Word Count: {transcript_data.get('word_count', 0)}\n")
            f.write(f"Auto-generated: {transcript_data.get('is_generated', False)}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("TRANSCRIPT:\n")
            f.write("="*80 + "\n\n")
            f.write(transcript_data.get('full_text', ''))

        print(f"   💾 Saved to: {json_path} and {txt_path}")


if __name__ == "__main__":
    # Test the extractor
    extractor = TranscriptExtractor()

    # Test with a sample video ID (replace with actual video ID)
    test_video_id = "dQw4w9WgXcQ"  # Example video ID
    transcript = extractor.get_transcript(test_video_id)

    if transcript:
        print("\n" + "="*80)
        print("TRANSCRIPT SAMPLE:")
        print("="*80)
        print(f"Video ID: {transcript['video_id']}")
        print(f"Language: {transcript['language']}")
        print(f"Auto-generated: {transcript['is_generated']}")
        print(f"Word Count: {transcript['word_count']}")
        print(f"\nFirst 500 characters:")
        print(transcript['full_text'][:500] + "...")
