"""
Video Transcript Summarization Module
Generates summaries of video transcripts using AI
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranscriptSummarizer:
    """Generate summaries of video transcripts using AI"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the summarizer

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            print("⚠️  Warning: No OpenAI API key found. Summary generation will be limited.")
            self.use_openai = False
        else:
            try:
                from langchain.llms import OpenAI
                from langchain.chains.summarize import load_summarize_chain
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain.docstore.document import Document

                self.llm = OpenAI(temperature=0.3, openai_api_key=self.api_key)
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    chunk_overlap=200
                )
                self.use_openai = True
                print("✅ OpenAI summarization enabled")
            except Exception as e:
                print(f"⚠️  Could not initialize OpenAI: {e}")
                self.use_openai = False

    def summarize_transcript(
        self,
        transcript_text: str,
        video_info: Dict,
        summary_type: str = "detailed"
    ) -> Dict:
        """
        Generate a summary of a video transcript

        Args:
            transcript_text: Full transcript text
            video_info: Video metadata dictionary
            summary_type: Type of summary - 'brief', 'detailed', or 'comprehensive'

        Returns:
            Dictionary containing summary and key points
        """
        print(f"🤖 Generating {summary_type} summary...")

        if not transcript_text:
            return {
                'summary': "No transcript available",
                'key_points': [],
                'topics': [],
                'word_count': 0
            }

        if self.use_openai:
            return self._summarize_with_ai(transcript_text, video_info, summary_type)
        else:
            return self._summarize_basic(transcript_text, video_info)

    def _summarize_with_ai(
        self,
        transcript_text: str,
        video_info: Dict,
        summary_type: str
    ) -> Dict:
        """
        Generate AI-powered summary using OpenAI

        Args:
            transcript_text: Full transcript text
            video_info: Video metadata
            summary_type: Type of summary

        Returns:
            Summary dictionary
        """
        from langchain.docstore.document import Document
        from langchain.chains.summarize import load_summarize_chain
        from langchain.prompts import PromptTemplate

        try:
            # Split text into chunks if needed
            chunks = self.text_splitter.split_text(transcript_text)
            docs = [Document(page_content=chunk) for chunk in chunks]

            # Define summary prompts based on type
            if summary_type == "brief":
                prompt_template = """Write a brief 2-3 sentence summary of this software engineering video transcript:

{text}

BRIEF SUMMARY:"""
            elif summary_type == "comprehensive":
                prompt_template = """Analyze this software engineering video transcript and provide:
1. A comprehensive summary (3-4 paragraphs)
2. Key topics discussed
3. Main takeaways
4. Target audience

Transcript:
{text}

COMPREHENSIVE ANALYSIS:"""
            else:  # detailed
                prompt_template = """Summarize this software engineering video transcript in detail.
Include:
- Main topic and purpose
- Key points and concepts discussed
- Practical advice or insights shared
- Conclusion

Transcript:
{text}

DETAILED SUMMARY:"""

            PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

            # Use map_reduce for longer transcripts, stuff for shorter ones
            if len(docs) > 1:
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce",
                    map_prompt=PROMPT,
                    combine_prompt=PROMPT
                )
            else:
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="stuff",
                    prompt=PROMPT
                )

            # Generate summary
            summary = chain.run(docs)

            # Extract key points using AI
            key_points = self._extract_key_points(transcript_text)

            # Extract topics
            topics = self._extract_topics(transcript_text)

            result = {
                'video_id': video_info.get('video_id', ''),
                'title': video_info.get('title', ''),
                'channel': video_info.get('channel', ''),
                'url': video_info.get('url', ''),
                'summary': summary.strip(),
                'key_points': key_points,
                'topics': topics,
                'word_count': len(transcript_text.split()),
                'summary_type': summary_type
            }

            print(f"   ✓ Summary generated successfully")
            return result

        except Exception as e:
            print(f"   ❌ Error generating AI summary: {e}")
            return self._summarize_basic(transcript_text, video_info)

    def _extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """
        Extract key points from transcript using AI

        Args:
            text: Transcript text
            max_points: Maximum number of key points

        Returns:
            List of key points
        """
        try:
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "..."

            prompt = f"""Extract {max_points} key points from this software engineering video transcript.
Format as a numbered list.

Transcript:
{text}

KEY POINTS (one per line, numbered):"""

            response = self.llm(prompt)
            points = [line.strip() for line in response.split('\n') if line.strip() and line.strip()[0].isdigit()]
            return points[:max_points]

        except Exception as e:
            print(f"   ⚠️  Could not extract key points: {e}")
            return []

    def _extract_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """
        Extract main topics from transcript

        Args:
            text: Transcript text
            max_topics: Maximum number of topics

        Returns:
            List of topics
        """
        try:
            # Truncate if too long
            if len(text) > 6000:
                text = text[:6000] + "..."

            prompt = f"""Identify the {max_topics} main topics discussed in this software engineering video.
Return only the topic names, one per line.

Transcript:
{text}

MAIN TOPICS (one per line):"""

            response = self.llm(prompt)
            topics = [line.strip() for line in response.split('\n') if line.strip() and len(line.strip()) > 3]
            return topics[:max_topics]

        except Exception as e:
            print(f"   ⚠️  Could not extract topics: {e}")
            return []

    def _summarize_basic(self, transcript_text: str, video_info: Dict) -> Dict:
        """
        Generate basic summary without AI (fallback method)

        Args:
            transcript_text: Full transcript text
            video_info: Video metadata

        Returns:
            Basic summary dictionary
        """
        # Get first few sentences as summary
        sentences = transcript_text.split('.')[:5]
        summary = '. '.join(sentences) + '.'

        return {
            'video_id': video_info.get('video_id', ''),
            'title': video_info.get('title', ''),
            'channel': video_info.get('channel', ''),
            'url': video_info.get('url', ''),
            'summary': summary,
            'key_points': [],
            'topics': [],
            'word_count': len(transcript_text.split()),
            'summary_type': 'basic'
        }

    def summarize_batch(
        self,
        transcripts: Dict[str, Dict],
        videos_info: List[Dict],
        summary_type: str = "detailed"
    ) -> List[Dict]:
        """
        Generate summaries for multiple transcripts

        Args:
            transcripts: Dictionary mapping video IDs to transcript data
            videos_info: List of video information dictionaries
            summary_type: Type of summary to generate

        Returns:
            List of summary dictionaries
        """
        print(f"\n📊 Generating summaries for {len(transcripts)} videos...")

        summaries = []

        for video_info in videos_info:
            video_id = video_info['video_id']
            transcript_data = transcripts.get(video_id)

            if not transcript_data:
                print(f"\n⚠️  No transcript for {video_info['title'][:50]}...")
                continue

            print(f"\n{'='*80}")
            print(f"Video: {video_info['title'][:70]}...")

            summary = self.summarize_transcript(
                transcript_data['full_text'],
                video_info,
                summary_type
            )

            summaries.append(summary)

        print(f"\n✅ Generated {len(summaries)} summaries\n")
        return summaries


if __name__ == "__main__":
    # Test the summarizer
    summarizer = TranscriptSummarizer()

    # Sample transcript
    sample_transcript = """
    Hello everyone, welcome back to my channel. Today I'm going to talk about
    what it's like to be a software engineer at a big tech company.
    I'll cover the day-to-day responsibilities, the skills you need,
    and some tips for getting hired. Let's dive in!
    """

    sample_video = {
        'video_id': 'test123',
        'title': 'Day in the Life of a Software Engineer',
        'channel': 'Tech Career Channel'
    }

    summary = summarizer.summarize_transcript(sample_transcript, sample_video, "detailed")

    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(summary['summary'])
