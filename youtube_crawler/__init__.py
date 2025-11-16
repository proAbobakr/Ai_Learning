"""
YouTube Software Engineer Video Crawler
A tool to search, extract transcripts, and summarize videos about software engineering
"""

from .youtube_search import YouTubeSearcher
from .transcript_extractor import TranscriptExtractor
from .summarizer import TranscriptSummarizer

__version__ = "1.0.0"
__all__ = ["YouTubeSearcher", "TranscriptExtractor", "TranscriptSummarizer"]
