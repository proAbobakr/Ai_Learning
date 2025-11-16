"""
Apple Documentation Crawler System
A comprehensive system for crawling, categorizing, and formatting Apple developer documentation
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .crawler import AppleDocsCrawler, DocumentationPage
from .categorizer import DocumentationCategorizer, CategoryStats
from .visualizer import DocumentationVisualizer
from .formatters import HumanReadableFormatter, LLMFormatter

__all__ = [
    'AppleDocsCrawler',
    'DocumentationPage',
    'DocumentationCategorizer',
    'CategoryStats',
    'DocumentationVisualizer',
    'HumanReadableFormatter',
    'LLMFormatter',
]
