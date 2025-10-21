"""
Detectors module for content detection in images.
"""

from .text_detector import TextDetector
from .table_detector import TableDetector, PaddleTableDetector

__all__ = [
    "TextDetector",
    "TableDetector",
    "PaddleTableDetector",
]
