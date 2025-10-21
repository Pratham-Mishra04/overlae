"""
Utilities module for agentOS image analysis.
"""

from .types import Rect, Block, Predicates, AnalysisResult
from .protocols import OCRProvider, Detector
from .ocr import TesseractOCR
from .rules import TaskRulesEngine, default_rules
from .analyzer import Analyzer

__all__ = [
    "Rect",
    "Block",
    "Predicates",
    "AnalysisResult",
    "OCRProvider",
    "Detector",
    "TesseractOCR",
    "TaskRulesEngine",
    "default_rules",
    "Analyzer",
]
