"""
Core data types for agentOS image analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Rect:
    """Rectangle representing a bounding box."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Block:
    """A detected block (text line, table, etc.) in an image."""

    id: str
    kind: str  # "line", "table", etc.
    bbox: Rect
    text: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Predicates:
    """Boolean predicates about image content."""

    has_text: bool = False
    has_table: bool = False


@dataclass
class AnalysisResult:
    """Complete analysis result for an image."""

    meta: Dict[str, Any]
    predicates: Predicates
    blocks: List[Block]
    eligible_tasks: List[str] = field(default_factory=list)
    rationale: Dict[str, List[str]] = field(default_factory=dict)
