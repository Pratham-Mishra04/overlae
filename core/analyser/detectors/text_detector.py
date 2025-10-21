"""
Text detection implementation.
"""

from typing import Dict, Any, List, Tuple

if False:  # TYPE_CHECKING
    from PIL import Image

from utils.types import Block, Rect
from utils.protocols import Detector


class TextDetector(Detector):
    """Detector for text content with noise filtering."""

    def __init__(self, min_text_length: int = 3, min_word_count: int = 2):
        self.min_text_length = min_text_length
        self.min_word_count = min_word_count

    def detect(
        self, image: "Image.Image", ocr: Dict[str, Any]
    ) -> Tuple[List[Block], Dict[str, Any]]:
        """Detect text blocks in image using OCR results."""
        blocks: List[Block] = []
        lines = ocr.get("lines", [])
        valid_lines = []

        for idx, ln in enumerate(lines):
            text = ln.get("text", "").strip()

            # Filter out meaningless text
            if self._is_valid_text(text):
                x, y, w, h = ln["bbox"]
                blocks.append(
                    Block(
                        id=f"line_{idx}",
                        kind="line",
                        bbox=Rect(x, y, w, h),
                        text=text,
                    )
                )
                valid_lines.append(ln)

        # More sophisticated has_text logic
        has_meaningful_text = self._has_meaningful_text(valid_lines)

        return blocks, {
            "has_text": has_meaningful_text,
            "text_line_count": len(valid_lines),
            "total_detected_lines": len(lines),
        }

    def _is_valid_text(self, text: str) -> bool:
        """Filter out noise, gibberish, and meaningless OCR detections."""
        if not text or len(text) < self.min_text_length:
            return False

        # Remove whitespace and special characters for analysis
        clean_text = "".join(c for c in text if c.isalnum())
        if len(clean_text) < 2:
            return False

        # Filter out obvious OCR noise (mostly repeating characters)
        if len(set(clean_text.lower())) < 2:  # Less than 2 unique characters
            return False

        # Filter out strings that are mostly special characters or numbers only
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count == 0 and len(text) < 10:  # No letters and short
            return False

        return True

    def _has_meaningful_text(self, lines: List[Dict[str, Any]]) -> bool:
        """Determine if the detected lines constitute meaningful text content."""
        if not lines:
            return False

        # Count total words across all lines
        total_words = 0
        for line in lines:
            text = line.get("text", "")
            words = text.split()
            # Filter out single-character "words" and obvious noise
            valid_words = [w for w in words if len(w) > 1 or w.isalpha()]
            total_words += len(valid_words)

        # Require minimum word count for meaningful text
        return total_words >= self.min_word_count
