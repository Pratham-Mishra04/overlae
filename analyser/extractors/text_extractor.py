"""
Text extractor implementation.
"""

from typing import Dict, Any, List

if False:  # TYPE_CHECKING
    from PIL import Image

from utils.types import Block


class TextExtractor:
    """Extractor for extracting structured text data from detected text blocks."""

    def __init__(self):
        pass

    def extract(
        self, image: "Image.Image", ocr: Dict[str, Any], text_blocks: List[Block]
    ) -> Dict[str, Any]:
        """Extract structured text metadata from detected text blocks."""
        lines = [ln["text"] for ln in ocr.get("lines", []) if ln.get("text")]
        full_text = "\n".join(lines).strip()
        word_count = sum(len(ln.split()) for ln in lines)

        return {
            "text": {
                "full_text": full_text,
                "line_count": len(lines),
                "word_count": word_count,
                "blocks_found": len(text_blocks),
            }
        }
