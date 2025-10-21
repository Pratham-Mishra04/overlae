"""
Protocol definitions for agentOS components.
"""

from typing import Dict, Any, List, Tuple, Protocol, runtime_checkable

if False:  # TYPE_CHECKING
    from PIL import Image

from .types import Block


@runtime_checkable
class OCRProvider(Protocol):
    """Protocol for OCR providers."""

    def run_ocr(self, image: "Image.Image") -> Dict[str, Any]:
        """
        Run OCR on an image.

        Returns:
        {
          "lines": [{"text": "...", "bbox": [x,y,w,h]}],
          "words": [{"text": "...", "bbox": [x,y,w,h], "line_index": int}]
        }
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """Protocol for content detectors."""

    def detect(
        self, image: "Image.Image", ocr: Dict[str, Any]
    ) -> Tuple[List[Block], Dict[str, Any]]:
        """
        Detect content in an image.

        Args:
            image: PIL Image
            ocr: OCR results from OCRProvider

        Returns:
            Tuple of (detected_blocks, detection_metadata)
        """
        ...
