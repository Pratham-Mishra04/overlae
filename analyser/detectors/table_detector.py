"""
Table detection implementations.
"""

import os
import tempfile
from typing import Dict, Any, List, Tuple, Optional

if False:  # TYPE_CHECKING
    from PIL import Image

# Optional deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from paddleocr import PPStructure  # type: ignore
except Exception:
    PPStructure = None

from utils.types import Block, Rect
from utils.protocols import Detector


class TableDetector(Detector):
    """
    Dual-strategy table detector:
      A) OpenCV ruling detection (if cv2 available)
      B) OCR layout heuristic (column clustering)
    """

    def __init__(self, cv_min_conf: float = 0.6, ocr_min_conf: float = 0.7):
        self.cv_min_conf = cv_min_conf
        self.ocr_min_conf = ocr_min_conf

    def detect(
        self, image: "Image.Image", ocr: Dict[str, Any]
    ) -> Tuple[List[Block], Dict[str, Any]]:
        """Detect tables using OpenCV + OCR heuristics."""
        tables: List[Block] = []
        cv_conf, cv_boxes = 0.0, []

        if cv2 is not None:
            try:
                cv_conf, cv_boxes = self._cv_detect_tables(image)
                for i, (x, y, w, h) in enumerate(cv_boxes):
                    tables.append(
                        Block(id=f"table_cv_{i}", kind="table", bbox=Rect(x, y, w, h))
                    )
            except Exception:
                pass

        ocr_conf, ocr_boxes = self._ocr_layout_detect(ocr)
        for i, (x, y, w, h) in enumerate(ocr_boxes):
            tables.append(
                Block(id=f"table_ocr_{i}", kind="table", bbox=Rect(x, y, w, h))
            )

        has_table = (
            (cv_conf >= self.cv_min_conf)
            or (ocr_conf >= self.ocr_min_conf)
            or (len(tables) > 0)
        )
        return tables, {"has_table": bool(has_table)}

    def _cv_detect_tables(
        self, image: "Image.Image"
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """OpenCV-based table detection using morphological operations."""
        img = image.convert("L")
        arr = cv2.cvtColor(cv2.imread(self._pil_to_temp(img)), cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8
        )
        horiz = cv2.morphologyEx(
            thr,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
            iterations=2,
        )
        vert = cv2.morphologyEx(
            thr,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)),
            iterations=2,
        )
        grid = cv2.add(horiz, vert)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 80 and h > 40:
                boxes.append((x, y, w, h))
        density = cv2.countNonZero(grid) / max(1, arr.shape[0] * arr.shape[1])
        conf = min(1.0, 0.3 * len(boxes) + 5.0 * density)
        return conf, boxes

    @staticmethod
    def _pil_to_temp(img: "Image.Image") -> str:
        """Save PIL image to temporary file."""
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(path, format="PNG")
        return path

    def _ocr_layout_detect(
        self, ocr: Dict[str, Any]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """OCR-based table detection using column clustering."""
        lines = ocr.get("lines", [])
        if not lines or len(lines) < 8:
            return 0.0, []
        xs = [ln["bbox"][0] for ln in lines]
        xs.sort()
        # bucket left edges
        buckets: List[List[int]] = []
        for x in xs:
            placed = False
            for b in buckets:
                if abs(b[-1] - x) < 18:
                    b.append(x)
                    placed = True
                    break
            if not placed:
                buckets.append([x])
        cols = [b for b in buckets if len(b) >= 3]
        if len(cols) >= 3:
            # region that spans all lines
            min_x = min(ln["bbox"][0] for ln in lines)
            min_y = min(ln["bbox"][1] for ln in lines)
            max_x = max(ln["bbox"][0] + ln["bbox"][2] for ln in lines)
            max_y = max(ln["bbox"][1] + ln["bbox"][3] for ln in lines)
            return min(1.0, 0.2 * len(cols)), [
                (min_x, min_y, max_x - min_x, max_y - min_y)
            ]
        return 0.0, []


class PaddleTableDetector(Detector):
    """
    Enhanced table detector using PaddleOCR's PPStructure.
    Falls back to TableDetector if PaddleOCR is not available.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        min_confidence: float = 0.7,
        min_table_area: int = 2000,  # Minimum table area in pixels
        min_cells: int = 4,  # Minimum number of cells to be considered a table
    ):
        self.use_gpu = use_gpu
        self.lang = lang
        self.min_confidence = min_confidence
        self.min_table_area = min_table_area
        self.min_cells = min_cells
        self._paddle_engine = None
        self._fallback_detector = None

        if PPStructure is not None:
            try:
                # Initialize PaddleOCR structure engine
                self._paddle_engine = PPStructure(
                    use_gpu=use_gpu,
                    lang=lang,
                    layout=False,  # We only want table detection
                    show_log=False,
                )
            except Exception:
                self._paddle_engine = None

        # Fallback to original detector if PaddleOCR fails
        if self._paddle_engine is None:
            self._fallback_detector = TableDetector()

    def detect(
        self, image: "Image.Image", ocr: Dict[str, Any]
    ) -> Tuple[List[Block], Dict[str, Any]]:
        """Detect tables using PaddleOCR with validation."""
        if self._paddle_engine is not None:
            return self._paddle_detect(image, ocr)
        else:
            # Fallback to original table detector
            return self._fallback_detector.detect(image, ocr)

    def _paddle_detect(
        self, image: "Image.Image", ocr: Dict[str, Any]
    ) -> Tuple[List[Block], Dict[str, Any]]:
        """PaddleOCR-based table detection with validation."""
        try:
            # Convert PIL image to numpy array for PaddleOCR
            import numpy as np

            img_array = np.array(image)

            # Run PaddleOCR structure detection
            paddle_results = self._paddle_engine(img_array)

            tables: List[Block] = []
            table_count = 0
            total_detections = 0
            filtered_count = 0

            for result in paddle_results:
                if result.get("type") == "table":
                    total_detections += 1

                    # Extract bounding box
                    bbox = result.get("bbox", [0, 0, 0, 0])
                    if len(bbox) < 4:
                        continue

                    x, y, x2, y2 = bbox[:4]
                    w, h = x2 - x, y2 - y

                    # Apply validation filters
                    if not self._is_valid_table_detection(result, w, h):
                        filtered_count += 1
                        continue

                    # Store HTML for later use by tasks, but don't process it here
                    html_content = result.get("res", {}).get("html", None)
                    confidence = result.get("confidence", 1.0)

                    tables.append(
                        Block(
                            id=f"table_paddle_{table_count}",
                            kind="table",
                            bbox=Rect(int(x), int(y), int(w), int(h)),
                            text=html_content,  # Raw HTML for tasks to process
                            confidence=confidence,
                        )
                    )
                    table_count += 1

            # Calculate confidence with validation
            base_confidence = 0.0
            if table_count > 0:
                # Higher confidence if we have multiple valid tables
                base_confidence = min(1.0, 0.85 + 0.05 * table_count)
                # Reduce confidence if many detections were filtered out
                if total_detections > 0:
                    filter_ratio = filtered_count / total_detections
                    base_confidence *= 1.0 - filter_ratio * 0.3

            has_table = table_count > 0

            return tables, {
                "has_table": has_table,
                "paddle_confidence": base_confidence,
                "paddle_table_count": table_count,
                "paddle_total_detections": total_detections,
                "paddle_filtered_count": filtered_count,
            }

        except Exception as e:
            # If PaddleOCR fails, fall back to original detector
            print(f"PaddleOCR detection failed: {e}")
            if self._fallback_detector is None:
                self._fallback_detector = TableDetector()
            return self._fallback_detector.detect(image, ocr)

    def _is_valid_table_detection(
        self, result: Dict[str, Any], width: int, height: int
    ) -> bool:
        """Validate if a PaddleOCR table detection is likely a real table."""

        # Check minimum area
        area = width * height
        if area < self.min_table_area:
            return False

        # Check aspect ratio (tables shouldn't be too thin/tall)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > 20:  # Too thin/tall to be a reasonable table
            return False

        # Check confidence if available
        confidence = result.get("confidence", 1.0)
        if confidence < self.min_confidence:
            return False

        # Analyze table structure if HTML is available
        html = result.get("res", {}).get("html", "")
        if html:
            return self._validate_table_structure(html)

        return True

    def _validate_table_structure(self, html: str) -> bool:
        """Validate table structure from HTML to ensure it's a real table."""
        try:
            import re

            # Count rows and cells
            row_matches = re.findall(r"<tr[^>]*>", html, re.IGNORECASE)
            if len(row_matches) < 2:  # Need at least 2 rows
                return False

            # Count total cells
            cell_matches = re.findall(r"<(?:td|th)[^>]*>", html, re.IGNORECASE)
            if len(cell_matches) < self.min_cells:
                return False

            # Check for reasonable cells per row ratio
            avg_cells_per_row = len(cell_matches) / len(row_matches)
            if (
                avg_cells_per_row < 2
            ):  # Each row should have at least 2 cells on average
                return False

            # Look for actual text content in cells
            cell_content = re.findall(
                r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", html, re.DOTALL | re.IGNORECASE
            )
            non_empty_cells = sum(1 for content in cell_content if content.strip())

            if (
                non_empty_cells < self.min_cells / 2
            ):  # At least half the cells should have content
                return False

            return True

        except Exception:
            # If HTML parsing fails, be conservative and accept the detection
            return True
