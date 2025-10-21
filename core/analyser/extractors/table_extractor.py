"""
Table extractor implementation.
"""

import re
from dataclasses import asdict
from typing import Dict, Any, List, Optional

if False:  # TYPE_CHECKING
    from PIL import Image

from utils.types import Block


class TableExtractor:
    """Extractor for extracting structured table data from detected table blocks."""

    def __init__(self):
        pass

    def extract(
        self, image: "Image.Image", ocr: Dict[str, Any], table_blocks: List[Block]
    ) -> Dict[str, Any]:
        """Extract structured table metadata from detected table blocks."""
        words = ocr.get("words", [])
        tables_meta: List[Dict[str, Any]] = []

        for tb in table_blocks:
            # Check if this is a PaddleOCR table with HTML structure
            if tb.id.startswith("table_paddle_") and tb.text:
                paddle_table_meta = self._parse_paddle_table_html(tb)
                if paddle_table_meta:
                    tables_meta.append(paddle_table_meta)
                    continue

            # Fallback to OCR-based reconstruction
            ocr_table_meta = self._extract_from_ocr(tb, words)
            if ocr_table_meta:
                tables_meta.append(ocr_table_meta)

        return {"tables": tables_meta}

    def _parse_paddle_table_html(self, table_block: Block) -> Optional[Dict[str, Any]]:
        """Parse HTML table structure from PaddleOCR results."""
        try:
            html = table_block.text
            if not html or "<table" not in html.lower():
                return None

            # Simple HTML table parser using regex
            # Extract all rows
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = re.findall(row_pattern, html, re.DOTALL | re.IGNORECASE)

            if not rows:
                return None

            # Parse each row to extract cells
            grid = []
            for row_html in rows:
                # Extract cells (both <td> and <th>)
                cell_pattern = r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>"
                cells = re.findall(cell_pattern, row_html, re.DOTALL | re.IGNORECASE)

                # Clean cell content (remove HTML tags and normalize whitespace)
                clean_cells = []
                for cell in cells:
                    # Remove HTML tags
                    clean_cell = re.sub(r"<[^>]+>", "", cell)
                    # Normalize whitespace
                    clean_cell = re.sub(r"\s+", " ", clean_cell).strip()
                    clean_cells.append(clean_cell)

                if clean_cells:  # Only add non-empty rows
                    grid.append(clean_cells)

            if not grid:
                return None

            # Ensure all rows have the same number of columns (pad with empty strings)
            max_cols = max(len(row) for row in grid) if grid else 0
            for row in grid:
                while len(row) < max_cols:
                    row.append("")

            return {
                "id": table_block.id,
                "bbox": asdict(table_block.bbox),
                "columns": max_cols,
                "rows": grid,
                "csv": self._to_csv(grid),
                "source": "paddle_ocr_html",
            }

        except Exception as e:
            print(f"Failed to parse PaddleOCR HTML table: {e}")
            return None

    def _extract_from_ocr(
        self, table_block: Block, words: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract table structure from OCR words within table bounding box."""
        x0, y0, w, h = (
            table_block.bbox.x,
            table_block.bbox.y,
            table_block.bbox.w,
            table_block.bbox.h,
        )
        x1, y1 = x0 + w, y0 + h

        # Words inside table bbox
        table_words = []
        for wobj in words:
            wx, wy, ww, wh = wobj["bbox"]
            cx, cy = wx + ww / 2, wy + wh / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                table_words.append({"text": wobj["text"], "bbox": wobj["bbox"]})

        if not table_words:
            return {
                "id": table_block.id,
                "bbox": asdict(table_block.bbox),
                "columns": 0,
                "rows": [],
                "csv": "",
                "source": "ocr_fallback",
            }

        # Estimate columns by clustering word left edges into buckets
        lefts = sorted([w["bbox"][0] for w in table_words])
        col_anchors: List[int] = []
        for lx in lefts:
            if not col_anchors or abs(col_anchors[-1] - lx) > 35:
                col_anchors.append(lx)

        # Merge overly close anchors
        merged = [col_anchors[0]]
        for a in col_anchors[1:]:
            if abs(merged[-1] - a) < 20:  # fuse very close
                merged[-1] = (merged[-1] + a) // 2
            else:
                merged.append(a)
        col_anchors = merged

        # Estimate rows by clustering word vertical centers
        centers_y = sorted([w["bbox"][1] + w["bbox"][3] / 2 for w in table_words])
        row_anchors: List[int] = []
        for cy in centers_y:
            if not row_anchors or abs(row_anchors[-1] - cy) > 14:
                row_anchors.append(int(cy))

        # Build empty grid
        C, R = max(1, len(col_anchors)), max(1, len(row_anchors))
        grid: List[List[str]] = [["" for _ in range(C)] for _ in range(R)]

        # Helper: assign word to nearest row/col
        def nearest(arr: List[int], v: float) -> int:
            best_i, best_d = 0, 1e9
            for i, a in enumerate(arr):
                d = abs(a - v)
                if d < best_d:
                    best_d, best_i = d, i
            return best_i

        for wobj in table_words:
            wx, wy, ww, wh = wobj["bbox"]
            cx, cy = wx + ww / 2, wy + wh / 2
            r = nearest(row_anchors, cy)
            c = nearest(col_anchors, wx)
            if grid[r][c]:
                grid[r][c] += " " + wobj["text"]
            else:
                grid[r][c] = wobj["text"]

        # Compact whitespace
        for r in range(R):
            for c in range(C):
                grid[r][c] = grid[r][c].strip()

        return {
            "id": table_block.id,
            "bbox": asdict(table_block.bbox),
            "columns": C,
            "rows": grid,
            "csv": self._to_csv(grid),
            "source": "ocr_reconstruction",
        }

    def _to_csv(self, rows: List[List[str]]) -> str:
        """Convert 2D grid to CSV string."""
        out = []
        for row in rows:
            escaped = []
            for cell in row:
                cell = str(cell).replace('"', '""')
                if ("," in cell) or ('"' in cell) or ("\n" in cell):
                    escaped.append(f'"{cell}"')
                else:
                    escaped.append(cell)
            out.append(",".join(escaped))
        return "\n".join(out)
