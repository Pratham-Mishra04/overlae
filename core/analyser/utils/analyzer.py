"""
Main analyzer for orchestrating detection and task execution.
"""

from typing import Dict, Any, List, Optional

if False:  # TYPE_CHECKING
    from PIL import Image

from .types import AnalysisResult, Predicates, Block
from .protocols import OCRProvider, Detector
from .rules import TaskRulesEngine, default_rules
from detectors import TextDetector, PaddleTableDetector
from extractors import TextExtractor, TableExtractor


class Analyzer:
    """Main analyzer that orchestrates detection and optional task execution."""

    def __init__(
        self,
        ocr_provider: OCRProvider,
        detectors: Optional[List[Detector]] = None,
        rules_engine: Optional[TaskRulesEngine] = None,
    ):
        self.ocr_provider = ocr_provider
        self.detectors = detectors or [TextDetector(), PaddleTableDetector()]
        self.rules_engine = rules_engine or default_rules()

        # Task instances for when metadata is needed
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()

    def analyze(
        self,
        image: "Image.Image",
        meta_overrides: Optional[Dict[str, Any]] = None,
        extract_task_metadata: bool = False,
    ) -> AnalysisResult:
        """
        Analyze image for content detection and eligible tasks.

        Args:
            image: PIL Image to analyze
            meta_overrides: Additional metadata to include
            extract_task_metadata: Whether to run expensive task extraction
        """
        meta = {"width": image.width, "height": image.height, "mode": image.mode}
        if meta_overrides:
            meta.update(meta_overrides)

        # Run OCR once
        ocr = self.ocr_provider.run_ocr(image)

        # Run all detectors (lightweight, fast)
        all_blocks: List[Block] = []
        preds = Predicates()

        for detector in self.detectors:
            blocks, delta = detector.detect(image, ocr)
            all_blocks.extend(blocks)
            # Merge detection results
            for k, v in delta.items():
                setattr(preds, k, getattr(preds, k, None) or v)

        # Determine eligible tasks based on predicates
        eligible_tasks, rationale = self.rules_engine.evaluate(preds)

        # Create basic result
        result = AnalysisResult(
            meta=meta,
            predicates=preds,
            blocks=all_blocks,
            eligible_tasks=eligible_tasks,
            rationale=rationale,
        )

        # Optionally extract task metadata (expensive operations)
        if extract_task_metadata:
            result = self._add_task_metadata(result, image, ocr)

        return result

    def _add_task_metadata(
        self, result: AnalysisResult, image: "Image.Image", ocr: Dict[str, Any]
    ) -> AnalysisResult:
        """Add task-specific metadata to the result."""
        task_metadata: Dict[str, Any] = {}

        # Extract text metadata if text tasks are eligible
        text_tasks = ["summarise", "aiSearchWithInput", "searchOnGoogle", "copyAsText"]
        if any(t in result.eligible_tasks for t in text_tasks):
            text_blocks = [b for b in result.blocks if b.kind == "line"]
            text_meta = self.text_extractor.extract(image, ocr, text_blocks)
            task_metadata.update(text_meta)

        # Extract table metadata if table tasks are eligible
        table_tasks = [
            "convertToPDF",
            "convertToWordDoc",
            "copyAsMarkdown",
            "exportToCSV",
            "exportToXLSX",
        ]
        if any(t in result.eligible_tasks for t in table_tasks):
            table_blocks = [b for b in result.blocks if b.kind == "table"]
            table_meta = self.table_extractor.extract(image, ocr, table_blocks)
            task_metadata.update(table_meta)

        # Add task metadata to result meta
        if task_metadata:
            result.meta["task_metadata"] = task_metadata

        return result
