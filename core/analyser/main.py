#!/usr/bin/env python3
"""
Modular image analyzer for agentOS.

Analyzes screenshots and returns:
- Predicates (has_text, has_table, ...)
- Eligible tasks based on rules
- Optional task-specific metadata (only when needed)

Usage:
  python main.py --image ./sample.png --pretty
  python main.py --image ./sample.png --with-metadata --pretty
"""

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import asdict
from typing import Any

# Import from our modular structure
from utils import TesseractOCR, Analyzer
from utils.types import AnalysisResult

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


def load_image(path_or_base64: str) -> "Image.Image":
    """Load image from file path or base64 string."""
    if Image is None:
        raise RuntimeError("PIL (Pillow) is required to load images.")
    if os.path.exists(path_or_base64):
        return Image.open(path_or_base64).convert("RGB")
    try:
        b = base64.b64decode(path_or_base64)
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as e:
        raise ValueError("Input is neither a valid file path nor base64 image.") from e


def dataclass_to_json(obj) -> Any:
    """Convert dataclass objects to JSON-serializable format."""
    if isinstance(obj, list):
        return [dataclass_to_json(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze screenshot and suggest eligible tasks (modular version)."
    )
    parser.add_argument("--image", required=True, help="Path or base64 image")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    parser.add_argument(
        "--with-metadata", action="store_true", help="Extract task metadata (slower)"
    )
    parser.add_argument("--lang", default="eng", help="OCR language")
    args = parser.parse_args()

    # Check dependencies
    if Image is None:
        print(
            "ERROR: Requires pillow. Install: pip install pillow",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        # Initialize components
        print("Initializing analyzer...")
        ocr = TesseractOCR(lang=args.lang)
        analyzer = Analyzer(ocr_provider=ocr)

        # Load and analyze image
        print("Loading image...")
        img = load_image(args.image)

        print("Running analysis...")
        result = analyzer.analyze(img, extract_task_metadata=args.with_metadata)

        # Prepare output
        payload = {
            "meta": result.meta,
            "predicates": dataclass_to_json(result.predicates),
            "eligible_tasks": result.eligible_tasks,
            "rationale": result.rationale,
            "blocks": [dataclass_to_json(b) for b in result.blocks],
        }

        # Add task metadata if extracted
        if "task_metadata" in result.meta:
            payload["task_metadata"] = result.meta["task_metadata"]

        # Output results
        print("\n" + "=" * 50)
        print("ANALYSIS RESULTS:")
        print("=" * 50)
        print(
            json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
        )

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.pretty:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
