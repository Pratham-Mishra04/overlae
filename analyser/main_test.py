#!/usr/bin/env python3
"""
Test script for the modular architecture.
Tests both fast detection and full extraction.
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add the main directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import TesseractOCR, Analyzer
from detectors import TextDetector, PaddleTableDetector


def create_test_images():
    """Create test images to validate detection accuracy."""

    # Test 1: Empty/noise image (should be has_text=False, has_table=False)
    noise_img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(noise_img)
    # Add some random noise/artifacts
    for _ in range(20):
        x = np.random.randint(0, 350)
        y = np.random.randint(0, 250)
        draw.rectangle([x, y, x + 5, y + 5], fill="gray")

    # Test 2: Clear text image (should be has_text=True, has_table=False)
    text_img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(text_img)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    draw.text((20, 50), "This is a clear text document.", fill="black", font=font)
    draw.text((20, 80), "It contains multiple lines of text.", fill="black", font=font)
    draw.text(
        (20, 110), "Should be detected as text, not table.", fill="black", font=font
    )

    # Test 3: Simple table image (should be has_text=True, has_table=True)
    table_img = Image.new("RGB", (500, 400), "white")
    draw = ImageDraw.Draw(table_img)

    # Draw table structure
    # Headers
    draw.text((20, 20), "Name", fill="black", font=font)
    draw.text((120, 20), "Age", fill="black", font=font)
    draw.text((220, 20), "City", fill="black", font=font)

    # Draw lines
    draw.line([(10, 15), (350, 15)], fill="black", width=2)  # Top
    draw.line([(10, 45), (350, 45)], fill="black", width=1)  # Header separator
    draw.line([(10, 75), (350, 75)], fill="black", width=1)  # Row 1
    draw.line([(10, 105), (350, 105)], fill="black", width=1)  # Row 2
    draw.line([(10, 135), (350, 135)], fill="black", width=2)  # Bottom

    # Vertical lines
    draw.line([(10, 15), (10, 135)], fill="black", width=2)  # Left
    draw.line([(110, 15), (110, 135)], fill="black", width=1)  # Col 1
    draw.line([(210, 15), (210, 135)], fill="black", width=1)  # Col 2
    draw.line([(350, 15), (350, 135)], fill="black", width=2)  # Right

    # Data
    draw.text((20, 55), "John", fill="black", font=font)
    draw.text((120, 55), "25", fill="black", font=font)
    draw.text((220, 55), "NYC", fill="black", font=font)

    draw.text((20, 85), "Alice", fill="black", font=font)
    draw.text((120, 85), "30", fill="black", font=font)
    draw.text((220, 85), "LA", fill="black", font=font)

    return {"noise": noise_img, "text": text_img, "table": table_img}


def test_modular_architecture():
    """Test the modular architecture with fast and full analysis."""

    print("Creating test images...")
    test_images = create_test_images()

    # Save test images for inspection
    for name, img in test_images.items():
        img.save(f"test_{name}.png")
        print(f"Saved test_{name}.png")

    try:
        print("\nInitializing modular analyzer...")
        ocr = TesseractOCR()

        # Test with improved detectors
        analyzer = Analyzer(
            ocr_provider=ocr,
            detectors=[
                TextDetector(min_text_length=3, min_word_count=2),
                PaddleTableDetector(
                    min_confidence=0.7, min_table_area=2000, min_cells=4
                ),
            ],
        )

        print("\n" + "=" * 60)
        print("TESTING MODULAR ARCHITECTURE")
        print("=" * 60)

        for name, img in test_images.items():
            print(f"\n--- Testing {name} image ---")

            # Test fast detection (no metadata extraction)
            print("🚀 Fast analysis (detection only):")
            fast_result = analyzer.analyze(img, extract_task_metadata=False)

            print(f"  has_text: {fast_result.predicates.has_text}")
            print(f"  has_table: {fast_result.predicates.has_table}")
            print(f"  blocks found: {len(fast_result.blocks)}")
            print(f"  eligible_tasks: {fast_result.eligible_tasks}")

            # Test full analysis (with metadata extraction)
            if fast_result.eligible_tasks:  # Only extract if there are eligible tasks
                print("📊 Full analysis (with task metadata):")
                full_result = analyzer.analyze(img, extract_task_metadata=True)

                task_metadata = full_result.meta.get("task_metadata", {})
                if "text" in task_metadata:
                    text_info = task_metadata["text"]
                    print(f"  text_line_count: {text_info.get('line_count', 0)}")
                    print(f"  word_count: {text_info.get('word_count', 0)}")

                if "tables" in task_metadata:
                    tables = task_metadata["tables"]
                    print(f"  tables_found: {len(tables)}")
                    for i, table in enumerate(tables):
                        print(
                            f"    table_{i}: {table['columns']}x{len(table['rows'])} cells"
                        )
                        if table.get("csv"):
                            lines = table["csv"].split("\n")
                            print(f"    preview: {lines[0] if lines else 'empty'}")
            else:
                print("📊 No eligible tasks, skipping metadata extraction")

        print(f"\n✅ Modular architecture test completed!")
        print("Key benefits:")
        print("- Fast detection without expensive metadata extraction")
        print("- Separated concerns (detectors vs tasks)")
        print("- Easy to add new detectors and tasks")
        print("- Clean, maintainable code structure")

    except Exception as e:
        print(f"Error during testing: {e}")
        print("Note: This test requires tesseract and optionally PaddleOCR.")
        print("Install with: pip install -r requirements.txt")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_modular_architecture()
