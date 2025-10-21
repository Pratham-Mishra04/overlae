"""
OCR provider implementations.
"""

from typing import Dict, Any, List

# Optional deps
try:
    import pytesseract  # type: ignore
    from pytesseract import Output as TesseractOutput  # type: ignore
except Exception:
    pytesseract = None
    TesseractOutput = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

from .protocols import OCRProvider


class TesseractOCR(OCRProvider):
    """Tesseract-based OCR provider."""

    def __init__(self, lang: str = "eng"):
        if pytesseract is None or Image is None:
            raise RuntimeError("pytesseract and PIL are required for TesseractOCR")
        self.lang = lang

    def run_ocr(self, image: "Image.Image") -> Dict[str, Any]:
        """Run Tesseract OCR on image."""
        data = pytesseract.image_to_data(
            image, lang=self.lang, output_type=TesseractOutput.DICT
        )
        n = len(data["text"])

        # Build lines
        lines_map: Dict[int, Dict[str, Any]] = {}
        words_out: List[Dict[str, Any]] = []

        for i in range(n):
            txt = data["text"][i]
            if txt is None or txt.strip() == "":
                continue
            left, top = int(data["left"][i]), int(data["top"][i])
            width, height = int(data["width"][i]), int(data["height"][i])
            line_num = int(data["line_num"][i])

            # Words list (fine-grained)
            words_out.append(
                {
                    "text": txt,
                    "bbox": [left, top, width, height],
                    "line_index": line_num,
                }
            )

            # Aggregate to lines
            if line_num not in lines_map:
                lines_map[line_num] = {"text": txt, "bbox": [left, top, width, height]}
            else:
                lines_map[line_num]["text"] += " " + txt
                l, t, w, h = lines_map[line_num]["bbox"]
                r = max(l + w, left + width)
                b = max(t + h, top + height)
                nl = min(l, left)
                nt = min(t, top)
                lines_map[line_num]["bbox"] = [nl, nt, r - nl, b - nt]

        lines = [
            {"text": v["text"].strip(), "bbox": v["bbox"]}
            for _, v in sorted(lines_map.items(), key=lambda kv: kv[0])
        ]
        return {"lines": lines, "words": words_out}
