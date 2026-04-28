"""Mandoku format export service."""
import sys
from pathlib import Path
from datetime import date

from ..config import PIPELINE_DIR

# Try to import pipeline formatter
try:
    if str(PIPELINE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR.parent))
    from pipeline.formatter import format_mandoku
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False


def format_page_mandoku(
    chars: list[dict],
    book_name: str,
    volume_code: str,
    page_number: int,
    uncertain_threshold: float = 0.7,
) -> str:
    """Format characters into Mandoku text format.

    chars: list of dicts with keys: display_text, column_index, is_confirmed, ocr_confidence
    """
    volume_num = volume_code.replace("GGZBCK", "")

    lines = [
        f"#+TITLE: {book_name}",
        f"#+SOURCE: 故宮珍本叢刊第{volume_num}冊",
        f"#+DATE: {date.today().isoformat()}",
        f"#+PAGE: {page_number}",
        f"#+PROOFREADER: proofread-app",
        "",
    ]

    current_col = -1
    col_text: list[str] = []

    for c in chars:
        col_idx = c["column_index"]
        if col_idx != current_col:
            if col_text:
                lines.append("¶".join(col_text) if len(col_text) > 1 else col_text[0])
            col_text = []
            current_col = col_idx

        text = c["display_text"]
        if not c.get("is_confirmed", False) and c.get("ocr_confidence", 0) < uncertain_threshold:
            text = f"[?{text}]"

        col_text.append(text)

    if col_text:
        lines.append("".join(col_text))

    return "\n".join(lines) + "\n"
