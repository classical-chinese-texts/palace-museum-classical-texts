"""Kraken CHAT model integration via subprocess + ALTO XML parsing.

Key: ALTO Glyph HEIGHT is always 0 for baseline OCR.
We compute real char height from consecutive glyph VPOS within each TextLine.
After parsing, ink-tightening refines bboxes to actual character boundaries.
"""
import statistics
import subprocess
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from ..config import KRAKEN_ENV, CHAT_MODELS_DIR, KRAKEN_SEG_MODEL, KRAKEN_REC_MODEL
from .char_detection import DetectedChar


def run_kraken_ocr(image_path: str) -> list[DetectedChar]:
    """Run Kraken OCR with CHAT models and parse ALTO XML output."""
    import tempfile

    kraken_bin = KRAKEN_ENV / "bin" / "kraken"
    seg_model = CHAT_MODELS_DIR / KRAKEN_SEG_MODEL
    rec_model = CHAT_MODELS_DIR / KRAKEN_REC_MODEL

    if not kraken_bin.exists():
        raise FileNotFoundError(f"Kraken binary not found: {kraken_bin}")
    if not seg_model.exists():
        raise FileNotFoundError(f"Seg model not found: {seg_model}")
    if not rec_model.exists():
        raise FileNotFoundError(f"Rec model not found: {rec_model}")

    # Use temp file for output (kraken doesn't accept /dev/stdout)
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        output_path = tmp.name

    try:
        cmd = [
            str(kraken_bin),
            "-i", image_path, output_path,
            "--alto",
            "segment", "-bl", "-d", "vertical-rl",
            "-i", str(seg_model),
            "ocr", "-m", str(rec_model),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Kraken OCR failed: {result.stderr[:500]}")

        import pathlib
        xml_str = pathlib.Path(output_path).read_text()
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Kraken OCR timed out for {image_path}")
    finally:
        import os
        os.unlink(output_path)

    return _parse_alto_xml(xml_str, image_path)


def _parse_alto_xml(xml_str: str, image_path: str) -> list[DetectedChar]:
    """Parse ALTO XML, computing real char heights from consecutive glyph positions."""
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse ALTO XML: {e}")

    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    chars: list[DetectedChar] = []

    textlines = root.findall(".//alto:TextLine", ns)
    if not textlines:
        textlines = root.findall(".//{http://www.loc.gov/standards/alto/ns-v4#}TextLine")

    for tl in textlines:
        line_vpos = float(tl.get("VPOS", 0))
        line_height = float(tl.get("HEIGHT", 0))
        line_hpos = float(tl.get("HPOS", 0))
        line_width = float(tl.get("WIDTH", 0))
        line_end = line_vpos + line_height

        glyphs = tl.findall(".//alto:Glyph", ns)
        if not glyphs:
            glyphs = tl.findall(".//{http://www.loc.gov/standards/alto/ns-v4#}Glyph")
        if not glyphs:
            continue

        # Extract glyph data
        glyph_data = []
        for g in glyphs:
            content = g.get("CONTENT", "")
            if not content or not content.strip():
                continue
            glyph_data.append({
                "text": content,
                "x": float(g.get("HPOS", 0)),
                "y": float(g.get("VPOS", 0)),
                "w": float(g.get("WIDTH", 0)),
                "gc": float(g.get("GC", 0)),
            })

        if not glyph_data:
            continue

        # Determine if this line is vertical
        if len(glyph_data) >= 2:
            y_span = abs(glyph_data[-1]["y"] - glyph_data[0]["y"])
            x_span = abs(glyph_data[-1]["x"] - glyph_data[0]["x"])
            is_vertical = y_span > x_span
        else:
            is_vertical = line_height > line_width

        # Compute median char height (robust to outliers) for fallback
        if len(glyph_data) >= 2 and is_vertical:
            diffs = [glyph_data[i+1]["y"] - glyph_data[i]["y"]
                     for i in range(len(glyph_data) - 1)
                     if glyph_data[i+1]["y"] > glyph_data[i]["y"]]
            median_char_h = statistics.median(diffs) if diffs else 60.0
            # Cap: no char should be taller than 1.5x median
            max_char_h = median_char_h * 1.5
        else:
            median_char_h = 60.0
            max_char_h = 90.0

        for i, gd in enumerate(glyph_data):
            if is_vertical:
                # Height: distance to next glyph, capped at max
                if i < len(glyph_data) - 1:
                    char_h = glyph_data[i + 1]["y"] - gd["y"]
                    if char_h <= 0:
                        char_h = median_char_h
                    else:
                        char_h = min(char_h, max_char_h)
                else:
                    # Last character: use median height (remaining space is unreliable)
                    char_h = median_char_h

                # Width: use glyph WIDTH if available, else line width
                char_w = gd["w"] if gd["w"] > 5 else line_width

                chars.append(DetectedChar(
                    bbox_x=gd["x"],
                    bbox_y=gd["y"],
                    bbox_w=char_w,
                    bbox_h=char_h,
                    text=gd["text"],
                    confidence=gd["gc"],
                    engine="kraken",
                ))
            else:
                # Horizontal text: width from next glyph, height from line
                if i < len(glyph_data) - 1:
                    char_w = glyph_data[i + 1]["x"] - gd["x"]
                    if char_w <= 0:
                        char_w = gd["w"] if gd["w"] > 0 else 30.0
                else:
                    char_w = gd["w"] if gd["w"] > 0 else 30.0

                chars.append(DetectedChar(
                    bbox_x=gd["x"],
                    bbox_y=gd["y"],
                    bbox_w=char_w,
                    bbox_h=line_height if line_height > 0 else 60.0,
                    text=gd["text"],
                    confidence=gd["gc"],
                    engine="kraken",
                ))

    # CCA-anchor bboxes using real ink contours (replaces _ink_tighten)
    from .cca_service import cca_anchor_characters, discover_missing_chars
    chars = cca_anchor_characters(image_path, chars)

    # First pass: assign reading order so gap-fill can use column_index
    _assign_reading_order(chars)

    # Discover characters kraken missed (scan column gaps for ink)
    gap_chars = discover_missing_chars(image_path, chars)
    if gap_chars:
        chars.extend(gap_chars)
        _assign_reading_order(chars)  # Re-assign with new chars included

    return chars


def _ink_tighten(chars: list[DetectedChar], image_path: str) -> list[DetectedChar]:
    """Tighten bboxes to actual ink extent using image binarization.

    Uses morphological opening to remove noise (dots, stains) before
    computing ink projections. Adds small padding after tightening.
    """
    img = cv2.imread(image_path)
    if img is None:
        return chars

    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening: remove small noise (< 3x3 px)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    search_pad = 5   # extra pixels to search for ink
    result_pad = 3    # padding added to final tight bbox
    # Minimum ink density: at least 5% of crop pixels must be ink
    min_ink_ratio = 0.03

    for c in chars:
        x1 = max(0, int(c.bbox_x) - search_pad)
        y1 = max(0, int(c.bbox_y) - search_pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + search_pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + search_pad)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Check ink density
        ink_ratio = np.count_nonzero(crop) / crop.size
        if ink_ratio < min_ink_ratio:
            continue

        # Find ink extent in x direction
        col_proj = np.sum(crop, axis=0)
        ink_cols = np.where(col_proj > 0)[0]
        if len(ink_cols) > 2:
            new_x = max(0, int(ink_cols[0] + x1 - result_pad))
            new_x2 = min(w_img, int(ink_cols[-1] + x1 + 1 + result_pad))
            c.bbox_x = float(new_x)
            c.bbox_w = float(new_x2 - new_x)

        # Find ink extent in y direction
        row_proj = np.sum(crop, axis=1)
        ink_rows = np.where(row_proj > 0)[0]
        if len(ink_rows) > 2:
            new_y = max(0, int(ink_rows[0] + y1 - result_pad))
            new_y2 = min(h_img, int(ink_rows[-1] + y1 + 1 + result_pad))
            c.bbox_y = float(new_y)
            c.bbox_h = float(new_y2 - new_y)

    return chars


def _assign_reading_order(chars: list[DetectedChar], x_threshold: float = 40.0):
    """Assign column/char indices for traditional Chinese reading order."""
    if not chars:
        return

    chars.sort(key=lambda c: -c.bbox_x)

    columns: list[list[DetectedChar]] = []
    current_col: list[DetectedChar] = [chars[0]]

    for c in chars[1:]:
        cur_center = current_col[0].bbox_x + current_col[0].bbox_w / 2
        c_center = c.bbox_x + c.bbox_w / 2
        if abs(c_center - cur_center) < x_threshold:
            current_col.append(c)
        else:
            columns.append(current_col)
            current_col = [c]
    columns.append(current_col)

    for col_idx, col in enumerate(columns):
        col.sort(key=lambda c: c.bbox_y)
        for char_idx, c in enumerate(col):
            c.column_index = col_idx
            c.char_index = char_idx
