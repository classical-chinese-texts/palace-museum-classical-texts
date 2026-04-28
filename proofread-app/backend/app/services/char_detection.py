"""Character-level detection: PaddleOCR lines → valley-based splitting + ink tightening."""
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", category=DeprecationWarning)

PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "scripts" / "pipeline"
if str(PIPELINE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR.parent))

_ocr_instance = None


@dataclass
class DetectedChar:
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    text: str | None
    confidence: float
    column_index: int = 0
    char_index: int = 0
    engine: str = "paddle"
    alternatives: list = field(default_factory=list)


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        from pipeline.engines import init_paddle_ocr
        _ocr_instance = init_paddle_ocr(model="hybrid")
    return _ocr_instance


def _find_valleys(projection: np.ndarray, min_width: int = 3) -> list[dict]:
    """Find valleys (low ink regions) in a projection profile."""
    if len(projection) == 0:
        return []

    threshold = max(np.max(projection) * 0.08, 5)
    valleys = []
    in_valley = False
    start = 0

    for i, val in enumerate(projection):
        if val <= threshold and not in_valley:
            start = i
            in_valley = True
        elif val > threshold and in_valley:
            width = i - start
            if width >= min_width:
                center = start + width // 2
                valleys.append({'center': center, 'start': start, 'end': i, 'width': width})
            in_valley = False

    # Handle valley at the end
    if in_valley:
        width = len(projection) - start
        if width >= min_width:
            center = start + width // 2
            valleys.append({'center': center, 'start': start, 'end': len(projection), 'width': width})

    return valleys


def _valley_split(
    full_image: np.ndarray,
    line_x: float, line_y: float, line_w: float, line_h: float,
    n_chars: int, is_vertical: bool,
) -> list[tuple[float, float, float, float]]:
    """Split line into per-char bboxes using valley detection with forward tracking.

    1. Compute projection profile (sum of ink per row/column)
    2. Find valleys (gaps between characters)
    3. Greedily match valleys to expected split positions (forward tracking)
    4. Tighten each cell bbox to actual ink extent
    """
    if n_chars <= 0:
        return []
    if n_chars == 1:
        return [(line_x, line_y, line_w, line_h)]

    h_img, w_img = full_image.shape[:2]
    pad = 2
    y1 = max(0, int(line_y) - pad)
    y2 = min(h_img, int(line_y + line_h) + pad)
    x1 = max(0, int(line_x) - pad)
    x2 = min(w_img, int(line_x + line_w) + pad)

    crop = full_image[y1:y2, x1:x2]
    if crop.size == 0:
        return _even_split(line_x, line_y, line_w, line_h, n_chars, is_vertical)

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if is_vertical:
        proj = np.sum(binary, axis=1).astype(float)
    else:
        proj = np.sum(binary, axis=0).astype(float)

    # Smooth projection to reduce noise
    if len(proj) > 7:
        kernel = np.ones(3) / 3
        proj = np.convolve(proj, kernel, mode='same')

    length = len(proj)
    char_size = length / n_chars

    # Find valleys
    valleys = _find_valleys(proj, min_width=2)

    # Greedy forward-tracking: match each expected split to nearest valley
    splits = []
    pos = 0.0  # Current position (start of current char)

    for i in range(1, n_chars):
        target = pos + char_size  # Expected split position

        # Find best valley near target (within 40%-160% of char_size from pos)
        min_pos = pos + char_size * 0.4
        max_pos = pos + char_size * 1.6

        best_valley = None
        best_dist = float('inf')

        for v in valleys:
            vc = v['center']
            if vc <= min_pos:
                continue
            if vc > max_pos:
                continue  # Don't break - valleys might not be sorted by center
            dist = abs(vc - target)
            # Prefer wider valleys (more likely to be real character gap)
            adjusted_dist = dist - v['width'] * 0.5
            if adjusted_dist < best_dist:
                best_dist = adjusted_dist
                best_valley = v

        if best_valley is not None:
            split = best_valley['center']
        else:
            split = int(target)

        splits.append(split)
        pos = split  # Forward tracking: next char starts here

    # Convert splits to bboxes
    boundaries = [0] + splits + [length]

    bboxes = []
    for i in range(n_chars):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        if seg_end <= seg_start:
            seg_end = seg_start + 1

        if is_vertical:
            # Tighten x (secondary axis) to ink extent
            seg = binary[seg_start:seg_end, :]
            col_proj = np.sum(seg, axis=0)
            ink_cols = np.where(col_proj > 0)[0]

            if len(ink_cols) > 0:
                tight_x = float(ink_cols[0] + x1 - 2)
                tight_w = float(ink_cols[-1] - ink_cols[0] + 5)
                tight_x = max(line_x - 2, tight_x)
                tight_w = min(line_w + 4, tight_w)
            else:
                tight_x = line_x
                tight_w = line_w

            # Also tighten y within segment
            row_proj = np.sum(seg, axis=1)
            ink_rows = np.where(row_proj > 0)[0]
            if len(ink_rows) > 0:
                tight_y = float(ink_rows[0] + y1 + seg_start - 1)
                tight_h = float(ink_rows[-1] - ink_rows[0] + 3)
            else:
                tight_y = float(seg_start + y1)
                tight_h = float(seg_end - seg_start)

            bboxes.append((tight_x, tight_y, tight_w, tight_h))

        else:
            seg = binary[:, seg_start:seg_end]
            row_proj = np.sum(seg, axis=1)
            ink_rows = np.where(row_proj > 0)[0]

            if len(ink_rows) > 0:
                tight_y = float(ink_rows[0] + y1 - 2)
                tight_h = float(ink_rows[-1] - ink_rows[0] + 5)
                tight_y = max(line_y - 2, tight_y)
                tight_h = min(line_h + 4, tight_h)
            else:
                tight_y = line_y
                tight_h = line_h

            col_proj = np.sum(seg, axis=0)
            ink_cols = np.where(col_proj > 0)[0]
            if len(ink_cols) > 0:
                tight_x = float(ink_cols[0] + x1 + seg_start - 1)
                tight_w = float(ink_cols[-1] - ink_cols[0] + 3)
            else:
                tight_x = float(seg_start + x1)
                tight_w = float(seg_end - seg_start)

            bboxes.append((tight_x, tight_y, tight_w, tight_h))

    return bboxes


def _even_split(
    line_x: float, line_y: float, line_w: float, line_h: float,
    n_chars: int, is_vertical: bool,
) -> list[tuple[float, float, float, float]]:
    """Fallback: divide line bbox evenly."""
    bboxes = []
    if is_vertical:
        char_h = line_h / n_chars
        for i in range(n_chars):
            bboxes.append((line_x, line_y + i * char_h, line_w, char_h))
    else:
        char_w = line_w / n_chars
        for i in range(n_chars):
            bboxes.append((line_x + i * char_w, line_y, char_w, line_h))
    return bboxes


def detect_characters_paddle(image_path: str) -> list[DetectedChar]:
    """Detect characters: PaddleOCR lines → valley-based split + ink tightening."""
    ocr = _get_ocr()

    results = ocr.predict(image_path)
    if not results:
        return []

    res = results[0].json['res']
    dt_polys = res['dt_polys']
    rec_texts = res['rec_texts']
    rec_scores = res['rec_scores']

    full_image = cv2.imread(image_path)
    if full_image is None:
        return []

    all_chars: list[DetectedChar] = []

    for idx in range(len(rec_texts)):
        polygon = dt_polys[idx]
        text = rec_texts[idx]
        confidence = float(rec_scores[idx])

        if not text or not text.strip():
            continue
        if confidence < 0.3:
            continue

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        line_x = float(min(xs))
        line_y = float(min(ys))
        line_w = float(max(xs)) - line_x
        line_h = float(max(ys)) - line_y

        if line_w <= 0 or line_h <= 0:
            continue

        text_chars = list(text)
        n = len(text_chars)
        is_vertical = line_h > line_w * 1.2

        char_bboxes = _valley_split(
            full_image, line_x, line_y, line_w, line_h, n, is_vertical
        )

        for ci, (bx, by, bw, bh) in enumerate(char_bboxes):
            if ci < len(text_chars):
                all_chars.append(DetectedChar(
                    bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                    text=text_chars[ci], confidence=confidence,
                ))

    _assign_reading_order(all_chars)
    return all_chars


def _assign_reading_order(chars: list[DetectedChar], x_threshold: float = 40.0):
    """Assign column_index and char_index for traditional Chinese reading order."""
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
