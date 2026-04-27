"""Character boundary refinement via ink projection analysis.

Ink-tighten: shrinks kraken's estimated bboxes to actual ink boundaries.
Progressive re-center: chars with 0% ink at search_pad=5 get re-centered
using a wider search, but bbox SIZE is capped at the original kraken size.

Tested on P1 (1398x2016, 260 chars):
- 3 chars had 0% ink at pad=5 (kraken positioning error)
- Progressive re-center fixes all 3 without expanding into neighbors
- Same-column overlaps are left as-is (6px from result_pad is acceptable)
"""
import logging

import cv2
import numpy as np

from .char_detection import DetectedChar

logger = logging.getLogger(__name__)


def cca_anchor_characters(
    image_path: str,
    kraken_chars: list[DetectedChar],
    search_pad: int = 5,
    result_pad: int = 3,
    min_ink_ratio: float = 0.03,
) -> list[DetectedChar]:
    """Tighten kraken bboxes to actual ink boundaries.

    For each character:
    1. Crop binary region around kraken bbox (+ search_pad)
    2. If ink found: project X/Y axes → tight bbox + result_pad
    3. If no ink found (kraken misposition): expand search to find ink,
       then re-center the ORIGINAL-SIZED bbox on the found ink center
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image %s, returning original bboxes", image_path)
        return kraken_chars

    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    tightened = 0
    recentered = 0

    for c in kraken_chars:
        x1 = max(0, int(c.bbox_x) - search_pad)
        y1 = max(0, int(c.bbox_y) - search_pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + search_pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + search_pad)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ink_ratio = np.count_nonzero(crop) / crop.size
        if ink_ratio >= min_ink_ratio:
            # Normal case: tighten to ink extent
            _project_and_tighten(c, crop, x1, y1, w_img, h_img, result_pad)
            tightened += 1
        else:
            # Rare case (3/260 on P1): kraken bbox is offset from actual char.
            # Expand search to find ink, then re-center original-sized bbox.
            if _recenter_on_ink(c, binary, h_img, w_img, min_ink_ratio):
                recentered += 1

    logger.info("Ink-tightened %d, re-centered %d / %d chars",
                tightened, recentered, len(kraken_chars))
    return kraken_chars


def _project_and_tighten(
    c: DetectedChar, crop: np.ndarray,
    x_offset: int, y_offset: int,
    w_img: int, h_img: int,
    result_pad: int,
):
    """Tighten bbox to ink extent using X/Y axis projection."""
    col_proj = np.sum(crop, axis=0)
    ink_cols = np.where(col_proj > 0)[0]
    if len(ink_cols) > 2:
        new_x = max(0, int(ink_cols[0] + x_offset - result_pad))
        new_x2 = min(w_img, int(ink_cols[-1] + x_offset + 1 + result_pad))
        c.bbox_x = float(new_x)
        c.bbox_w = float(new_x2 - new_x)

    row_proj = np.sum(crop, axis=1)
    ink_rows = np.where(row_proj > 0)[0]
    if len(ink_rows) > 2:
        new_y = max(0, int(ink_rows[0] + y_offset - result_pad))
        new_y2 = min(h_img, int(ink_rows[-1] + y_offset + 1 + result_pad))
        c.bbox_y = float(new_y)
        c.bbox_h = float(new_y2 - new_y)


def _recenter_on_ink(
    c: DetectedChar, binary: np.ndarray,
    h_img: int, w_img: int,
    min_ink_ratio: float,
) -> bool:
    """Re-center a mispositioned bbox by finding ink in a wider area.

    Preserves the original bbox SIZE — only shifts position.
    Searches progressively at 15px and 25px expansion.
    """
    orig_w = c.bbox_w
    orig_h = c.bbox_h

    for pad in [15, 25]:
        x1 = max(0, int(c.bbox_x) - pad)
        y1 = max(0, int(c.bbox_y) - pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + pad)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ink_ratio = np.count_nonzero(crop) / crop.size
        if ink_ratio < min_ink_ratio:
            continue

        # Found ink — compute ink center of mass
        ink_points = np.where(crop > 0)
        if len(ink_points[0]) == 0:
            continue

        ink_cy = float(np.mean(ink_points[0])) + y1
        ink_cx = float(np.mean(ink_points[1])) + x1

        # Re-center original-sized bbox on ink center
        new_x = max(0.0, ink_cx - orig_w / 2)
        new_y = max(0.0, ink_cy - orig_h / 2)
        if new_x + orig_w > w_img:
            new_x = float(w_img) - orig_w
        if new_y + orig_h > h_img:
            new_y = float(h_img) - orig_h

        c.bbox_x = new_x
        c.bbox_y = new_y
        # Keep original size
        c.bbox_w = orig_w
        c.bbox_h = orig_h
        return True

    return False
