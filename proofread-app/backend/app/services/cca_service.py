"""Character boundary refinement via ink projection analysis.

Ink-tighten: shrinks kraken's estimated bboxes to actual ink boundaries.
The tightened bbox is capped at original_bbox ± result_pad to prevent
expanding into neighboring characters in dense vertical text.

Progressive re-center: chars with 0% ink at search_pad=5 get re-centered
using a wider search, but bbox SIZE is capped at the original kraken size.

Tested on P1 (1398x2016, 260 chars):
- search_pad=5 finds ink that may be slightly outside kraken bbox
- Capping at original ± result_pad prevents 16px systematic overlap
- 3 chars with 0% ink get re-centered via progressive search
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

    # Compute median char size for detecting broken kraken bboxes
    # P1 stats: median w=58, h=67
    areas = [c.bbox_w * c.bbox_h for c in kraken_chars if c.bbox_w > 0 and c.bbox_h > 0]
    median_area = float(np.median(areas)) if areas else 0.0

    for c in kraken_chars:
        orig_x = c.bbox_x
        orig_y = c.bbox_y
        orig_w = c.bbox_w
        orig_h = c.bbox_h
        # Skip cap if original bbox is broken (< 25% of median area)
        # e.g. 悔: 8x16=128 vs median ~3886 → skip cap, use full ink extent
        skip_cap = (orig_w * orig_h) < (median_area * 0.25)

        # Use wider search for broken bboxes so we find the full character
        # P1: 悔 was 8x16 → search_pad=5 only finds 18x16; pad=20 finds full char
        effective_pad = 20 if skip_cap else search_pad
        x1 = max(0, int(c.bbox_x) - effective_pad)
        y1 = max(0, int(c.bbox_y) - effective_pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + effective_pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + effective_pad)

        crop = binary[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        ink_ratio = np.count_nonzero(crop) / crop.size
        if ink_ratio >= min_ink_ratio:
            # Normal case: tighten to ink extent, capped at original ± result_pad
            _project_and_tighten(
                c, crop, x1, y1, w_img, h_img, result_pad,
                orig_x, orig_y, orig_w, orig_h,
                skip_cap=skip_cap,
            )
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
    orig_x: float, orig_y: float,
    orig_w: float, orig_h: float,
    skip_cap: bool = False,
):
    """Tighten bbox to ink extent using X/Y axis projection.

    The result is capped at original_bbox ± result_pad to prevent
    expansion into neighboring characters in dense vertical text.
    Skip cap if original bbox is broken (too small).
    # P1 統計：median Y gap = -6px with cap (was -16px without)
    """
    # X axis: tighten
    col_proj = np.sum(crop, axis=0)
    ink_cols = np.where(col_proj > 0)[0]
    if len(ink_cols) > 2:
        new_x = max(0, int(ink_cols[0] + x_offset - result_pad))
        new_x2 = min(w_img, int(ink_cols[-1] + x_offset + 1 + result_pad))
        if not skip_cap:
            # Cap: don't expand beyond original bbox ± result_pad
            cap_x1 = max(0, int(orig_x) - result_pad)
            cap_x2 = min(w_img, int(orig_x + orig_w) + result_pad)
            new_x = max(new_x, cap_x1)
            new_x2 = min(new_x2, cap_x2)
        if new_x2 > new_x:
            c.bbox_x = float(new_x)
            c.bbox_w = float(new_x2 - new_x)

    # Y axis: tighten
    row_proj = np.sum(crop, axis=1)
    ink_rows = np.where(row_proj > 0)[0]
    if len(ink_rows) > 2:
        new_y = max(0, int(ink_rows[0] + y_offset - result_pad))
        new_y2 = min(h_img, int(ink_rows[-1] + y_offset + 1 + result_pad))
        if not skip_cap:
            # Cap: don't expand beyond original bbox ± result_pad
            cap_y1 = max(0, int(orig_y) - result_pad)
            cap_y2 = min(h_img, int(orig_y + orig_h) + result_pad)
            new_y = max(new_y, cap_y1)
            new_y2 = min(new_y2, cap_y2)
        if new_y2 > new_y:
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
