"""Character boundary refinement via ink projection analysis.

Ink-tighten: shrinks kraken's estimated bboxes to actual ink boundaries.
The tightened bbox is capped at original_bbox ± result_pad to prevent
expanding into neighboring characters in dense vertical text.

Progressive re-center: chars with 0% ink at search_pad=5 get re-centered
using a wider search, but bbox SIZE is capped at the original kraken size.

Gap-fill: discovers characters kraken missed entirely by scanning column
gaps for ink segments via Y projection. Uses median char height from
existing detections to determine segment boundaries.

Border line removal (v2, 2026-04-26):
Uses create_clean_binary() to remove horizontal/vertical border lines
from the binary image before ink projection. This prevents chars like "太"
from being anchored to the divider line at y≈1089-1103.

Region-constrained gap-fill (v2):
Scans top and bottom text regions separately (skips divider area)
to avoid false discoveries on the border line.

Tested on P1 (1398x2016, 260 chars):
- search_pad=5 finds ink that may be slightly outside kraken bbox
- Capping at original ± result_pad prevents 16px systematic overlap
- 3 chars with 0% ink get re-centered via progressive search
- Gap-fill discovers ~70+ missing chars in 14 column gaps
"""
import logging

import cv2
import numpy as np

from .char_detection import DetectedChar
from .page_segmentation import PageLayout, create_clean_binary, get_char_region

logger = logging.getLogger(__name__)


def _make_binary(image_path: str) -> np.ndarray | None:
    """Create standard binary image (no border line removal)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def cca_anchor_characters(
    image_path: str,
    kraken_chars: list[DetectedChar],
    search_pad: int = 5,
    result_pad: int = 3,
    min_ink_ratio: float = 0.03,
    layout: PageLayout | None = None,
) -> list[DetectedChar]:
    """Tighten kraken bboxes to actual ink boundaries.

    For each character:
    1. Crop binary region around kraken bbox (+ search_pad)
    2. If ink found: project X/Y axes → tight bbox + result_pad
    3. If no ink found (kraken misposition): expand search to find ink,
       then re-center the ORIGINAL-SIZED bbox on the found ink center

    When layout is provided, uses create_clean_binary() to remove border
    lines from the binary image before processing. This prevents chars
    near the divider line (e.g. "太" at y≈1087) from being anchored
    to border ink instead of character ink.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image %s, returning original bboxes", image_path)
        return kraken_chars

    h_img, w_img = img.shape[:2]

    # Use clean binary (border lines + divider removed) when layout is available
    if layout is not None:
        binary = create_clean_binary(image_path, layout=layout)
        if binary is None:
            logger.warning("create_clean_binary failed, falling back to standard binary")
            binary = _make_binary(image_path)
    else:
        binary = _make_binary(image_path)

    if binary is None:
        return kraken_chars

    tightened = 0
    recentered = 0

    # Compute median char size for detecting broken kraken bboxes
    # P1 stats: median w=58, h=67
    areas = [c.bbox_w * c.bbox_h for c in kraken_chars if c.bbox_w > 0 and c.bbox_h > 0]
    heights = [c.bbox_h for c in kraken_chars if c.bbox_h > 0]
    widths = [c.bbox_w for c in kraken_chars if c.bbox_w > 0]
    median_area = float(np.median(areas)) if areas else 0.0
    median_h = float(np.median(heights)) if heights else 0.0
    median_w = float(np.median(widths)) if widths else 0.0

    for c in kraken_chars:
        orig_x = c.bbox_x
        orig_y = c.bbox_y
        orig_w = c.bbox_w
        orig_h = c.bbox_h
        # Skip cap if original bbox is broken:
        # 1. Area too small (< 25% of median) — e.g. 悔: 8x16=128 vs median ~3886
        # 2. Badly shaped — one dimension is extreme relative to median
        #    e.g. 太(post-CCA-v1): h=28 (42% of median 67), w=82 (141% of median 58)
        #    Area 2296 is fine (60% of 3886) but shape is wrong
        skip_cap = (orig_w * orig_h) < (median_area * 0.25)
        if not skip_cap and median_h > 0 and median_w > 0:
            h_ratio = orig_h / median_h
            w_ratio = orig_w / median_w
            # Bbox height < 50% of median OR width > 180% of median → broken shape
            if h_ratio < 0.5 or w_ratio > 1.8 or h_ratio > 2.0 or w_ratio < 0.3:
                skip_cap = True

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

            # Post-tightening check: if result is unreasonably small,
            # the char may have been tightened onto noise/border residue
            # instead of actual character ink.
            # P1: 太 PRE=(1132,1081,76,63) → tightened to h=10 (15% of median 67)
            # because divider cleanup left small ink fragments.
            # Fix: restore Kraken's original position and median size,
            # then re-center using region-masked binary (exclude opposite
            # region to prevent ink from neighboring chars pulling the
            # center-of-mass in the wrong direction).
            if median_h > 0 and c.bbox_h < median_h * 0.4:
                # Restore to original Kraken position with median size
                c.bbox_x = orig_x
                c.bbox_y = orig_y
                c.bbox_w = median_w
                c.bbox_h = median_h

                # For divider-adjacent chars, mask opposite region
                # Use bbox TOP edge (not center) to determine region,
                # since divider-adjacent chars have centers past the divider.
                # P1: 太 orig_y=1081 < divider_start=1089 → top region
                region_binary = binary
                if layout is not None:
                    if orig_y < layout.divider_y_start:
                        # Char starts in top region → zero out bottom
                        region_binary = binary.copy()
                        region_binary[layout.divider_y_start:, :] = 0
                    elif orig_y > layout.divider_y_end:
                        # Char starts in bottom region → zero out top
                        region_binary = binary.copy()
                        region_binary[:layout.divider_y_end, :] = 0

                if _recenter_on_ink(c, region_binary, h_img, w_img, min_ink_ratio):
                    recentered += 1
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

    for pad in [15, 25, 40]:
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


def discover_missing_chars(
    image_path: str,
    existing_chars: list[DetectedChar],
    x_pad: int = 8,
    merge_gap: int = 20,
    min_seg_height: int = 15,
    ink_threshold_ratio: float = 0.05,
    layout: PageLayout | None = None,
) -> list[DetectedChar]:
    """Discover characters kraken missed by scanning column gaps for ink.

    Algorithm:
    1. Group existing chars into columns by X center
    2. For each column, compute X range from median char center ± median_w/2
    3. Scan Y range using Y projection (sum of ink per row)
    4. Find ink segments (contiguous rows above threshold)
    5. Merge close segments (strokes of same char, gap < merge_gap)
    6. Split over-tall segments (> 1.5 × median_h)
    7. Filter out segments overlapping existing chars
    8. Create DetectedChar entries for remaining segments (text=None, confidence=0)

    Region-constrained (v2): when layout is provided, scans top and bottom
    text regions SEPARATELY (skips divider area y=1089-1103). This prevents
    false gap-fill discoveries on the border line and ensures proper
    region boundaries.

    Tested on P1:
    - median_h=67, median_w=58 (from 260 existing chars)
    - merge_gap=20 (~30% median_h): merges multi-stroke chars
    - min_seg_height=15: filters noise and border line fragments
    """
    if not existing_chars or len(existing_chars) < 5:
        return []

    # Use clean binary (border lines + divider removed) when layout is available
    if layout is not None:
        binary = create_clean_binary(image_path, layout=layout)
    else:
        binary = _make_binary(image_path)

    if binary is None:
        logger.warning("Cannot create binary for %s, skipping gap-fill", image_path)
        return []

    h_img, w_img = binary.shape[:2]

    # Page-level statistics from existing detections
    # P1 stats: median_h=67, median_w=58
    heights = [c.bbox_h for c in existing_chars if c.bbox_h > 0]
    widths = [c.bbox_w for c in existing_chars if c.bbox_w > 0]
    median_h = float(np.median(heights))
    median_w = float(np.median(widths))

    # Define Y scan ranges: either per-region (top/bottom) or full page
    if layout is not None:
        # Scan top and bottom text regions separately
        # Skip divider area to avoid false discoveries on border line
        scan_ranges = [
            (layout.top_text.y_start, layout.top_text.y_end),
            (layout.bottom_text.y_start, layout.bottom_text.y_end),
        ]
    else:
        # Fallback: full page Y extent
        page_top = min(c.bbox_y for c in existing_chars)
        page_bottom = max(c.bbox_y + c.bbox_h for c in existing_chars)
        scan_ranges = [(max(0, int(page_top) - 5), min(h_img, int(page_bottom) + 5))]

    # Group chars by column_index (set by _assign_reading_order before this call)
    from collections import defaultdict
    col_groups: dict[int, list[DetectedChar]] = defaultdict(list)
    for c in existing_chars:
        col_groups[c.column_index].append(c)

    # Pre-sort ALL existing chars by Y for cross-column overlap checking
    all_chars_sorted = sorted(existing_chars, key=lambda c: c.bbox_y)

    discovered: list[DetectedChar] = []

    for col_idx, col_chars in col_groups.items():
        if len(col_chars) < 2:
            continue  # Skip singleton columns (unreliable X position)

        # Column X center from existing chars
        x_centers = [c.bbox_x + c.bbox_w / 2 for c in col_chars]
        col_x_center = float(np.median(x_centers))

        x1 = max(0, int(col_x_center - median_w / 2 - x_pad))
        x2 = min(w_img, int(col_x_center + median_w / 2 + x_pad))

        # Collect ALL existing chars whose X center falls in the scan range
        chars_in_range = [
            c for c in all_chars_sorted
            if x1 <= c.bbox_x + c.bbox_w / 2 <= x2
        ]

        # Scan each Y range separately (top region, bottom region)
        for y1, y2 in scan_ranges:
            y1 = max(0, y1)
            y2 = min(h_img, y2)
            crop = binary[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            row_proj = np.sum(crop, axis=1).astype(float)

            segments = _find_ink_segments(
                row_proj, median_h, merge_gap, min_seg_height, ink_threshold_ratio,
            )

            for seg_start_rel, seg_end_rel in segments:
                seg_y1 = seg_start_rel + y1  # absolute Y coordinates
                seg_y2 = seg_end_rel + y1

                if _overlaps_existing(seg_y1, seg_y2, chars_in_range):
                    continue

                # Tighten X bbox within this segment
                seg_crop = crop[seg_start_rel:seg_end_rel, :]
                col_proj = np.sum(seg_crop, axis=0)
                ink_cols = np.where(col_proj > 0)[0]
                if len(ink_cols) > 2:
                    tight_x = float(ink_cols[0] + x1 - 3)
                    tight_w = float(ink_cols[-1] - ink_cols[0] + 7)
                else:
                    continue  # No real ink in this segment

                tight_y = float(seg_y1)
                tight_h = float(seg_y2 - seg_y1)

                # Sanity: reject border-line-shaped segments
                if tight_w > tight_h * 2.5 and tight_h < median_h * 0.4:
                    continue

                discovered.append(DetectedChar(
                    bbox_x=tight_x,
                    bbox_y=tight_y,
                    bbox_w=tight_w,
                    bbox_h=tight_h,
                    text=None,
                    confidence=0.0,
                    engine="gap_fill",
                ))

    # Deduplicate: remove discoveries whose center is within 15px of another
    deduped = _dedup_discovered(discovered)

    logger.info("Gap-fill discovered %d chars (%d before dedup) across %d columns",
                len(deduped), len(discovered), len(col_groups))
    return deduped


def _dedup_discovered(chars: list[DetectedChar], min_dist: float = 15.0) -> list[DetectedChar]:
    """Remove duplicate discoveries whose centers are within min_dist pixels.

    When adjacent columns have overlapping X scan ranges, the same ink region
    gets discovered twice. Keep the one with the tighter (smaller area) bbox.
    """
    if len(chars) <= 1:
        return chars

    # Sort by Y then X for consistent ordering
    chars.sort(key=lambda c: (c.bbox_y, c.bbox_x))
    keep = [True] * len(chars)

    for i in range(len(chars)):
        if not keep[i]:
            continue
        ci_cx = chars[i].bbox_x + chars[i].bbox_w / 2
        ci_cy = chars[i].bbox_y + chars[i].bbox_h / 2
        ci_area = chars[i].bbox_w * chars[i].bbox_h

        for j in range(i + 1, len(chars)):
            if not keep[j]:
                continue
            cj_cy = chars[j].bbox_y + chars[j].bbox_h / 2
            # Early exit: Y distance too large (chars sorted by Y)
            if cj_cy - ci_cy > min_dist * 3:
                break

            cj_cx = chars[j].bbox_x + chars[j].bbox_w / 2
            dist = ((ci_cx - cj_cx) ** 2 + (ci_cy - cj_cy) ** 2) ** 0.5
            if dist < min_dist:
                # Keep the one with smaller area (tighter bbox)
                cj_area = chars[j].bbox_w * chars[j].bbox_h
                if cj_area < ci_area:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [c for c, k in zip(chars, keep) if k]


def _group_columns(
    chars: list[DetectedChar], x_threshold: float = 40.0,
) -> list[list[DetectedChar]]:
    """Group characters into columns by X center proximity."""
    if not chars:
        return []

    sorted_chars = sorted(chars, key=lambda c: -(c.bbox_x + c.bbox_w / 2))
    columns: list[list[DetectedChar]] = [[sorted_chars[0]]]

    for c in sorted_chars[1:]:
        c_center = c.bbox_x + c.bbox_w / 2
        col_center = np.mean([ch.bbox_x + ch.bbox_w / 2 for ch in columns[-1]])
        if abs(c_center - col_center) < x_threshold:
            columns[-1].append(c)
        else:
            columns.append([c])

    return columns


def _find_ink_segments(
    row_proj: np.ndarray,
    median_h: float,
    merge_gap: int = 20,
    min_seg_height: int = 15,
    threshold_ratio: float = 0.05,
) -> list[tuple[int, int]]:
    """Find character-sized ink segments in a Y projection profile.

    1. Find contiguous rows above ink threshold
    2. Merge close segments (multi-stroke chars like 八, 心)
    3. Split over-tall segments at internal projection minima

    Args:
        row_proj: Sum of ink per row (Y axis projection).
        median_h: Median character height (67px on P1).
        merge_gap: Max gap between segments to merge (20px ≈ 30% of median_h).
        min_seg_height: Reject segments shorter than this (15px = noise filter).
        threshold_ratio: Ink threshold as fraction of max projection value.

    Returns:
        List of (start_row, end_row) tuples in projection-relative coordinates.
    """
    if len(row_proj) == 0 or row_proj.max() == 0:
        return []

    threshold = max(row_proj.max() * threshold_ratio, 50)

    # Phase 1: Find raw ink segments
    raw_segments: list[list[int]] = []
    in_ink = False
    start = 0
    for i, val in enumerate(row_proj):
        if val > threshold and not in_ink:
            start = i
            in_ink = True
        elif val <= threshold and in_ink:
            raw_segments.append([start, i])
            in_ink = False
    if in_ink:
        raw_segments.append([start, len(row_proj)])

    if not raw_segments:
        return []

    # Phase 2: Merge close segments (strokes of same character)
    # merge_gap=20 ≈ 30% of median_h=67, handles chars like 八 心 小
    merged: list[list[int]] = [raw_segments[0]]
    for s, e in raw_segments[1:]:
        gap = s - merged[-1][1]
        combined_h = e - merged[-1][0]
        # Merge if gap is small AND combined height is reasonable for one char
        if gap < merge_gap and combined_h <= median_h * 1.4:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Phase 3: Split over-tall segments at internal minima
    final: list[tuple[int, int]] = []
    for s, e in merged:
        height = e - s
        if height < min_seg_height:
            continue
        if height > median_h * 1.5:
            # Split into n estimated characters
            n_est = max(2, round(height / median_h))
            seg_proj = row_proj[s:e]
            splits = [s]
            for k in range(1, n_est):
                target = int(k * height / n_est)
                # Search ±30% of expected char size for the minimum
                search_range = int(height / n_est * 0.3)
                lo = max(0, target - search_range)
                hi = min(len(seg_proj), target + search_range)
                if hi > lo:
                    local_min_idx = lo + int(np.argmin(seg_proj[lo:hi]))
                    splits.append(s + local_min_idx)
            splits.append(e)
            for k in range(len(splits) - 1):
                seg_h = splits[k + 1] - splits[k]
                if seg_h >= min_seg_height:
                    final.append((splits[k], splits[k + 1]))
        else:
            final.append((s, e))

    return final


def _overlaps_existing(
    seg_y1: float, seg_y2: float, sorted_chars: list[DetectedChar],
) -> bool:
    """Check if a Y segment overlaps any existing character bbox."""
    for c in sorted_chars:
        c_y1 = c.bbox_y
        c_y2 = c.bbox_y + c.bbox_h
        # Overlap if ranges intersect (with 5px tolerance)
        if seg_y1 < c_y2 - 5 and seg_y2 > c_y1 + 5:
            return True
        # Early exit: chars are sorted by Y, if c starts after segment ends, stop
        if c_y1 > seg_y2:
            break
    return False
