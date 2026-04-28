"""Page deskew + column-grid character detection for scanned book pages.

Handles skewed/perspective-distorted scans (e.g., Google Books / BSB):
1. Detect text frame borders → perspective correction → crop to text area
2. Detect column X positions via vertical ink projection
3. Split each column into individual characters using equal-spacing + valley snap

Algorithm tuned on BSB-C285 P36 (1635×2707 → 1361×1943 after correction).

This module is an alternative to Kraken segmentation for scans where
CHAT models fail (different scan source, different margins/aspect ratio).
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeskewResult:
    """Result of page deskew + text frame detection."""
    corrected_image: np.ndarray   # Perspective-corrected, cropped to text frame
    skew_angle: float             # Detected skew angle in degrees
    corners: dict                 # Original frame corner coordinates
    output_path: str | None = None  # Path if saved to disk


def deskew_page(image_path: str, save_path: str | None = None) -> DeskewResult | None:
    """Detect text frame borders, apply perspective correction, crop.

    Steps:
    1. Morphological line detection → find horizontal & vertical frame borders
    2. Fit lines to border ink → compute precise corner intersections
    3. Perspective transform to make frame a perfect rectangle
    4. Crop with small padding to remove border lines

    Returns None if frame borders cannot be detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image: %s", image_path)
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Detect horizontal border lines ---
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_mask > 0, axis=1)
    h_rows = np.where(h_proj > w * 0.15)[0]
    h_groups = _group_consecutive(h_rows, max_gap=5)

    # Inner horizontal borders (skip page edges)
    inner_h = [g for g in h_groups if g[0] > 50 and g[-1] < h - 100]
    if len(inner_h) < 2:
        logger.warning("Cannot find 2 inner horizontal borders (found %d)", len(inner_h))
        return None

    top_border = inner_h[0]
    bot_border = inner_h[-1]

    # --- Detect vertical border lines ---
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Find vertical lines spanning > 50% of page height
    v_proj = np.sum(v_mask > 0, axis=0)
    v_cols = np.where(v_proj > h * 0.15)[0]
    v_groups = _group_consecutive(v_cols, max_gap=5)

    long_v = []
    for g in v_groups:
        mask_col = v_mask[:, g[0]:g[-1] + 1]
        rows = np.where(np.any(mask_col > 0, axis=1))[0]
        if len(rows) > 1 and rows[-1] - rows[0] > h * 0.5:
            long_v.append((g, rows[0], rows[-1]))

    # Determine right border (the innermost long vertical line close to text)
    # and left border (may not exist as a continuous line)
    if not long_v:
        logger.warning("No vertical border lines detected")
        return None

    # --- Fit lines and compute corners ---
    top_line = _fit_line_in_region(h_mask, top_border[0] - 5, top_border[-1] + 5, 0, w)
    bot_line = _fit_line_in_region(h_mask, bot_border[0] - 5, bot_border[-1] + 5, 0, w)

    if top_line is None or bot_line is None:
        logger.warning("Cannot fit horizontal border lines")
        return None

    # Right border: rightmost long vertical line within text area
    # (exclude lines at the very right edge of page, e.g., book spine binding)
    right_v = [v for v in long_v if v[0][-1] < w * 0.95]
    if right_v:
        rv = right_v[-1]  # Rightmost
        right_line = _fit_line_in_region(v_mask, 0, h, rv[0][0] - 5, rv[0][-1] + 5)
    else:
        rv = long_v[-1]
        right_line = _fit_line_in_region(v_mask, 0, h, rv[0][0] - 5, rv[0][-1] + 5)

    # Left border: try to find, otherwise infer from horizontal line endpoints
    left_v = [v for v in long_v if v[0][0] < w * 0.3]
    if left_v:
        lv = left_v[0]
        left_line = _fit_line_in_region(v_mask, 0, h, lv[0][0] - 5, lv[0][-1] + 5)
    else:
        # Infer from H-line endpoints
        top_row = h_mask[top_border[0]:top_border[-1] + 1, :]
        bot_row = h_mask[bot_border[0]:bot_border[-1] + 1, :]
        t_cols = np.where(np.any(top_row > 0, axis=0))[0]
        b_cols = np.where(np.any(bot_row > 0, axis=0))[0]
        if len(t_cols) > 0 and len(b_cols) > 0:
            avg_x = (t_cols[0] + b_cols[0]) / 2.0
            avg_y = (top_border[0] + bot_border[-1]) / 2.0
            left_line = (0.0, 1.0, float(avg_x), float(avg_y))
        else:
            left_line = (0.0, 1.0, 30.0, float(h / 2))

    if right_line is None:
        logger.warning("Cannot fit right border line")
        return None

    # Compute corners
    tl = _line_intersection(top_line, left_line)
    tr = _line_intersection(top_line, right_line)
    bl = _line_intersection(bot_line, left_line)
    br = _line_intersection(bot_line, right_line)

    if not all([tl, tr, bl, br]):
        logger.warning("Cannot compute all 4 corners")
        return None

    # Skew angle
    skew = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))

    # Perspective transform
    src_pts = np.float32([tl, tr, br, bl])
    w_top = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    w_bot = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    h_left = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
    h_right = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)

    out_w = int(max(w_top, w_bot))
    out_h = int(max(h_left, h_right))
    dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, M, (out_w, out_h),
                                     borderMode=cv2.BORDER_REPLICATE)

    # Crop out border lines (asymmetric padding)
    # Bottom padding is smaller to avoid cutting off last 1-2 chars per column
    pad = 18
    pad_bottom = 8  # less aggressive bottom crop to preserve bottom chars
    cropped = corrected[pad:out_h - pad_bottom, pad:out_w - pad]

    logger.info("Deskew: skew=%.3f°, corners TL=%s TR=%s BL=%s BR=%s, "
                "output=%dx%d", skew, tl, tr, bl, br,
                cropped.shape[1], cropped.shape[0])

    if save_path:
        cv2.imwrite(save_path, cropped)

    return DeskewResult(
        corrected_image=cropped,
        skew_angle=skew,
        corners={"TL": tl, "TR": tr, "BL": bl, "BR": br},
        output_path=save_path,
    )


def detect_columns_and_chars(
    image: np.ndarray,
    min_col_width: int = 30,
    col_gap: int = 20,
) -> list[dict]:
    """Detect column strips and split into character bboxes.

    Algorithm (tuned on BSB-C285 P36, 1361×1943):
    1. Binarize + remove border lines
    2. Vertical projection → find column X strips
    3. For each column:
       a. char_pitch ≈ col_width × 0.85 (square chars + small gap)
       b. Place equal-spaced grid lines
       c. Snap each grid line to nearest ink valley (±30% of pitch)
       d. Tighten X within each cell to actual ink extent

    Returns list of dicts with keys: col, x, y, w, h
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean: noise removal + line removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 6, 80), 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 6, 80)))
    h_lines = cv2.dilate(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    v_lines = cv2.dilate(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    clean = binary.copy()
    clean[h_lines > 0] = 0
    clean[v_lines > 0] = 0

    # --- Column detection via vertical projection ---
    v_proj = np.sum(clean > 0, axis=0).astype(float)
    if len(v_proj) > 15:
        v_proj = np.convolve(v_proj, np.ones(7) / 7, mode="same")

    v_threshold = max(np.max(v_proj) * 0.05, 10)
    ink_cols = np.where(v_proj > v_threshold)[0]
    col_groups = _group_consecutive(ink_cols, max_gap=col_gap)
    col_groups = [g for g in col_groups if len(g) > min_col_width]

    # --- Character splitting per column ---
    all_chars: list[dict] = []

    for col_i, g in enumerate(col_groups):
        x_start, x_end = g[0], g[-1]
        col_width = x_end - x_start
        strip = clean[:, x_start:x_end + 1]

        bboxes = _split_column_chars(strip, col_width)
        for b in bboxes:
            all_chars.append({
                "col": col_i,
                "x": float(x_start + b["local_x"]),
                "y": float(b["y"]),
                "w": float(b["local_w"]),
                "h": float(b["h"]),
            })

    logger.info("Column-grid detection: %d columns, %d chars",
                len(col_groups), len(all_chars))
    return all_chars


# ---- Internal helpers ----

def _split_column_chars(
    strip: np.ndarray, col_width: int,
) -> list[dict]:
    """Split a column strip into character bboxes.

    Equal-spacing grid + valley-snap:
    - char_pitch = col_width * 0.85 (capped 50-120px)
    - n_chars = total_ink_height / char_pitch
    - Grid lines at regular intervals, snapped to nearest valley ±30%
    """
    h_proj = np.sum(strip > 0, axis=1).astype(float)
    if len(h_proj) > 5:
        smooth = np.convolve(h_proj, np.ones(3) / 3, mode="same")
    else:
        smooth = h_proj

    ink_threshold = max(col_width * 0.02, 2)
    ink_rows = np.where(smooth > ink_threshold)[0]
    if len(ink_rows) < 10:
        return []

    ink_start = int(ink_rows[0])
    ink_end = int(ink_rows[-1])
    # Extend ink_end by half a char pitch to avoid cutting the last char
    strip_height = strip.shape[0]
    est_pitch = col_width * 0.85
    ink_end_extended = min(strip_height - 1, int(ink_end + est_pitch * 0.5))
    ink_end = ink_end_extended
    total_h = ink_end - ink_start

    # Character pitch from column width
    # BSB-C285 P36: col_width=113 → pitch≈96 → 19 chars in 1826px ✓
    char_pitch = col_width * 0.85
    char_pitch = max(50.0, min(char_pitch, 120.0))

    n_chars = max(1, round(total_h / char_pitch))
    actual_pitch = total_h / n_chars

    # Find valleys (adaptive local threshold)
    valleys = _find_valleys_adaptive(h_proj)

    # Place grid, snap to valleys
    splits: list[int] = []
    for i in range(1, n_chars):
        target = ink_start + i * actual_pitch
        snap_range = actual_pitch * 0.3

        best_v = None
        best_d = float("inf")
        for v in valleys:
            d = abs(v - target)
            if d < snap_range and d < best_d:
                best_d = d
                best_v = v

        splits.append(int(best_v) if best_v is not None else int(target))

    # Create bboxes
    boundaries = [ink_start] + splits + [ink_end]
    bboxes: list[dict] = []

    for i in range(len(boundaries) - 1):
        sy, ey = boundaries[i], boundaries[i + 1]
        if ey - sy < 8:
            continue

        # Tighten X to ink extent
        sub = strip[sy:ey + 1, :]
        cp = np.sum(sub > 0, axis=0)
        ix = np.where(cp > 0)[0]
        if len(ix) > 0:
            tx = max(0, ix[0] - 2)
            tw = ix[-1] - ix[0] + 5
        else:
            tx = 0
            tw = strip.shape[1]

        bboxes.append({"local_x": tx, "y": sy, "local_w": tw, "h": ey - sy})

    return bboxes


def _find_valleys_adaptive(proj: np.ndarray) -> list[int]:
    """Find valleys using adaptive local-max threshold."""
    smooth = np.convolve(proj, np.ones(3) / 3, mode="same") if len(proj) > 5 else proj
    window = 100

    local_max = np.zeros_like(smooth)
    for i in range(len(smooth)):
        lo = max(0, i - window // 2)
        hi = min(len(smooth), i + window // 2)
        local_max[i] = np.max(smooth[lo:hi])

    # Relaxed threshold at edges (first/last 15% of strip) to catch
    # border chars whose inter-char gaps have less ink contrast
    n = len(smooth)
    edge_zone = int(n * 0.15)
    threshold = np.full_like(smooth, 0.15)
    threshold[:edge_zone] = 0.25
    threshold[-edge_zone:] = 0.25

    valley_mask = (smooth < local_max * threshold) & (local_max > 5)
    valley_rows = np.where(valley_mask)[0]
    vg = _group_consecutive(valley_rows, max_gap=5)
    return [(g[0] + g[-1]) // 2 for g in vg if len(g) >= 2]


def _fit_line_in_region(
    mask: np.ndarray, y1: int, y2: int, x1: int, x2: int,
) -> tuple[float, float, float, float] | None:
    """Fit a line to ink points in the given region. Returns (vx, vy, x0, y0)."""
    y1, y2 = max(0, y1), min(mask.shape[0], y2)
    x1, x2 = max(0, x1), min(mask.shape[1], x2)
    region = mask[y1:y2, x1:x2]
    pts = np.column_stack(np.where(region > 0))
    if len(pts) < 10:
        return None
    pts_xy = np.float32([(x1 + c, y1 + r) for r, c in pts])
    line = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    return tuple(line.flatten())


def _line_intersection(
    l1: tuple, l2: tuple,
) -> tuple[int, int] | None:
    """Intersection of two lines (vx, vy, x0, y0)."""
    vx1, vy1, x01, y01 = l1
    vx2, vy2, x02, y02 = l2
    denom = vx1 * vy2 - vy1 * vx2
    if abs(denom) < 1e-10:
        return None
    t1 = (vx2 * (y01 - y02) - vy2 * (x01 - x02)) / denom
    ix = x01 + t1 * vx1
    iy = y01 + t1 * vy1
    return (int(round(ix)), int(round(iy)))


def _group_consecutive(indices: np.ndarray | list, max_gap: int = 5) -> list[list[int]]:
    """Group consecutive indices with at most max_gap between them."""
    if len(indices) == 0:
        return []
    indices = list(indices)
    groups: list[list[int]] = [[indices[0]]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= max_gap:
            groups[-1].append(indices[i])
        else:
            groups.append([indices[i]])
    return groups
