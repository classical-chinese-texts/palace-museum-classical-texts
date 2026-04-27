"""Page layout segmentation for traditional Chinese woodblock prints.

Detects the three standard regions of a thread-bound book page:
1. Upper text region (上半葉)
2. Center divider (版心/書口) — contains book title, volume number
3. Lower text region (下半葉)

Uses morphological line detection + ink projection analysis.

Tested on GGZBCK421 P1 (1398×2016):
- Top border: y=177-214 (double line, 4+10px thick)
- Divider: y=1089-1103 (15px thick)
- Bottom border: y=1981-1985 (5px thick)
- Top text region: y=215-1088 (873px)
- Bottom text region: y=1104-1980 (876px)
- 版心 vertical strip: x≈620-875 (fishtail marks at x=620, x=875)
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PageRegion:
    """A rectangular region of the page."""
    y_start: int
    y_end: int
    x_start: int
    x_end: int

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        return self.x_end - self.x_start


@dataclass
class PageLayout:
    """Detected page layout with text regions and divider."""
    top_text: PageRegion        # 上半葉
    bottom_text: PageRegion     # 下半葉

    # Divider line Y range (separating top and bottom halves)
    divider_y_start: int
    divider_y_end: int

    # 版心 vertical strip (center strip with book title)
    # None if not detected
    banxin_x_start: int | None = None
    banxin_x_end: int | None = None

    # Page outer borders
    border_top: int = 0
    border_bottom: int = 0
    border_left: int = 0
    border_right: int = 0

    @property
    def divider_y_center(self) -> float:
        """Y coordinate at center of the divider line."""
        return (self.divider_y_start + self.divider_y_end) / 2.0

    @property
    def has_banxin(self) -> bool:
        return self.banxin_x_start is not None and self.banxin_x_end is not None


def detect_page_layout(image_path: str) -> PageLayout | None:
    """Detect page layout (text regions + divider + borders).

    Algorithm:
    1. Binary threshold → morphological horizontal line detection
    2. Group consecutive line rows into line segments
    3. Find the two largest Y gaps between horizontal lines
       → these are the top and bottom text regions
    4. The divider is the horizontal line(s) between those gaps
    5. Detect vertical border lines → page left/right boundaries
    6. Detect 版心 vertical lines in center area

    Returns None if page layout cannot be determined (< 2 horizontal lines).
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image %s", image_path)
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Detect horizontal lines ---
    # Morphological opening with wide horizontal kernel
    # P1: w=1398, kernel=349 → detects lines spanning ≥25% of page width
    h_kernel_size = max(w // 4, 100)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
    h_lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Project to Y axis: sum of ink per row
    h_proj = np.sum(h_lines_mask > 0, axis=1)
    # P1: threshold = 1398 * 0.15 = 210; actual lines have ink ≥ 500
    h_line_threshold = w * 0.15
    h_line_rows = np.where(h_proj > h_line_threshold)[0]

    if len(h_line_rows) < 4:
        logger.warning("Only %d horizontal line rows detected (need ≥ 4)", len(h_line_rows))
        return None

    # Group consecutive rows into line segments
    # P1: max_gap=5 → groups [177-180], [205-214], [1089-1103], [1981-1985]
    h_line_groups = _group_consecutive(h_line_rows, max_gap=5)
    h_line_groups.sort(key=lambda g: g[0])

    logger.info("Detected %d horizontal lines: %s",
                len(h_line_groups),
                [(g[0], g[-1]) for g in h_line_groups])

    if len(h_line_groups) < 2:
        logger.warning("Only %d horizontal line group(s), need ≥ 2", len(h_line_groups))
        return None

    # --- Find the two largest gaps between consecutive line groups ---
    # P1 gaps: line0→1 = 25px, line1→2 = 875px, line2→3 = 878px
    # The two largest gaps are the text regions
    gaps: list[tuple[int, int]] = []  # (gap_size, gap_index)
    for i in range(len(h_line_groups) - 1):
        gap_start_y = h_line_groups[i][-1]
        gap_end_y = h_line_groups[i + 1][0]
        gap_size = gap_end_y - gap_start_y
        gaps.append((gap_size, i))

    gaps.sort(reverse=True)

    if len(gaps) < 2:
        logger.warning("Only %d gap(s) between lines, need ≥ 2", len(gaps))
        return None

    # The two largest gaps → top and bottom text regions
    gap1_idx = min(gaps[0][1], gaps[1][1])  # Upper gap (smaller index)
    gap2_idx = max(gaps[0][1], gaps[1][1])  # Lower gap (larger index)

    # Top text region: from end of line[gap1_idx] to start of line[gap1_idx+1]
    top_y_start = h_line_groups[gap1_idx][-1] + 1
    top_y_end = h_line_groups[gap1_idx + 1][0] - 1

    # Bottom text region: from end of line[gap2_idx] to start of line[gap2_idx+1]
    bottom_y_start = h_line_groups[gap2_idx][-1] + 1
    bottom_y_end = h_line_groups[gap2_idx + 1][0] - 1

    # Divider: line group(s) between the two text regions
    # P1: gap1_idx=1, gap2_idx=2 → divider is line group 2 (y=1089-1103)
    divider_y_start = h_line_groups[gap1_idx + 1][0]
    divider_y_end = h_line_groups[gap2_idx][-1]

    # Validate: text regions should be substantial (> 20% of page height each)
    min_region_h = h * 0.2
    if top_y_end - top_y_start < min_region_h or bottom_y_end - bottom_y_start < min_region_h:
        logger.warning("Text regions too small: top=%d, bottom=%d (min=%d)",
                       top_y_end - top_y_start, bottom_y_end - bottom_y_start, int(min_region_h))
        return None

    # --- Detect vertical lines for borders and 版心 ---
    v_kernel_size = max(h // 4, 100)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
    v_lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    v_proj = np.sum(v_lines_mask > 0, axis=0)
    v_line_threshold = h * 0.15
    v_line_cols = np.where(v_proj > v_line_threshold)[0]

    border_left = 0
    border_right = w
    banxin_x_start = None
    banxin_x_end = None

    if len(v_line_cols) > 0:
        v_line_groups = _group_consecutive(v_line_cols, max_gap=5)
        v_line_groups.sort(key=lambda g: g[0])

        if v_line_groups:
            # Leftmost vertical line → left border
            border_left = v_line_groups[0][-1] + 1
            # Rightmost vertical line → right border
            border_right = v_line_groups[-1][0] - 1

        # Find 版心 vertical boundaries
        # 版心 lines are in the center area (30%-70% of page width)
        # P1: fishtail marks at x≈620 and x≈875
        center_vlines = [
            g for g in v_line_groups
            if g[0] > w * 0.3 and g[-1] < w * 0.7
        ]
        if len(center_vlines) >= 2:
            banxin_x_start = center_vlines[0][0]
            banxin_x_end = center_vlines[-1][-1]
            logger.info("Detected 版心 vertical strip: x=%d-%d", banxin_x_start, banxin_x_end)

    layout = PageLayout(
        top_text=PageRegion(
            y_start=top_y_start,
            y_end=top_y_end,
            x_start=border_left,
            x_end=border_right,
        ),
        bottom_text=PageRegion(
            y_start=bottom_y_start,
            y_end=bottom_y_end,
            x_start=border_left,
            x_end=border_right,
        ),
        divider_y_start=divider_y_start,
        divider_y_end=divider_y_end,
        banxin_x_start=banxin_x_start,
        banxin_x_end=banxin_x_end,
        border_top=h_line_groups[0][0],
        border_bottom=h_line_groups[-1][-1],
        border_left=border_left,
        border_right=border_right,
    )

    logger.info("Page layout: top_text=y[%d-%d], divider=y[%d-%d], "
                "bottom_text=y[%d-%d], borders=x[%d-%d]",
                top_y_start, top_y_end,
                divider_y_start, divider_y_end,
                bottom_y_start, bottom_y_end,
                border_left, border_right)

    return layout


def create_clean_binary(
    image_path: str, layout: PageLayout | None = None,
) -> np.ndarray | None:
    """Create a binary image with border lines removed.

    Removes long horizontal and vertical lines (borders, dividers)
    that interfere with character bbox detection and gap-fill scanning.

    When layout is provided, also explicitly zeroes out the divider
    region (y_start-5 to y_end+5) to ensure complete removal of
    border line ink that morphological operations may miss.

    P1 lesson: morphological line removal left residual ink at y=1089-1111
    (divider at 1089-1103). This caused CCA to tighten "太" onto the
    border instead of the actual character at y≈1048-1075.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Detect and remove horizontal lines
    # P1: border at y=177-214 (double line), divider at y=1089-1103, border at y=1981-1985
    # Kernel width = w//6 → detects lines spanning ≥ ~17% of width
    h_kernel_size = max(w // 6, 80)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Detect and remove vertical lines
    # P1: borders at x=118-129 and x=1278-1291, 版心 lines at x=620, x=875
    v_kernel_size = max(h // 6, 80)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    # Dilate lines slightly before subtraction to ensure complete removal
    # A 3px dilation handles anti-aliased edges
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines = cv2.dilate(h_lines, dilate_kernel, iterations=1)
    v_lines = cv2.dilate(v_lines, dilate_kernel, iterations=1)

    # Subtract lines from binary
    clean = binary.copy()
    clean[h_lines > 0] = 0
    clean[v_lines > 0] = 0

    # Explicitly zero out the divider region when layout is known
    # Morphological removal misses residual ink (P1: y=1089-1111 had ink
    # after morph removal of line at y=1089-1103). Zero out ±5px of divider.
    if layout is not None:
        div_y1 = max(0, layout.divider_y_start - 5)
        div_y2 = min(h, layout.divider_y_end + 5)
        clean[div_y1:div_y2, :] = 0

    return clean


def get_char_region(
    bbox_y: float, bbox_h: float, layout: PageLayout,
) -> str:
    """Determine which region a character belongs to.

    Returns:
        'top' — upper text region (上半葉)
        'bottom' — lower text region (下半葉)
        'divider' — on the horizontal divider line (版心區)
        'outside' — outside all known regions
    """
    center_y = bbox_y + bbox_h / 2

    if layout.top_text.y_start <= center_y <= layout.top_text.y_end:
        return "top"
    elif layout.bottom_text.y_start <= center_y <= layout.bottom_text.y_end:
        return "bottom"
    elif layout.divider_y_start - 10 <= center_y <= layout.divider_y_end + 10:
        return "divider"
    else:
        return "outside"


def is_in_banxin(bbox_x: float, bbox_w: float, layout: PageLayout) -> bool:
    """Check if a character is in the 版心 (center vertical strip)."""
    if not layout.has_banxin:
        return False
    center_x = bbox_x + bbox_w / 2
    return layout.banxin_x_start <= center_x <= layout.banxin_x_end


def _group_consecutive(indices: np.ndarray, max_gap: int = 3) -> list[list[int]]:
    """Group consecutive indices with at most max_gap between them."""
    if len(indices) == 0:
        return []

    groups: list[list[int]] = [[int(indices[0])]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= max_gap:
            groups[-1].append(int(indices[i]))
        else:
            groups.append([int(indices[i])])

    return groups
