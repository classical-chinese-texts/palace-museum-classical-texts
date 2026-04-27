"""OCR orchestration — dual engine dispatch + result merging."""
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import CONFIDENCE_THRESHOLD, PIPELINE_DIR
from ..models import Page, Character, CorrectionLog
from .char_detection import DetectedChar, detect_characters_paddle
from .kraken_service import run_kraken_ocr

# Import pipeline merge utilities
if str(PIPELINE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR.parent))


async def run_ocr_pipeline(page_id: int, engines: list[str], db: AsyncSession):
    """Run OCR pipeline for a page and store results in DB."""
    page = await db.get(Page, page_id)
    if not page:
        raise ValueError(f"Page {page_id} not found")

    image_path = page.image_path

    results: dict[str, list[DetectedChar]] = {}

    # Run engines (in thread pool for CPU-bound work)
    if "paddle" in engines:
        try:
            paddle_chars = await asyncio.to_thread(detect_characters_paddle, image_path)
            results["paddle"] = paddle_chars
        except Exception as e:
            print(f"PaddleOCR failed for page {page_id}: {e}")

    if "kraken" in engines:
        try:
            kraken_chars = await asyncio.to_thread(run_kraken_ocr, image_path)
            results["kraken"] = kraken_chars
        except Exception as e:
            print(f"Kraken failed for page {page_id}: {e}")

    if not results:
        page.ocr_status = "failed"
        await db.commit()
        return

    # Merge results or use single engine
    if len(results) == 2:
        merged = _merge_dual_engine(results["paddle"], results["kraken"])
    else:
        engine_name = list(results.keys())[0]
        merged = results[engine_name]

    # Clear existing correction logs and characters for this page
    existing = await db.execute(
        select(Character).where(Character.page_id == page_id)
    )
    existing_chars = existing.scalars().all()
    if existing_chars:
        char_ids = [c.id for c in existing_chars]
        # Delete correction logs first (FK constraint)
        logs = await db.execute(
            select(CorrectionLog).where(CorrectionLog.character_id.in_(char_ids))
        )
        for log in logs.scalars().all():
            await db.delete(log)
        for c in existing_chars:
            await db.delete(c)

    # Store merged results
    gap_fill_chars: list[tuple[Character, DetectedChar]] = []
    for det in merged:
        char = Character(
            page_id=page_id,
            bbox_x=det.bbox_x,
            bbox_y=det.bbox_y,
            bbox_w=det.bbox_w,
            bbox_h=det.bbox_h,
            column_index=det.column_index,
            char_index=det.char_index,
            ocr_text=det.text,
            ocr_confidence=det.confidence,
            ocr_engine=det.engine,
            alternatives=getattr(det, "alternatives", []),
        )
        db.add(char)
        if det.engine == "gap_fill" and det.text is None:
            gap_fill_chars.append((char, det))

    await db.flush()  # Assign IDs before template matching

    # Auto-identify gap-fill chars using template DB
    if gap_fill_chars:
        identified = await _template_identify_gap_fills(
            gap_fill_chars, image_path, db,
        )
        print(f"Template auto-identified {identified}/{len(gap_fill_chars)} gap-fill chars")

    # Update page stats
    total = len(merged)
    low_conf = sum(1 for d in merged if d.confidence < CONFIDENCE_THRESHOLD)

    page.ocr_status = "done"
    page.ocr_engine = ",".join(engines)
    page.total_chars = total
    page.low_confidence_chars = low_conf
    page.confirmed_chars = 0

    await db.commit()


def _merge_dual_engine(
    paddle_chars: list[DetectedChar],
    kraken_chars: list[DetectedChar],
) -> list[DetectedChar]:
    """Merge results from PaddleOCR and Kraken using IoU alignment."""
    try:
        from pipeline.merge import vote_character, ENGINE_WEIGHTS
    except ImportError:
        ENGINE_WEIGHTS = {"paddle": 0.8, "kraken": 1.0}

    if not paddle_chars and not kraken_chars:
        return []
    if not paddle_chars:
        return kraken_chars
    if not kraken_chars:
        return paddle_chars

    merged: list[DetectedChar] = []
    used_kraken = set()

    for p_char in paddle_chars:
        best_match = None
        best_iou = 0.0

        for k_idx, k_char in enumerate(kraken_chars):
            if k_idx in used_kraken:
                continue
            iou = _compute_iou(p_char, k_char)
            if iou > best_iou:
                best_iou = iou
                best_match = (k_idx, k_char)

        if best_match and best_iou > 0.3:
            k_idx, k_char = best_match
            used_kraken.add(k_idx)

            # Both engines detected this character
            if p_char.text == k_char.text:
                # Agreement — high confidence
                merged_char = DetectedChar(
                    bbox_x=(p_char.bbox_x + k_char.bbox_x) / 2,
                    bbox_y=(p_char.bbox_y + k_char.bbox_y) / 2,
                    bbox_w=(p_char.bbox_w + k_char.bbox_w) / 2,
                    bbox_h=(p_char.bbox_h + k_char.bbox_h) / 2,
                    text=p_char.text,
                    confidence=max(p_char.confidence, k_char.confidence),
                    column_index=p_char.column_index,
                    char_index=p_char.char_index,
                    engine="both",
                )
            else:
                # Disagreement — use weighted vote
                p_weight = ENGINE_WEIGHTS.get("paddle", 0.8) * p_char.confidence
                k_weight = ENGINE_WEIGHTS.get("kraken", 1.0) * k_char.confidence

                if k_weight >= p_weight:
                    winner = k_char
                    alt_text = p_char.text
                    alt_conf = p_char.confidence
                    alt_engine = "paddle"
                else:
                    winner = p_char
                    alt_text = k_char.text
                    alt_conf = k_char.confidence
                    alt_engine = "kraken"

                total = p_weight + k_weight
                merged_char = DetectedChar(
                    bbox_x=(p_char.bbox_x + k_char.bbox_x) / 2,
                    bbox_y=(p_char.bbox_y + k_char.bbox_y) / 2,
                    bbox_w=(p_char.bbox_w + k_char.bbox_w) / 2,
                    bbox_h=(p_char.bbox_h + k_char.bbox_h) / 2,
                    text=winner.text,
                    confidence=max(p_weight, k_weight) / total if total > 0 else 0.5,
                    column_index=p_char.column_index,
                    char_index=p_char.char_index,
                    engine=winner.engine,
                )
                # Store alternative as attribute
                merged_char.__dict__["alternatives"] = [
                    {"text": alt_text, "confidence": alt_conf, "engine": alt_engine}
                ]

            merged.append(merged_char)
        else:
            # Only PaddleOCR detected — lower confidence
            p_char.confidence *= 0.8
            merged.append(p_char)

    # Add unmatched Kraken chars
    for k_idx, k_char in enumerate(kraken_chars):
        if k_idx not in used_kraken:
            k_char.confidence *= 0.8
            merged.append(k_char)

    # Re-sort and re-assign reading order
    _reassign_order(merged)
    return merged


def _compute_iou(a: DetectedChar, b: DetectedChar) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    x1 = max(a.bbox_x, b.bbox_x)
    y1 = max(a.bbox_y, b.bbox_y)
    x2 = min(a.bbox_x + a.bbox_w, b.bbox_x + b.bbox_w)
    y2 = min(a.bbox_y + a.bbox_h, b.bbox_y + b.bbox_h)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = a.bbox_w * a.bbox_h
    area_b = b.bbox_w * b.bbox_h
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _reassign_order(chars: list[DetectedChar], x_threshold: float = 40.0):
    """Re-sort and assign reading order after merging."""
    if not chars:
        return

    chars.sort(key=lambda c: -c.bbox_x)
    columns: list[list[DetectedChar]] = []
    current_col: list[DetectedChar] = [chars[0]]

    for c in chars[1:]:
        if abs(c.bbox_x - current_col[0].bbox_x) < x_threshold:
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


async def _template_identify_gap_fills(
    gap_fill_chars: list[tuple["Character", DetectedChar]],
    image_path: str,
    db: AsyncSession,
    min_similarity: float = 0.45,
) -> int:
    """Try to auto-identify gap-fill characters using the template DB.

    For each gap-fill char (text=None, confidence=0):
    1. Crop the character image from the page
    2. Run template matching (pHash + SSIM)
    3. If top match similarity >= min_similarity, fill in the text
       with confidence = similarity * 0.8 (lower than OCR to indicate
       it's a template-based guess, not a direct OCR result)

    Args:
        gap_fill_chars: List of (DB Character, DetectedChar) pairs.
        image_path: Path to the page image.
        db: Database session.
        min_similarity: Minimum SSIM to accept a template match (0.45 =
            empirically works for same-book characters; different books
            may need lower threshold).

    Returns:
        Number of successfully identified characters.
    """
    import cv2
    from .template_service import match_templates

    img = cv2.imread(image_path)
    if img is None:
        return 0

    h_img, w_img = img.shape[:2]
    identified = 0

    for char_obj, det in gap_fill_chars:
        x1 = max(0, int(det.bbox_x))
        y1 = max(0, int(det.bbox_y))
        x2 = min(w_img, int(det.bbox_x + det.bbox_w))
        y2 = min(h_img, int(det.bbox_y + det.bbox_h))
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        try:
            matches = await match_templates(crop, db, top_k=1)
        except Exception:
            continue

        if not matches:
            continue

        best = matches[0]
        if best["similarity"] >= min_similarity:
            char_obj.ocr_text = best["text"]
            char_obj.ocr_confidence = round(best["similarity"] * 0.8, 3)
            char_obj.ocr_engine = "template"
            identified += 1

    return identified
