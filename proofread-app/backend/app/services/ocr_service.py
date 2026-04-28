"""OCR orchestration — dual engine dispatch + result merging."""
import asyncio
import logging
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import CONFIDENCE_THRESHOLD, PIPELINE_DIR
from ..models import Page, Character, CorrectionLog
from .char_detection import DetectedChar, detect_characters_paddle
from .kraken_service import run_kraken_ocr

logger = logging.getLogger(__name__)

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
    if "deskew" in engines:
        try:
            deskew_chars = await asyncio.to_thread(
                _run_deskew_pipeline, image_path,
            )
            results["deskew"] = deskew_chars
        except Exception as e:
            logger.error("Deskew pipeline failed for page %d: %s", page_id, e)

    if "paddle" in engines:
        try:
            paddle_chars = await asyncio.to_thread(detect_characters_paddle, image_path)
            results["paddle"] = paddle_chars
        except Exception as e:
            logger.error("PaddleOCR failed for page %d: %s", page_id, e)

    if "kraken" in engines:
        try:
            kraken_chars = await asyncio.to_thread(run_kraken_ocr, image_path)
            results["kraken"] = kraken_chars
        except Exception as e:
            logger.error("Kraken failed for page %d: %s", page_id, e)

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

    # Auto-apply confusion matrix: flag known error-prone OCR results
    await _apply_confusion_matrix(page_id, db)

    # Proactive scan: for chars frequently missed by OCR, check if any
    # unrecognized gap-fill bboxes match their templates
    await _scan_for_frequently_missed(page_id, image_path, db)

    # Update page stats (after confusion matrix may have changed confidence)
    recount = await db.execute(
        select(Character).where(
            Character.page_id == page_id, Character.is_deleted == False
        )
    )
    all_chars = recount.scalars().all()
    total = len(all_chars)
    low_conf = sum(1 for c in all_chars if c.ocr_confidence < CONFIDENCE_THRESHOLD)

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


def _run_deskew_pipeline(image_path: str) -> list[DetectedChar]:
    """Deskew + column-grid detection + PaddleOCR recognition.

    For scanned book pages (e.g., BSB/Google-digitized) where:
    - Kraken CHAT models fail (wrong scan format)
    - PaddleOCR's text detection misses sparse/tabular regions

    Pipeline:
    1. Perspective-correct the page using text frame borders
    2. Detect column strips via vertical ink projection
    3. Split each column into character bboxes (equal-spacing + valley snap)
    4. Run PaddleOCR on corrected image for text matching
    5. Run rec-only on unmatched bboxes
    6. Filter border noise
    """
    import os
    import cv2
    import numpy as np
    from .deskew_service import deskew_page, detect_columns_and_chars

    # Step 1: Deskew
    corrected_path = image_path.rsplit(".", 1)[0] + "_corrected.png"
    deskew_result = deskew_page(image_path, save_path=corrected_path)

    if deskew_result is None:
        # Fallback: use original image without perspective correction
        corrected_img = cv2.imread(image_path)
        if corrected_img is None:
            return []
        working_path = image_path
    else:
        corrected_img = deskew_result.corrected_image
        working_path = corrected_path
        logger.info("Deskew: skew=%.3f°, saved to %s",
                     deskew_result.skew_angle, corrected_path)

    # Step 2-3: Column detection + character splitting
    grid_bboxes = detect_columns_and_chars(corrected_img)
    if not grid_bboxes:
        logger.warning("No columns/chars detected by grid")
        return []

    logger.info("Grid detected %d bboxes", len(grid_bboxes))

    # Step 4: Run PaddleOCR on full corrected image for text matching
    ocr = _get_paddle_ocr()
    paddle_results = ocr.predict(working_path)
    ocr_lines = []
    if paddle_results:
        res = paddle_results[0].json["res"]
        for poly, text, score in zip(
            res.get("dt_polys", []),
            res.get("rec_texts", []),
            res.get("rec_scores", []),
        ):
            if not text or not text.strip() or score < 0.3:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            ocr_lines.append({
                "text": text,
                "score": score,
                "x": min(xs), "y": min(ys),
                "w": max(xs) - min(xs), "h": max(ys) - min(ys),
            })

    # Match OCR text to grid bboxes by spatial overlap
    chars: list[DetectedChar] = []
    matched_bbox_indices: set[int] = set()

    for line in ocr_lines:
        line_chars = list(line["text"])
        lx, ly, lw, lh = line["x"], line["y"], line["w"], line["h"]
        is_vert = lh > lw * 1.2

        if is_vert:
            char_h = lh / len(line_chars) if line_chars else lh
        else:
            char_w = lw / len(line_chars) if line_chars else lw

        for ci, ch in enumerate(line_chars):
            if is_vert:
                cx = lx + lw / 2
                cy = ly + ci * char_h + char_h / 2
            else:
                cx = lx + ci * char_w + char_w / 2
                cy = ly + lh / 2

            # Find best matching grid bbox
            best_idx = None
            best_dist = float("inf")
            for bi, gb in enumerate(grid_bboxes):
                if bi in matched_bbox_indices:
                    continue
                gcx = gb["x"] + gb["w"] / 2
                gcy = gb["y"] + gb["h"] / 2
                dist = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
                if dist < best_dist and dist < max(gb["w"], gb["h"]) * 1.5:
                    best_dist = dist
                    best_idx = bi

            if best_idx is not None:
                matched_bbox_indices.add(best_idx)
                gb = grid_bboxes[best_idx]
                chars.append(DetectedChar(
                    bbox_x=gb["x"], bbox_y=gb["y"],
                    bbox_w=gb["w"], bbox_h=gb["h"],
                    text=ch, confidence=line["score"],
                    engine="grid+paddle",
                ))

    # Add unmatched bboxes
    for bi, gb in enumerate(grid_bboxes):
        if bi not in matched_bbox_indices:
            chars.append(DetectedChar(
                bbox_x=gb["x"], bbox_y=gb["y"],
                bbox_w=gb["w"], bbox_h=gb["h"],
                text=None, confidence=0.0,
                engine="grid",
            ))

    # Step 5: Run rec-only on unmatched bboxes
    _rec_fill_unmatched(chars, corrected_img)

    # Step 6: Filter border noise (col 0 = leftmost, often page border)
    _filter_border_noise(chars, corrected_img.shape[1])

    # Assign reading order
    _reassign_order(chars)

    recognized = sum(1 for c in chars if c.text and not getattr(c, "_deleted", False))
    logger.info("Deskew pipeline: %d total, %d recognized", len(chars), recognized)

    # Remove deleted chars
    chars = [c for c in chars if not getattr(c, "_deleted", False)]
    return chars


_paddle_ocr_instance = None


def _get_paddle_ocr():
    """Lazy-init PaddleOCR (hybrid: v4 det + v5 rec)."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        import os
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        if str(PIPELINE_DIR.parent) not in sys.path:
            sys.path.insert(0, str(PIPELINE_DIR.parent))
        from pipeline.engines import init_paddle_ocr
        _paddle_ocr_instance = init_paddle_ocr(model="hybrid")
    return _paddle_ocr_instance


def _rec_fill_unmatched(
    chars: list[DetectedChar], image: "np.ndarray",
) -> None:
    """Run PaddleOCR rec model on individual unmatched bbox crops."""
    import os
    import cv2
    import tempfile

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddlex import create_model

    unmatched = [c for c in chars if not c.text and c.bbox_w >= 15 and c.bbox_h >= 15]
    if not unmatched:
        return

    rec = create_model("PP-OCRv5_server_rec")
    h_img, w_img = image.shape[:2]

    for c in unmatched:
        pad = 10
        y1 = max(0, int(c.bbox_y) - pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + pad)
        x1 = max(0, int(c.bbox_x) - pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + pad)
        crop = image[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, crop)
            try:
                for result in rec.predict(tmp.name):
                    r = result.json["res"]
                    text = r["rec_text"].strip()
                    score = r["rec_score"]
                    if text and score > 0.05:
                        c.text = text
                        c.confidence = score
                        c.engine = "grid+paddle_rec"
                    break
            except Exception:
                pass
            finally:
                os.unlink(tmp.name)


def _filter_border_noise(
    chars: list[DetectedChar], page_width: int,
) -> None:
    """Mark leftmost column as deleted if it's mostly narrow/noise bboxes.

    The leftmost 'column' from grid detection is often the page border
    line, not actual text. Detect this by checking if most bboxes in the
    column are very narrow (< 20px wide) or have very low confidence.
    """
    if not chars:
        return

    # Find leftmost column by X position
    min_x = min(c.bbox_x for c in chars)
    col0_threshold = min_x + 30  # chars within 30px of left edge

    col0 = [c for c in chars if c.bbox_x < col0_threshold]
    if not col0:
        return

    # Check if >50% are noise (narrow or unrecognized)
    noise_count = sum(
        1 for c in col0 if c.bbox_w < 20 or (not c.text) or c.confidence < 0.1
    )
    if noise_count > len(col0) * 0.5:
        for c in col0:
            c._deleted = True  # type: ignore[attr-defined]
        logger.info("Filtered %d border noise chars in leftmost column", len(col0))


async def _apply_confusion_matrix(page_id: int, db: AsyncSession) -> int:
    """Apply confusion matrix to newly OCR'd characters.

    For each character whose ocr_text appears in the confusion matrix (≥2 corrections):
    - Add likely corrections to alternatives
    - Reduce ocr_confidence to flag for user review (yellow highlight)

    Returns number of characters annotated.
    """
    from .confusion_service import get_confusion_dict

    confusion = await get_confusion_dict(db, min_count=2)
    if not confusion:
        return 0

    result = await db.execute(
        select(Character).where(
            Character.page_id == page_id,
            Character.is_deleted == False,
        )
    )
    chars = result.scalars().all()

    annotated = 0
    for c in chars:
        if not c.ocr_text or c.ocr_text not in confusion:
            continue

        suggestions = confusion[c.ocr_text]
        existing_alts = c.alternatives or []
        existing_texts = {a["text"] for a in existing_alts if isinstance(a, dict)}

        new_alts = list(existing_alts)
        for s in suggestions:
            if s["text"] not in existing_texts:
                new_alts.append({
                    "text": s["text"],
                    "confidence": min(0.9, s["count"] * 0.1),
                    "engine": "confusion_matrix",
                })

        if len(new_alts) > len(existing_alts):
            c.alternatives = new_alts
            # Dynamic penalty: more corrections → larger penalty (capped at 50%)
            # 2 corrections → 6%, 5 → 15%, 10 → 30%, 17+ → 50%
            total_corrections = sum(s["count"] for s in suggestions)
            penalty = min(0.5, total_corrections * 0.03)
            c.ocr_confidence = max(0.3, c.ocr_confidence * (1 - penalty))
            annotated += 1

    if annotated:
        logger.info("Confusion matrix annotated %d chars on page %d", annotated, page_id)

    return annotated


async def _scan_for_frequently_missed(
    page_id: int, image_path: str, db: AsyncSession,
) -> int:
    """For characters that OCR frequently misses, proactively template-match
    against unrecognized (text=None or confidence<0.1) bboxes on this page.

    This bridges the gap between "missed by grid detection" and "missed by OCR
    recognition": if a bbox exists but wasn't recognized, and a template for a
    frequently-missed character matches it, we fill it in preemptively.

    Returns number of characters identified.
    """
    from .confusion_service import get_frequently_missed_chars
    from .template_service import match_templates

    missed = await get_frequently_missed_chars(db, min_count=2)
    if not missed:
        return 0

    # Get the texts of frequently missed chars for filtering template results
    missed_texts = {m["text"] for m in missed}

    # Find unrecognized chars on this page
    result = await db.execute(
        select(Character).where(
            Character.page_id == page_id,
            Character.is_deleted == False,
            Character.ocr_confidence < 0.1,
        )
    )
    weak_chars = result.scalars().all()
    if not weak_chars:
        return 0

    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return 0

    h_img, w_img = img.shape[:2]
    identified = 0

    for c in weak_chars:
        x1 = max(0, int(c.bbox_x))
        y1 = max(0, int(c.bbox_y))
        x2 = min(w_img, int(c.bbox_x + c.bbox_w))
        y2 = min(h_img, int(c.bbox_y + c.bbox_h))
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        try:
            matches = await match_templates(crop, db, top_k=3)
        except Exception:
            continue

        # Prefer matches for frequently-missed characters
        for m in matches:
            if m["text"] in missed_texts and m["similarity"] >= 0.4:
                c.ocr_text = m["text"]
                c.ocr_confidence = round(m["similarity"] * 0.7, 3)
                c.ocr_engine = "missed_char_scan"
                c.alternatives = [
                    {"text": m["text"], "confidence": m["similarity"],
                     "engine": "template_missed_scan"}
                ]
                identified += 1
                break

    if identified:
        logger.info("Missed-char scan identified %d chars on page %d",
                     identified, page_id)

    return identified
