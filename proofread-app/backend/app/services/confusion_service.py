"""Confusion matrix from confirmed corrections — learns common OCR mistakes.

Uses Character table directly (ocr_text vs corrected_text) instead of
CorrectionLog, because CorrectionLog.old_text stores display_text which
may include previous manual corrections (noise), not the original OCR result.
"""
import logging
from collections import defaultdict

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Character

logger = logging.getLogger(__name__)


async def build_confusion_matrix(db: AsyncSession) -> list[dict]:
    """Build confusion matrix from confirmed character corrections.

    Queries Character directly: ocr_text (what OCR produced) vs
    corrected_text (what the user said it should be).

    Returns list of {ocr_text, correct_text, count} sorted by count desc.
    """
    result = await db.execute(
        select(
            Character.ocr_text,
            Character.corrected_text,
            func.count().label("cnt"),
        )
        .where(
            Character.is_confirmed == True,
            Character.is_deleted == False,
            Character.ocr_text.isnot(None),
            Character.corrected_text.isnot(None),
            Character.ocr_text != Character.corrected_text,
        )
        .group_by(Character.ocr_text, Character.corrected_text)
        .order_by(func.count().desc())
    )
    return [
        {"ocr_text": row.ocr_text, "correct_text": row.corrected_text, "count": row.cnt}
        for row in result.all()
    ]


async def get_likely_corrections(
    ocr_text: str, db: AsyncSession, min_count: int = 2,
) -> list[dict]:
    """Get likely correct texts for a given OCR result.

    Returns list of {text, count, probability} sorted by count desc.
    """
    result = await db.execute(
        select(
            Character.corrected_text,
            func.count().label("cnt"),
        )
        .where(
            Character.is_confirmed == True,
            Character.is_deleted == False,
            Character.ocr_text == ocr_text,
            Character.corrected_text.isnot(None),
            Character.ocr_text != Character.corrected_text,
        )
        .group_by(Character.corrected_text)
        .having(func.count() >= min_count)
        .order_by(func.count().desc())
    )
    rows = result.all()
    total = sum(r.cnt for r in rows) if rows else 1
    return [
        {"text": r.corrected_text, "count": r.cnt, "probability": r.cnt / total}
        for r in rows
    ]


async def get_missed_char_stats(db: AsyncSession) -> list[dict]:
    """Stats on manually added characters — where are they typically located?

    Groups by position within column (top 33% / middle 33% / bottom 33%)
    based on char_index relative to column size.
    """
    result = await db.execute(
        select(Character)
        .where(Character.ocr_engine == "manual", Character.is_deleted == False)
    )
    manual_chars = result.scalars().all()

    if not manual_chars:
        return [
            {"position": "top", "count": 0},
            {"position": "middle", "count": 0},
            {"position": "bottom", "count": 0},
        ]

    # For each manual char, get the total chars in its column
    counts = defaultdict(int)
    for c in manual_chars:
        max_result = await db.execute(
            select(func.max(Character.char_index))
            .where(
                Character.page_id == c.page_id,
                Character.column_index == c.column_index,
                Character.is_deleted == False,
            )
        )
        max_idx = max_result.scalar() or 1
        if max_idx == 0:
            max_idx = 1

        ratio = c.char_index / max_idx
        if ratio < 0.33:
            counts["top"] += 1
        elif ratio < 0.67:
            counts["middle"] += 1
        else:
            counts["bottom"] += 1

    return [
        {"position": pos, "count": counts.get(pos, 0)}
        for pos in ("top", "middle", "bottom")
    ]


async def get_frequently_missed_chars(db: AsyncSession, min_count: int = 2) -> list[dict]:
    """Which characters are most often manually added (missed by OCR)?

    Queries manually inserted characters that have been confirmed with text.
    Returns [{text, count}] sorted by frequency — these are characters that
    OCR systematically fails to detect (different from misrecognition).

    Use case: proactively template-scan for these chars after grid detection.
    """
    result = await db.execute(
        select(
            Character.corrected_text,
            func.count().label("cnt"),
        )
        .where(
            Character.ocr_engine == "manual",
            Character.is_deleted == False,
            Character.corrected_text.isnot(None),
        )
        .group_by(Character.corrected_text)
        .having(func.count() >= min_count)
        .order_by(func.count().desc())
    )
    return [
        {"text": row.corrected_text, "count": row.cnt}
        for row in result.all()
    ]


async def get_confusion_dict(db: AsyncSession, min_count: int = 2) -> dict[str, list[dict]]:
    """Get full confusion dictionary: {ocr_text: [{text, count}, ...]}.

    Used by OCR pipeline to auto-annotate likely errors.
    """
    matrix = await build_confusion_matrix(db)
    result: dict[str, list[dict]] = defaultdict(list)
    for entry in matrix:
        if entry["count"] >= min_count:
            result[entry["ocr_text"]].append({
                "text": entry["correct_text"],
                "count": entry["count"],
            })
    return dict(result)
