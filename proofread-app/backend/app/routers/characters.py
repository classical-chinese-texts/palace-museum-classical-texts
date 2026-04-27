"""Character CRUD router — correction, confirmation, merge, split."""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import CONFIDENCE_THRESHOLD
from ..database import get_db, async_session
from ..models import Page, Character, CorrectionLog

logger = logging.getLogger(__name__)

router = APIRouter(tags=["characters"])


class CharacterOut(BaseModel):
    id: int
    page_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    column_index: int
    char_index: int
    ocr_text: str | None
    ocr_confidence: float
    ocr_engine: str | None
    alternatives: list | None
    is_confirmed: bool
    corrected_text: str | None
    is_deleted: bool
    display_text: str

    model_config = {"from_attributes": True}


class CharacterCreate(BaseModel):
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    column_index: int
    char_index: int
    ocr_text: str | None = None


class CharacterUpdate(BaseModel):
    corrected_text: str | None = None
    is_confirmed: bool | None = None
    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_w: float | None = None
    bbox_h: float | None = None


class MergeRequest(BaseModel):
    character_ids: list[int]
    merged_text: str | None = None


class SplitRequest(BaseModel):
    split_position: float  # relative position (0-1) within bbox


@router.get("/api/pages/{page_id}/characters", response_model=list[CharacterOut])
async def list_characters(page_id: int, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    result = await db.execute(
        select(Character)
        .where(Character.page_id == page_id, Character.is_deleted == False)
        .order_by(Character.column_index, Character.char_index)
    )
    chars = result.scalars().all()
    return [CharacterOut(
        id=c.id, page_id=c.page_id,
        bbox_x=c.bbox_x, bbox_y=c.bbox_y, bbox_w=c.bbox_w, bbox_h=c.bbox_h,
        column_index=c.column_index, char_index=c.char_index,
        ocr_text=c.ocr_text, ocr_confidence=c.ocr_confidence,
        ocr_engine=c.ocr_engine, alternatives=c.alternatives,
        is_confirmed=c.is_confirmed, corrected_text=c.corrected_text,
        is_deleted=c.is_deleted, display_text=c.display_text,
    ) for c in chars]


@router.post("/api/pages/{page_id}/characters", response_model=CharacterOut, status_code=201)
async def create_character(page_id: int, body: CharacterCreate, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    char = Character(
        page_id=page_id,
        bbox_x=body.bbox_x, bbox_y=body.bbox_y,
        bbox_w=body.bbox_w, bbox_h=body.bbox_h,
        column_index=body.column_index, char_index=body.char_index,
        ocr_text=body.ocr_text,
    )
    db.add(char)

    log = CorrectionLog(character_id=0, old_text=None, new_text=body.ocr_text, action="add")
    db.add(log)

    await db.commit()
    await db.refresh(char)
    log.character_id = char.id
    await db.commit()

    await _update_page_stats(page_id, db)
    return CharacterOut(
        id=char.id, page_id=char.page_id,
        bbox_x=char.bbox_x, bbox_y=char.bbox_y, bbox_w=char.bbox_w, bbox_h=char.bbox_h,
        column_index=char.column_index, char_index=char.char_index,
        ocr_text=char.ocr_text, ocr_confidence=char.ocr_confidence,
        ocr_engine=char.ocr_engine, alternatives=char.alternatives,
        is_confirmed=char.is_confirmed, corrected_text=char.corrected_text,
        is_deleted=char.is_deleted, display_text=char.display_text,
    )


@router.patch("/api/characters/{char_id}", response_model=CharacterOut)
async def update_character(char_id: int, body: CharacterUpdate, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char or char.is_deleted:
        raise HTTPException(404, "Character not found")

    old_text = char.display_text

    if body.corrected_text is not None:
        char.corrected_text = body.corrected_text
        char.is_confirmed = True
        action = "correct"
    elif body.is_confirmed is not None:
        char.is_confirmed = body.is_confirmed
        action = "confirm"
    else:
        action = "correct"

    if body.bbox_x is not None:
        char.bbox_x = body.bbox_x
    if body.bbox_y is not None:
        char.bbox_y = body.bbox_y
    if body.bbox_w is not None:
        char.bbox_w = body.bbox_w
    if body.bbox_h is not None:
        char.bbox_h = body.bbox_h

    log = CorrectionLog(
        character_id=char.id,
        old_text=old_text,
        new_text=char.display_text,
        action=action,
    )
    db.add(log)
    await db.commit()
    await db.refresh(char)

    await _update_page_stats(char.page_id, db)

    # Auto-save template when character is confirmed with text
    if char.is_confirmed and char.display_text and char.display_text != "□":
        try:
            page = await db.get(Page, char.page_id)
            if page and page.image_path:
                from ..services.template_service import save_template
                await save_template(
                    text=char.display_text,
                    page_image_path=page.image_path,
                    bbox_x=char.bbox_x, bbox_y=char.bbox_y,
                    bbox_w=char.bbox_w, bbox_h=char.bbox_h,
                    page_id=char.page_id,
                    char_id=char.id,
                    db=db,
                )
                await db.commit()
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to save template for char %d", char.id, exc_info=True
            )

    return CharacterOut(
        id=char.id, page_id=char.page_id,
        bbox_x=char.bbox_x, bbox_y=char.bbox_y, bbox_w=char.bbox_w, bbox_h=char.bbox_h,
        column_index=char.column_index, char_index=char.char_index,
        ocr_text=char.ocr_text, ocr_confidence=char.ocr_confidence,
        ocr_engine=char.ocr_engine, alternatives=char.alternatives,
        is_confirmed=char.is_confirmed, corrected_text=char.corrected_text,
        is_deleted=char.is_deleted, display_text=char.display_text,
    )


@router.delete("/api/characters/{char_id}", status_code=204)
async def delete_character(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404, "Character not found")

    char.is_deleted = True
    log = CorrectionLog(
        character_id=char.id,
        old_text=char.display_text,
        new_text=None,
        action="delete",
    )
    db.add(log)
    await db.commit()
    await _update_page_stats(char.page_id, db)


@router.post("/api/characters/merge", response_model=CharacterOut)
async def merge_characters(body: MergeRequest, db: AsyncSession = Depends(get_db)):
    if len(body.character_ids) < 2:
        raise HTTPException(400, "Need at least 2 characters to merge")

    chars = []
    for cid in body.character_ids:
        c = await db.get(Character, cid)
        if not c or c.is_deleted:
            raise HTTPException(404, f"Character {cid} not found")
        chars.append(c)

    # Verify same page
    page_ids = {c.page_id for c in chars}
    if len(page_ids) > 1:
        raise HTTPException(400, "Characters must be on the same page")

    # Compute merged bbox
    min_x = min(c.bbox_x for c in chars)
    min_y = min(c.bbox_y for c in chars)
    max_x = max(c.bbox_x + c.bbox_w for c in chars)
    max_y = max(c.bbox_y + c.bbox_h for c in chars)

    # Keep first char, soft-delete rest
    primary = chars[0]
    primary.bbox_x = min_x
    primary.bbox_y = min_y
    primary.bbox_w = max_x - min_x
    primary.bbox_h = max_y - min_y
    if body.merged_text:
        primary.corrected_text = body.merged_text
    primary.is_confirmed = True

    for c in chars[1:]:
        c.is_deleted = True
        db.add(CorrectionLog(
            character_id=c.id, old_text=c.display_text, new_text=None, action="merge",
        ))

    db.add(CorrectionLog(
        character_id=primary.id,
        old_text=primary.ocr_text,
        new_text=primary.display_text,
        action="merge",
    ))

    await db.commit()
    await db.refresh(primary)
    await _update_page_stats(primary.page_id, db)

    return CharacterOut(
        id=primary.id, page_id=primary.page_id,
        bbox_x=primary.bbox_x, bbox_y=primary.bbox_y,
        bbox_w=primary.bbox_w, bbox_h=primary.bbox_h,
        column_index=primary.column_index, char_index=primary.char_index,
        ocr_text=primary.ocr_text, ocr_confidence=primary.ocr_confidence,
        ocr_engine=primary.ocr_engine, alternatives=primary.alternatives,
        is_confirmed=primary.is_confirmed, corrected_text=primary.corrected_text,
        is_deleted=primary.is_deleted, display_text=primary.display_text,
    )


@router.post("/api/characters/{char_id}/split", response_model=list[CharacterOut])
async def split_character(char_id: int, body: SplitRequest, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char or char.is_deleted:
        raise HTTPException(404, "Character not found")

    # For vertical text: split horizontally (top/bottom)
    split_y = char.bbox_y + char.bbox_h * body.split_position
    h1 = split_y - char.bbox_y
    h2 = (char.bbox_y + char.bbox_h) - split_y

    # Modify original to be top half
    old_text = char.display_text
    char.bbox_h = h1
    char.ocr_text = old_text[0] if old_text and len(old_text) > 0 else None
    char.corrected_text = None
    char.is_confirmed = False
    char.ocr_confidence = 0.0

    # Create bottom half
    char2 = Character(
        page_id=char.page_id,
        bbox_x=char.bbox_x,
        bbox_y=split_y,
        bbox_w=char.bbox_w,
        bbox_h=h2,
        column_index=char.column_index,
        char_index=char.char_index + 1,
        ocr_text=old_text[1] if old_text and len(old_text) > 1 else None,
        ocr_confidence=0.0,
    )
    db.add(char2)

    db.add(CorrectionLog(
        character_id=char.id, old_text=old_text, new_text=char.ocr_text, action="split",
    ))

    await db.commit()
    await db.refresh(char)
    await db.refresh(char2)
    await _update_page_stats(char.page_id, db)

    result = []
    for c in [char, char2]:
        result.append(CharacterOut(
            id=c.id, page_id=c.page_id,
            bbox_x=c.bbox_x, bbox_y=c.bbox_y, bbox_w=c.bbox_w, bbox_h=c.bbox_h,
            column_index=c.column_index, char_index=c.char_index,
            ocr_text=c.ocr_text, ocr_confidence=c.ocr_confidence,
            ocr_engine=c.ocr_engine, alternatives=c.alternatives,
            is_confirmed=c.is_confirmed, corrected_text=c.corrected_text,
            is_deleted=c.is_deleted, display_text=c.display_text,
        ))
    return result


@router.post("/api/pages/{page_id}/characters/confirm-all")
async def confirm_all_above_threshold(
    page_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    result = await db.execute(
        select(Character).where(
            Character.page_id == page_id,
            Character.is_deleted == False,
            Character.is_confirmed == False,
            Character.ocr_confidence >= CONFIDENCE_THRESHOLD,
        )
    )
    chars = result.scalars().all()
    count = 0
    template_candidates = []
    for c in chars:
        c.is_confirmed = True
        db.add(CorrectionLog(
            character_id=c.id, old_text=c.display_text,
            new_text=c.display_text, action="confirm",
        ))
        count += 1
        if c.display_text and c.display_text != "□":
            template_candidates.append({
                "text": c.display_text,
                "bbox_x": c.bbox_x, "bbox_y": c.bbox_y,
                "bbox_w": c.bbox_w, "bbox_h": c.bbox_h,
                "page_id": c.page_id, "char_id": c.id,
            })

    await db.commit()
    await _update_page_stats(page_id, db)

    # Save templates in background (non-blocking)
    if template_candidates and page.image_path:
        background_tasks.add_task(
            _save_templates_background,
            page.image_path, template_candidates,
        )

    return {"confirmed": count}


@router.post("/api/pages/{page_id}/characters/reorder")
async def reorder_characters(page_id: int, db: AsyncSession = Depends(get_db)):
    """Recalculate column_index and char_index based on bbox positions."""
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    result = await db.execute(
        select(Character).where(
            Character.page_id == page_id, Character.is_deleted == False
        )
    )
    chars = list(result.scalars().all())
    if not chars:
        return {"reordered": 0}

    # Sort by x descending (right to left columns), then y ascending (top to bottom)
    chars.sort(key=lambda c: (-c.bbox_x, c.bbox_y))

    # Group into columns: chars with similar x positions
    x_threshold = 40.0
    columns: list[list[Character]] = []
    current_col: list[Character] = [chars[0]]

    for c in chars[1:]:
        if abs(c.bbox_x - current_col[0].bbox_x) < x_threshold:
            current_col.append(c)
        else:
            columns.append(current_col)
            current_col = [c]
    columns.append(current_col)

    # Assign indices
    for col_idx, col in enumerate(columns):
        col.sort(key=lambda c: c.bbox_y)
        for char_idx, c in enumerate(col):
            c.column_index = col_idx
            c.char_index = char_idx

    await db.commit()
    return {"reordered": len(chars)}


async def _update_page_stats(page_id: int, db: AsyncSession):
    """Update page character statistics."""
    page = await db.get(Page, page_id)
    if not page:
        return

    total = await db.execute(
        select(func.count()).where(
            Character.page_id == page_id, Character.is_deleted == False
        )
    )
    confirmed = await db.execute(
        select(func.count()).where(
            Character.page_id == page_id, Character.is_deleted == False,
            Character.is_confirmed == True,
        )
    )
    low_conf = await db.execute(
        select(func.count()).where(
            Character.page_id == page_id, Character.is_deleted == False,
            Character.ocr_confidence < CONFIDENCE_THRESHOLD,
        )
    )

    page.total_chars = total.scalar() or 0
    page.confirmed_chars = confirmed.scalar() or 0
    page.low_confidence_chars = low_conf.scalar() or 0
    await db.commit()


async def _save_templates_background(
    image_path: str, candidates: list[dict]
):
    """Save templates for batch-confirmed characters (background task)."""
    from ..services.template_service import save_template

    saved = 0
    async with async_session() as db:
        for c in candidates:
            try:
                await save_template(
                    text=c["text"],
                    page_image_path=image_path,
                    bbox_x=c["bbox_x"], bbox_y=c["bbox_y"],
                    bbox_w=c["bbox_w"], bbox_h=c["bbox_h"],
                    page_id=c["page_id"], char_id=c["char_id"],
                    db=db,
                )
                saved += 1
            except Exception:
                logger.warning("Template save failed for char %d", c["char_id"], exc_info=True)
        await db.commit()
    logger.info("Saved %d/%d templates in background", saved, len(candidates))
