"""Character template API — query, match, and serve template images."""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import CharacterTemplate
from ..services import template_service

router = APIRouter(tags=["templates"])


class TemplateOut(BaseModel):
    id: int
    text: str
    image_path: str
    width: int | None
    height: int | None
    source_page_id: int | None
    source_char_id: int | None
    feature_hash: str | None
    is_active: bool

    model_config = {"from_attributes": True}


class MatchResult(BaseModel):
    text: str
    similarity: float
    template_id: int
    hamming_distance: int


@router.get("/api/templates", response_model=list[TemplateOut])
async def list_templates(
    text: str | None = Query(None, description="Filter by character text"),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List templates, optionally filtered by character text."""
    q = select(CharacterTemplate).where(CharacterTemplate.is_active == True)
    if text:
        q = q.where(CharacterTemplate.text == text)
    q = q.order_by(CharacterTemplate.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


@router.post("/api/templates/match", response_model=list[MatchResult])
async def match_template(
    file: UploadFile = File(...),
    top_k: int = Query(5, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Upload a character image and find most similar templates."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")

    results = await template_service.match_templates(image, db, top_k=top_k)
    return [MatchResult(
        text=r["text"],
        similarity=r["similarity"],
        template_id=r["template_id"],
        hamming_distance=r["hamming_distance"],
    ) for r in results]


@router.get("/api/characters/{char_id}/template-matches", response_model=list[MatchResult])
async def match_by_character(
    char_id: int,
    top_k: int = Query(5, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Find template matches for an existing character (server-side crop)."""
    from ..models import Character, Page
    char = await db.get(Character, char_id)
    if not char or char.is_deleted:
        raise HTTPException(404, "Character not found")

    page = await db.get(Page, char.page_id)
    if not page or not page.image_path:
        raise HTTPException(404, "Page image not found")

    from pathlib import Path as P
    if not P(page.image_path).exists():
        raise HTTPException(404, "Page image file not found")

    # Crop character from page image
    img = cv2.imread(page.image_path)
    if img is None:
        raise HTTPException(500, "Cannot read page image")

    h_img, w_img = img.shape[:2]
    x1 = max(0, int(char.bbox_x))
    y1 = max(0, int(char.bbox_y))
    x2 = min(w_img, int(char.bbox_x + char.bbox_w))
    y2 = min(h_img, int(char.bbox_y + char.bbox_h))
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return []

    results = await template_service.match_templates(crop, db, top_k=top_k)
    return [MatchResult(
        text=r["text"],
        similarity=r["similarity"],
        template_id=r["template_id"],
        hamming_distance=r["hamming_distance"],
    ) for r in results]


@router.get("/api/templates/{template_id}/image")
async def get_template_image(
    template_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Serve a template image file."""
    template = await db.get(CharacterTemplate, template_id)
    if not template or not template.is_active:
        raise HTTPException(404, "Template not found")

    from pathlib import Path
    path = Path(template.image_path)
    if not path.exists():
        raise HTTPException(404, "Template image file not found")

    return FileResponse(str(path), media_type="image/png")


@router.get("/api/characters/{char_id}/candidates")
async def get_candidates(
    char_id: int,
    top_k: int = Query(5, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Get all candidate suggestions for a character: templates + confusables + variants.

    Returns a unified list sorted by priority:
    1. Template matches (image-level similarity, most reliable)
    2. OCR alternatives (from dual-engine merge)
    3. Confusable characters (known similar-looking chars)
    4. Variant characters (standard ↔ variant mappings)
    """
    from ..models import Character, Page

    char = await db.get(Character, char_id)
    if not char or char.is_deleted:
        raise HTTPException(404, "Character not found")

    result: dict = {
        "template_matches": [],
        "confusables": [],
        "variants": [],
    }

    # Template matches (if page image exists)
    page = await db.get(Page, char.page_id)
    if page and page.image_path:
        from pathlib import Path as P
        if P(page.image_path).exists():
            img = cv2.imread(page.image_path)
            if img is not None:
                h_img, w_img = img.shape[:2]
                x1 = max(0, int(char.bbox_x))
                y1 = max(0, int(char.bbox_y))
                x2 = min(w_img, int(char.bbox_x + char.bbox_w))
                y2 = min(h_img, int(char.bbox_y + char.bbox_h))
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    matches = await template_service.match_templates(crop, db, top_k=top_k)
                    result["template_matches"] = matches

    # Confusables + Variants from dictionaries
    current_text = char.display_text
    if current_text and current_text != "□":
        try:
            from pipeline.dictionaries import get_confusable, normalize_variant
            confusables = get_confusable(current_text)
            result["confusables"] = confusables

            normalized = normalize_variant(current_text)
            if normalized != current_text:
                result["variants"] = [normalized]
            # Also check reverse: if current text IS the standard form,
            # show what variants map to it
        except ImportError:
            pass  # dictionaries module not available

    return result


@router.get("/api/templates/stats")
async def template_stats(db: AsyncSession = Depends(get_db)):
    """Get template database statistics."""
    return await template_service.get_template_stats(db)
