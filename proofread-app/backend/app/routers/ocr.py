"""OCR execution router."""
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db, async_session
from ..models import Page
from ..services.ocr_service import run_ocr_pipeline

router = APIRouter(tags=["ocr"])


class OCRRequest(BaseModel):
    engines: list[str] = ["paddle"]  # "paddle", "kraken", "deskew", or combinations


@router.post("/api/pages/{page_id}/ocr")
async def trigger_ocr(
    page_id: int,
    body: OCRRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    if page.ocr_status == "processing":
        raise HTTPException(409, "OCR already in progress")

    page.ocr_status = "processing"
    await db.commit()

    background_tasks.add_task(_run_ocr_background, page_id, body.engines)

    return {"status": "processing", "page_id": page_id, "engines": body.engines}


@router.get("/api/pages/{page_id}/ocr/status")
async def ocr_status(page_id: int, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    return {
        "page_id": page_id,
        "ocr_status": page.ocr_status,
        "ocr_engine": page.ocr_engine,
        "total_chars": page.total_chars,
        "low_confidence_chars": page.low_confidence_chars,
    }


class BatchOCRRequest(BaseModel):
    page_ids: list[int] | None = None  # Specific pages, or None for all pending
    engines: list[str] = ["kraken"]
    max_pages: int = 50  # Safety limit


@router.post("/api/ocr/batch")
async def trigger_batch_ocr(
    body: BatchOCRRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Queue OCR for multiple pages. Returns list of queued page IDs."""
    from sqlalchemy import select

    if body.page_ids:
        # Specific pages
        result = await db.execute(
            select(Page).where(
                Page.id.in_(body.page_ids),
                Page.ocr_status != "processing",
            )
        )
        pages = list(result.scalars())
    else:
        # All pending pages
        result = await db.execute(
            select(Page)
            .where(Page.ocr_status.in_(["pending", "failed"]))
            .order_by(Page.page_number)
            .limit(body.max_pages)
        )
        pages = list(result.scalars())

    if not pages:
        return {"status": "nothing_to_do", "queued": []}

    queued_ids = []
    for page in pages:
        page.ocr_status = "processing"
        queued_ids.append(page.id)
    await db.commit()

    background_tasks.add_task(_run_batch_ocr_background, queued_ids, body.engines)

    return {"status": "processing", "queued": queued_ids, "count": len(queued_ids)}


@router.get("/api/ocr/batch/status")
async def batch_ocr_status(db: AsyncSession = Depends(get_db)):
    """Get overview of OCR progress across all pages."""
    from sqlalchemy import select, func

    result = await db.execute(
        select(Page.ocr_status, func.count(Page.id)).group_by(Page.ocr_status)
    )
    counts = {row[0]: row[1] for row in result}
    return {
        "total": sum(counts.values()),
        "pending": counts.get("pending", 0),
        "processing": counts.get("processing", 0),
        "done": counts.get("done", 0),
        "failed": counts.get("failed", 0),
    }


async def _run_ocr_background(page_id: int, engines: list[str]):
    """Run OCR in background task."""
    async with async_session() as db:
        try:
            await run_ocr_pipeline(page_id, engines, db)
        except Exception as e:
            page = await db.get(Page, page_id)
            if page:
                page.ocr_status = "failed"
                await db.commit()
            raise


async def _run_batch_ocr_background(page_ids: list[int], engines: list[str]):
    """Run OCR for multiple pages sequentially."""
    import logging
    logger = logging.getLogger(__name__)

    for i, page_id in enumerate(page_ids):
        logger.info("Batch OCR: page %d (%d/%d)", page_id, i + 1, len(page_ids))
        async with async_session() as db:
            try:
                await run_ocr_pipeline(page_id, engines, db)
            except Exception as e:
                logger.error("Batch OCR failed for page %d: %s", page_id, e)
                page = await db.get(Page, page_id)
                if page:
                    page.ocr_status = "failed"
                    await db.commit()
