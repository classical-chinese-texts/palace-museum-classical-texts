"""Export router — Mandoku format and repo integration."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import TEXTS_DIR
from ..database import get_db
from ..models import Project, Page, Character

router = APIRouter(tags=["export"])


@router.get("/api/pages/{page_id}/export/mandoku")
async def export_mandoku(page_id: int, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")

    project = await db.get(Project, page.project_id)

    result = await db.execute(
        select(Character)
        .where(Character.page_id == page_id, Character.is_deleted == False)
        .order_by(Character.column_index, Character.char_index)
    )
    chars = result.scalars().all()

    # Build Mandoku text
    lines = [
        f"#+TITLE: {project.book_name}",
        f"#+SOURCE: 故宮珍本叢刊第{project.volume_code.replace('GGZBCK', '')}冊",
        f"#+PAGE: {page.page_number}",
        f"#+PROOFREADER: proofread-app",
        "",
    ]

    current_col = -1
    col_text = []

    for c in chars:
        if c.column_index != current_col:
            if col_text:
                lines.append("".join(col_text))
            col_text = []
            current_col = c.column_index

        text = c.display_text
        if not c.is_confirmed and c.ocr_confidence < 0.7:
            text = f"[?{text}]"
        col_text.append(text)

    if col_text:
        lines.append("".join(col_text))

    mandoku_text = "\n".join(lines) + "\n"
    return PlainTextResponse(mandoku_text, media_type="text/plain; charset=utf-8")


class ExportToRepoRequest(BaseModel):
    target_dir: str | None = None  # override output dir


@router.post("/api/projects/{project_id}/export/to-repo")
async def export_to_repo(
    project_id: int,
    body: ExportToRepoRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    # Determine output directory
    if body and body.target_dir:
        output_dir = Path(body.target_dir)
    else:
        output_dir = TEXTS_DIR / project.volume_code / project.book_name / "proofread"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all pages
    result = await db.execute(
        select(Page).where(Page.project_id == project_id).order_by(Page.page_number)
    )
    pages = result.scalars().all()

    exported = 0
    for page in pages:
        char_result = await db.execute(
            select(Character)
            .where(Character.page_id == page.id, Character.is_deleted == False)
            .order_by(Character.column_index, Character.char_index)
        )
        chars = char_result.scalars().all()
        if not chars:
            continue

        lines = [
            f"#+TITLE: {project.book_name}",
            f"#+SOURCE: 故宮珍本叢刊第{project.volume_code.replace('GGZBCK', '')}冊",
            f"#+PAGE: {page.page_number}",
            f"#+PROOFREADER: proofread-app",
            "",
        ]

        current_col = -1
        col_text = []
        for c in chars:
            if c.column_index != current_col:
                if col_text:
                    lines.append("".join(col_text))
                col_text = []
                current_col = c.column_index

            text = c.display_text
            if not c.is_confirmed and c.ocr_confidence < 0.7:
                text = f"[?{text}]"
            col_text.append(text)

        if col_text:
            lines.append("".join(col_text))

        filename = f"{page.page_number:03d}.txt"
        (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1

    return {"exported_pages": exported, "output_dir": str(output_dir)}
