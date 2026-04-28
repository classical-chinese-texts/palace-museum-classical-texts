"""Page upload and listing router."""
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image

from ..config import UPLOAD_DIR
from ..database import get_db
from ..models import Project, Page

router = APIRouter(tags=["pages"])


class PageOut(BaseModel):
    id: int
    project_id: int
    page_number: int
    image_path: str
    width: int | None
    height: int | None
    ocr_status: str
    proofread_status: str
    ocr_engine: str | None
    total_chars: int
    confirmed_chars: int
    low_confidence_chars: int

    model_config = {"from_attributes": True}


def _project_upload_dir(project_id: int) -> Path:
    d = UPLOAD_DIR / str(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.get("/api/projects/{project_id}/pages", response_model=list[PageOut])
async def list_pages(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    result = await db.execute(
        select(Page).where(Page.project_id == project_id).order_by(Page.page_number)
    )
    return result.scalars().all()


@router.post("/api/projects/{project_id}/pages/upload", response_model=list[PageOut])
async def upload_pages(
    project_id: int,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    upload_dir = _project_upload_dir(project_id)
    created_pages = []

    # Get current max page number
    result = await db.execute(
        select(Page.page_number).where(Page.project_id == project_id)
        .order_by(Page.page_number.desc()).limit(1)
    )
    max_page = result.scalar() or 0

    for i, file in enumerate(files):
        page_num = max_page + i + 1
        ext = Path(file.filename).suffix.lower() if file.filename else ".png"

        if ext == ".pdf":
            # PDF handling — extract pages using PyMuPDF
            import fitz
            pdf_path = upload_dir / f"upload_{page_num}.pdf"
            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            doc = fitz.open(str(pdf_path))
            for pdf_page_idx in range(len(doc)):
                pix = doc[pdf_page_idx].get_pixmap(dpi=300)
                img_filename = f"page_{max_page + i + pdf_page_idx + 1:04d}.png"
                img_path = upload_dir / img_filename
                pix.save(str(img_path))

                page = Page(
                    project_id=project_id,
                    page_number=max_page + i + pdf_page_idx + 1,
                    image_path=str(img_path),
                    width=pix.width,
                    height=pix.height,
                )
                db.add(page)
                created_pages.append(page)
            doc.close()
            pdf_path.unlink(missing_ok=True)
            max_page += len(doc) - 1  # adjust for multi-page PDF
        else:
            # Image file
            img_filename = f"page_{page_num:04d}{ext}"
            img_path = upload_dir / img_filename
            with open(img_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            with Image.open(img_path) as img:
                w, h = img.size

            page = Page(
                project_id=project_id,
                page_number=page_num,
                image_path=str(img_path),
                width=w,
                height=h,
            )
            db.add(page)
            created_pages.append(page)

    project.total_pages = (
        (await db.execute(
            select(Page.id).where(Page.project_id == project_id)
        )).all().__len__()
        + len(created_pages)
    )
    await db.commit()
    for p in created_pages:
        await db.refresh(p)

    return created_pages


@router.get("/api/pages/{page_id}", response_model=PageOut)
async def get_page(page_id: int, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    return page


@router.get("/api/pages/{page_id}/image")
async def get_page_image(page_id: int, db: AsyncSession = Depends(get_db)):
    page = await db.get(Page, page_id)
    if not page:
        raise HTTPException(404, "Page not found")
    image_path = Path(page.image_path)
    if not image_path.exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(str(image_path))
