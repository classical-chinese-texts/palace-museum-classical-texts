"""Project CRUD router."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Project, Page

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    volume_code: str
    book_name: str


class ProjectOut(BaseModel):
    id: int
    volume_code: str
    book_name: str
    total_pages: int
    completed_pages: int
    created_at: str

    model_config = {"from_attributes": True}


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    return [ProjectOut(
        id=p.id, volume_code=p.volume_code, book_name=p.book_name,
        total_pages=p.total_pages, completed_pages=p.completed_pages,
        created_at=p.created_at.isoformat(),
    ) for p in projects]


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(Project).where(
            Project.volume_code == body.volume_code,
            Project.book_name == body.book_name,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Project already exists")

    project = Project(volume_code=body.volume_code, book_name=body.book_name)
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return ProjectOut(
        id=project.id, volume_code=project.volume_code, book_name=project.book_name,
        total_pages=0, completed_pages=0, created_at=project.created_at.isoformat(),
    )


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return ProjectOut(
        id=project.id, volume_code=project.volume_code, book_name=project.book_name,
        total_pages=project.total_pages, completed_pages=project.completed_pages,
        created_at=project.created_at.isoformat(),
    )


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: int, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    await db.delete(project)
    await db.commit()
