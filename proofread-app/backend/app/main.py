"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import engine
from .models import Base
from .routers import projects, pages, characters, ocr, export, templates


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(
    title="OCR Proofreading Platform",
    description="故宮珍本叢刊 OCR 校對平台",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all methods including PATCH
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(projects.router)
app.include_router(pages.router)
app.include_router(characters.router)
app.include_router(ocr.router)
app.include_router(export.router)
app.include_router(templates.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
