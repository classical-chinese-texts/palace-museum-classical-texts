"""SQLAlchemy models for OCR proofreading."""
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    volume_code: Mapped[str] = mapped_column(String(20), nullable=False)
    book_name: Mapped[str] = mapped_column(String(100), nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    completed_pages: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    pages: Mapped[list["Page"]] = relationship(back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("volume_code", "book_name"),)


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    ocr_status: Mapped[str] = mapped_column(String(20), default="pending")
    proofread_status: Mapped[str] = mapped_column(String(20), default="pending")
    ocr_engine: Mapped[str | None] = mapped_column(String(50))
    total_chars: Mapped[int] = mapped_column(Integer, default=0)
    confirmed_chars: Mapped[int] = mapped_column(Integer, default=0)
    low_confidence_chars: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project: Mapped["Project"] = relationship(back_populates="pages")
    characters: Mapped[list["Character"]] = relationship(back_populates="page", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("project_id", "page_number"),)


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"))
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    column_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_index: Mapped[int] = mapped_column(Integer, nullable=False)
    ocr_text: Mapped[str | None] = mapped_column(Text)
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    ocr_engine: Mapped[str | None] = mapped_column(String(50))
    alternatives: Mapped[dict | None] = mapped_column(SQLiteJSON, default=list)
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    corrected_text: Mapped[str | None] = mapped_column(Text)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    page: Mapped["Page"] = relationship(back_populates="characters")
    corrections: Mapped[list["CorrectionLog"]] = relationship(back_populates="character")

    __table_args__ = (
        Index("idx_chars_order", "page_id", "column_index", "char_index"),
        Index("idx_chars_confidence", "page_id", "ocr_confidence"),
    )

    @property
    def display_text(self) -> str:
        return self.corrected_text or self.ocr_text or "□"


class CharacterTemplate(Base):
    __tablename__ = "character_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    source_page_id: Mapped[int | None] = mapped_column(ForeignKey("pages.id", ondelete="SET NULL"))
    source_char_id: Mapped[int | None] = mapped_column(Integer)
    feature_hash: Mapped[str | None] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source_page: Mapped["Page | None"] = relationship()

    __table_args__ = (
        Index("idx_template_text_hash", "text", "feature_hash"),
    )


class CorrectionLog(Base):
    __tablename__ = "correction_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    character_id: Mapped[int] = mapped_column(ForeignKey("characters.id"))
    old_text: Mapped[str | None] = mapped_column(Text)
    new_text: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    character: Mapped["Character"] = relationship(back_populates="corrections")
