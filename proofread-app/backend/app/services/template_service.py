"""Character template storage and matching service.

Confirmed characters are saved as 64x64 normalized templates with perceptual hashes.
Matching uses pHash for fast filtering, then SSIM for precision ranking.
"""
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import TEMPLATE_DIR, TEMPLATE_SIZE
from ..models import CharacterTemplate

logger = logging.getLogger(__name__)


def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute DCT-based perceptual hash (no external dependencies).

    1. Resize to (hash_size*4, hash_size*4) grayscale
    2. Apply DCT
    3. Keep top-left hash_size x hash_size low-frequency coefficients
    4. Binarize by median → hex string

    Returns:
        16-char hex string (64-bit hash).
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to hash_size * 4 for DCT
    dct_size = hash_size * 4
    resized = cv2.resize(gray, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
    resized = np.float32(resized)

    # DCT
    dct = cv2.dct(resized)

    # Keep low-frequency block
    dct_low = dct[:hash_size, :hash_size]

    # Median threshold
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()

    # Convert to hex
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def hamming_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    i1 = int(h1, 16)
    i2 = int(h2, 16)
    xor = i1 ^ i2
    return bin(xor).count("1")


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity (simplified, no scikit-image dependency).

    Uses the SSIM formula with default constants.
    Both images should be same size grayscale.
    """
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1 = np.float64(img1)
    img2 = np.float64(img2)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


def _normalize_template(image: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Normalize character image to size x size binarized, preserving aspect ratio.

    1. Convert to grayscale
    2. Otsu binarize (white ink on black bg)
    3. Tight-crop to ink bounding box
    4. Resize preserving aspect ratio, center on white canvas

    Binarization removes background/contrast differences between pages,
    making pHash more consistent for the same character across pages.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Binarize: ink = 255, background = 0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tight crop to ink extent (remove excess background)
    ink_points = np.where(binary > 0)
    if len(ink_points[0]) > 0:
        y1, y2 = ink_points[0].min(), ink_points[0].max() + 1
        x1, x2 = ink_points[1].min(), ink_points[1].max() + 1
        binary = binary[y1:y2, x1:x2]

    h, w = binary.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)

    scale = size / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center on black canvas (background = 0, ink = 255)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


async def save_template(
    text: str,
    page_image_path: str,
    bbox_x: float, bbox_y: float, bbox_w: float, bbox_h: float,
    page_id: int,
    char_id: int,
    db: AsyncSession,
) -> CharacterTemplate:
    """Crop character from page image, normalize, compute pHash, save to DB.

    Args:
        text: The confirmed character text.
        page_image_path: Path to the full page image.
        bbox_x/y/w/h: Character bounding box on the page.
        page_id: Source page ID.
        char_id: Source character ID (reference only).
        db: Database session.

    Returns:
        The created CharacterTemplate record.
    """
    # Crop from page image
    img = cv2.imread(page_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {page_image_path}")

    h_img, w_img = img.shape[:2]
    x1 = max(0, int(bbox_x))
    y1 = max(0, int(bbox_y))
    x2 = min(w_img, int(bbox_x + bbox_w))
    y2 = min(h_img, int(bbox_y + bbox_h))
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(f"Empty crop for char {char_id}")

    # Normalize to TEMPLATE_SIZE x TEMPLATE_SIZE
    normalized = _normalize_template(crop)
    phash = compute_phash(normalized)

    # Check for duplicate (same text + same hash = exact duplicate)
    existing = await db.execute(
        select(CharacterTemplate).where(
            CharacterTemplate.text == text,
            CharacterTemplate.feature_hash == phash,
            CharacterTemplate.is_active == True,
        )
    )
    dup = existing.scalar_one_or_none()
    if dup:
        logger.debug("Duplicate template skipped: text=%s hash=%s", text, phash)
        return dup

    # Save image file
    # Organize by character text: templates/{text_hex}/{hash}.png
    text_hex = text.encode("utf-8").hex()
    char_dir = TEMPLATE_DIR / text_hex
    char_dir.mkdir(parents=True, exist_ok=True)
    image_filename = f"{phash}_{char_id}.png"
    image_path = char_dir / image_filename

    cv2.imwrite(str(image_path), normalized)

    # Save to DB
    template = CharacterTemplate(
        text=text,
        image_path=str(image_path),
        width=TEMPLATE_SIZE,
        height=TEMPLATE_SIZE,
        source_page_id=page_id,
        source_char_id=char_id,
        feature_hash=phash,
    )
    db.add(template)
    await db.flush()

    logger.info("Saved template: text=%s hash=%s id=%d", text, phash, template.id)
    return template


async def match_templates(
    query_image: np.ndarray,
    db: AsyncSession,
    top_k: int = 5,
    max_hamming: int = 18,
) -> list[dict]:
    """Find most similar templates for a query character image.

    Two-phase matching:
    1. pHash: compute query hash, find all templates with Hamming distance < max_hamming
       (18 = ~28% bit difference tolerance for handwritten variation)
    2. SSIM: load candidate images, compute structural similarity, rank by score

    Args:
        query_image: Character image (any size, will be normalized).
        db: Database session.
        top_k: Number of results to return.
        max_hamming: Maximum pHash Hamming distance for candidates.

    Returns:
        List of dicts: [{text, similarity, template_id, image_path}, ...]
    """
    normalized = _normalize_template(query_image)
    query_hash = compute_phash(normalized)

    # Get all active templates
    result = await db.execute(
        select(CharacterTemplate).where(CharacterTemplate.is_active == True)
    )
    templates = result.scalars().all()

    if not templates:
        return []

    # Phase 1: pHash filtering
    candidates = []
    for t in templates:
        if not t.feature_hash:
            continue
        dist = hamming_distance(query_hash, t.feature_hash)
        if dist <= max_hamming:
            candidates.append((t, dist))

    if not candidates:
        return []

    # Sort by Hamming distance first (cheap pre-sort)
    candidates.sort(key=lambda x: x[1])

    # Phase 2: SSIM ranking (only top candidates to limit I/O)
    ssim_limit = min(len(candidates), top_k * 3)
    scored = []
    for t, h_dist in candidates[:ssim_limit]:
        t_img = cv2.imread(t.image_path, cv2.IMREAD_GRAYSCALE)
        if t_img is None:
            continue
        t_normalized = cv2.resize(t_img, (TEMPLATE_SIZE, TEMPLATE_SIZE))
        similarity = compute_ssim(normalized, t_normalized)
        scored.append({
            "text": t.text,
            "similarity": round(similarity, 3),
            "template_id": t.id,
            "image_path": t.image_path,
            "hamming_distance": h_dist,
        })

    # Sort by SSIM descending
    scored.sort(key=lambda x: -x["similarity"])

    # Deduplicate: keep best match per unique text
    seen_texts: set[str] = set()
    deduped = []
    for item in scored:
        if item["text"] not in seen_texts:
            seen_texts.add(item["text"])
            deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


async def get_template_stats(db: AsyncSession) -> dict:
    """Get template database statistics."""
    total = await db.execute(
        select(func.count()).where(CharacterTemplate.is_active == True)
    )
    unique_chars = await db.execute(
        select(func.count(func.distinct(CharacterTemplate.text))).where(
            CharacterTemplate.is_active == True
        )
    )
    return {
        "total_templates": total.scalar() or 0,
        "unique_characters": unique_chars.scalar() or 0,
    }
