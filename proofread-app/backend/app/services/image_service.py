"""Image processing utilities — cropping, enhancement, thumbnails."""
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image


def crop_character(image_path: str, bbox_x: float, bbox_y: float,
                   bbox_w: float, bbox_h: float, padding: int = 5) -> Image.Image:
    """Crop a single character from the page image with padding."""
    img = Image.open(image_path)
    x1 = max(0, int(bbox_x) - padding)
    y1 = max(0, int(bbox_y) - padding)
    x2 = min(img.width, int(bbox_x + bbox_w) + padding)
    y2 = min(img.height, int(bbox_y + bbox_h) + padding)
    return img.crop((x1, y1, x2, y2))


def generate_thumbnail(image_path: str, size: int = 200) -> bytes:
    """Generate a thumbnail for page navigation."""
    img = Image.open(image_path)
    img.thumbnail((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def enhance_for_ocr(image_path: str) -> np.ndarray:
    """Enhance image for better OCR results using CLAHE."""
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    return enhanced
