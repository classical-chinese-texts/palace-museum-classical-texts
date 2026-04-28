#!/usr/bin/env python3
"""Batch deskew + OCR for all BSB pages.

Usage: python3 batch_deskew_ocr.py [--start N] [--end N] [--force]
"""
import os, sys, warnings, sqlite3, time, argparse, tempfile, traceback
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

import cv2
import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), 'proofread.db')
PROJECT_ID = 3


@dataclass
class DC:
    bbox_x: float; bbox_y: float; bbox_w: float; bbox_h: float
    text: str | None; confidence: float; column_index: int = 0
    char_index: int = 0; engine: str = "grid"
    alternatives: list = field(default_factory=list)
    _deleted: bool = False


def init_models():
    """Initialize OCR models (once)."""
    from pipeline.engines import init_paddle_ocr
    from paddlex import create_model
    ocr = init_paddle_ocr(model='hybrid')
    rec = create_model('PP-OCRv5_server_rec')
    return ocr, rec


def process_page(page_id, page_number, image_path, ocr, rec):
    """Run full deskew pipeline on one page. Returns (n_active, n_recognized) or None."""
    from app.services.deskew_service import deskew_page, detect_columns_and_chars

    corrected_path = image_path.rsplit('.', 1)[0] + '_corrected.png'

    # Step 1: Deskew
    deskew_result = deskew_page(image_path, save_path=corrected_path)
    if deskew_result is not None:
        img = deskew_result.corrected_image
        working_path = corrected_path
    else:
        img = cv2.imread(image_path)
        if img is None:
            return None
        working_path = image_path
        corrected_path = image_path  # no correction

    h_img, w_img = img.shape[:2]

    # Step 2: Grid detection
    bboxes = detect_columns_and_chars(img)
    if not bboxes:
        # Blank or cover page — mark as done with 0 chars
        _save_results(page_id, corrected_path, w_img, h_img, [])
        return (0, 0)

    # Step 3: PaddleOCR on full image
    ocr_lines = []
    try:
        results = ocr.predict(working_path)
        if results:
            res = results[0].json['res']
            for poly, text, score in zip(
                res.get('dt_polys', []),
                res.get('rec_texts', []),
                res.get('rec_scores', []),
            ):
                if not text or not text.strip() or score < 0.3:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                ocr_lines.append({
                    'text': text, 'score': score,
                    'x': min(xs), 'y': min(ys),
                    'w': max(xs) - min(xs), 'h': max(ys) - min(ys),
                })
    except Exception as e:
        print(f"    PaddleOCR error: {e}")

    # Step 4: Match OCR text to grid bboxes
    chars = []
    matched = set()

    for line in ocr_lines:
        line_chars = list(line['text'])
        lx, ly, lw, lh = line['x'], line['y'], line['w'], line['h']
        is_vert = lh > lw * 1.2

        n = len(line_chars)
        if n == 0:
            continue
        char_h = lh / n if is_vert else lh
        char_w = lw / n if not is_vert else lw

        for ci, ch in enumerate(line_chars):
            if is_vert:
                cx = lx + lw / 2
                cy = ly + ci * char_h + char_h / 2
            else:
                cx = lx + ci * char_w + char_w / 2
                cy = ly + lh / 2

            best_idx = None
            best_dist = float('inf')
            for bi, gb in enumerate(bboxes):
                if bi in matched:
                    continue
                gcx = gb['x'] + gb['w'] / 2
                gcy = gb['y'] + gb['h'] / 2
                dist = ((cx - gcx)**2 + (cy - gcy)**2)**0.5
                if dist < best_dist and dist < max(gb['w'], gb['h']) * 1.5:
                    best_dist = dist
                    best_idx = bi

            if best_idx is not None:
                matched.add(best_idx)
                gb = bboxes[best_idx]
                chars.append(DC(
                    bbox_x=gb['x'], bbox_y=gb['y'],
                    bbox_w=gb['w'], bbox_h=gb['h'],
                    text=ch, confidence=line['score'],
                    engine='grid+paddle',
                ))

    for bi, gb in enumerate(bboxes):
        if bi not in matched:
            chars.append(DC(
                bbox_x=gb['x'], bbox_y=gb['y'],
                bbox_w=gb['w'], bbox_h=gb['h'],
                text=None, confidence=0.0, engine='grid',
            ))

    # Step 5: Rec on unmatched
    unmatched = [c for c in chars if not c.text and c.bbox_w >= 15 and c.bbox_h >= 15]
    for c in unmatched:
        pad = 10
        y1 = max(0, int(c.bbox_y) - pad)
        y2 = min(h_img, int(c.bbox_y + c.bbox_h) + pad)
        x1 = max(0, int(c.bbox_x) - pad)
        x2 = min(w_img, int(c.bbox_x + c.bbox_w) + pad)
        crop = img[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, crop)
            try:
                for result in rec.predict(tmp.name):
                    r = result.json['res']
                    text = r['rec_text'].strip()
                    score = r['rec_score']
                    if text and score > 0.05:
                        c.text = text
                        c.confidence = score
                        c.engine = 'grid+paddle_rec'
                    break
            except:
                pass
            finally:
                os.unlink(tmp.name)

    # Step 6: Filter border noise
    if chars:
        min_x = min(c.bbox_x for c in chars)
        col0_threshold = min_x + 30
        col0 = [c for c in chars if c.bbox_x < col0_threshold]
        noise = sum(1 for c in col0 if c.bbox_w < 20 or (not c.text) or c.confidence < 0.1)
        if noise > len(col0) * 0.5:
            for c in col0:
                c._deleted = True

        # Also filter rightmost edge noise
        max_x = max(c.bbox_x + c.bbox_w for c in chars)
        col_right = [c for c in chars if c.bbox_x + c.bbox_w > max_x - 30 and not c._deleted]
        noise_r = sum(1 for c in col_right if c.bbox_w < 20 or (not c.text) or c.confidence < 0.1)
        if noise_r > len(col_right) * 0.5:
            for c in col_right:
                c._deleted = True

    # Assign reading order
    active = [c for c in chars if not c._deleted]
    if not active:
        _save_results(page_id, corrected_path, w_img, h_img, [])
        return (0, 0)

    active.sort(key=lambda c: -(c.bbox_x + c.bbox_w / 2))
    columns = []
    cur_col = [active[0]]
    for c in active[1:]:
        cur_center = sum(ch.bbox_x + ch.bbox_w / 2 for ch in cur_col) / len(cur_col)
        c_center = c.bbox_x + c.bbox_w / 2
        if abs(c_center - cur_center) < 40:
            cur_col.append(c)
        else:
            columns.append(cur_col)
            cur_col = [c]
    columns.append(cur_col)

    for col_idx, col in enumerate(columns):
        col.sort(key=lambda c: c.bbox_y)
        for ci, c in enumerate(col):
            c.column_index = col_idx
            c.char_index = ci

    n_recognized = sum(1 for c in active if c.text)
    _save_results(page_id, corrected_path, w_img, h_img, active)
    return (len(active), n_recognized)


def _save_results(page_id, image_path, width, height, chars):
    """Save OCR results to SQLite."""
    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()
    now = datetime.utcnow().isoformat()

    # Delete existing
    cur.execute("DELETE FROM correction_log WHERE character_id IN (SELECT id FROM characters WHERE page_id=?)", (page_id,))
    cur.execute("DELETE FROM characters WHERE page_id=?", (page_id,))

    # Insert chars
    for c in chars:
        cur.execute("""
            INSERT INTO characters (page_id, bbox_x, bbox_y, bbox_w, bbox_h,
                column_index, char_index, ocr_text, ocr_confidence, ocr_engine,
                alternatives, is_confirmed, is_deleted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]', 0, 0, ?)
        """, (page_id, c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h,
              c.column_index, c.char_index, c.text, c.confidence, c.engine, now))

    # Update page
    low_conf = sum(1 for c in chars if c.confidence < 0.7)
    cur.execute("""
        UPDATE pages SET image_path=?, width=?, height=?,
            ocr_status='done', ocr_engine='deskew',
            total_chars=?, low_confidence_chars=?, confirmed_chars=0
        WHERE id=?
    """, (image_path, width, height, len(chars), low_conf, page_id))

    db.commit()
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=999)
    parser.add_argument('--force', action='store_true', help='Re-run even if already done')
    args = parser.parse_args()

    db = sqlite3.connect(DB_PATH)
    cur = db.cursor()

    if args.force:
        rows = cur.execute("""
            SELECT id, page_number, image_path FROM pages
            WHERE project_id=? AND page_number BETWEEN ? AND ?
            ORDER BY page_number
        """, (PROJECT_ID, args.start, args.end)).fetchall()
    else:
        rows = cur.execute("""
            SELECT id, page_number, image_path FROM pages
            WHERE project_id=? AND page_number BETWEEN ? AND ?
            AND (ocr_status != 'done' OR ocr_engine != 'deskew')
            ORDER BY page_number
        """, (PROJECT_ID, args.start, args.end)).fetchall()
    db.close()

    total = len(rows)
    print(f"=== Batch deskew OCR: {total} pages (P{args.start}-P{args.end}) ===")
    print(f"    Force={args.force}")

    if total == 0:
        print("Nothing to do.")
        return

    # Init models once
    print("Loading models...", flush=True)
    ocr, rec = init_models()
    print("Models loaded.", flush=True)

    stats = {'ok': 0, 'blank': 0, 'fail': 0, 'total_chars': 0, 'total_rec': 0}
    t0 = time.time()

    for i, (page_id, page_num, image_path) in enumerate(rows):
        t1 = time.time()
        try:
            # Use original (non-corrected) path for deskew input
            orig_path = image_path.replace('_corrected', '')
            result = process_page(page_id, page_num, orig_path, ocr, rec)

            if result is None:
                print(f"[{i+1}/{total}] P{page_num}: FAILED (cannot read image)")
                stats['fail'] += 1
            elif result[0] == 0:
                elapsed = time.time() - t1
                print(f"[{i+1}/{total}] P{page_num}: blank/cover ({elapsed:.1f}s)")
                stats['blank'] += 1
            else:
                n_active, n_rec = result
                elapsed = time.time() - t1
                pct = n_rec / n_active * 100 if n_active else 0
                print(f"[{i+1}/{total}] P{page_num}: {n_rec}/{n_active} chars ({pct:.0f}%) [{elapsed:.1f}s]")
                stats['ok'] += 1
                stats['total_chars'] += n_active
                stats['total_rec'] += n_rec
        except Exception as e:
            elapsed = time.time() - t1
            print(f"[{i+1}/{total}] P{page_num}: ERROR ({elapsed:.1f}s) {e}")
            traceback.print_exc()
            stats['fail'] += 1
            # Mark as failed
            db2 = sqlite3.connect(DB_PATH)
            db2.execute("UPDATE pages SET ocr_status='failed' WHERE id=?", (page_id,))
            db2.commit()
            db2.close()

    total_time = time.time() - t0
    avg = total_time / total if total else 0
    pct = stats['total_rec'] / stats['total_chars'] * 100 if stats['total_chars'] else 0

    print(f"\n=== Done in {total_time:.0f}s ({avg:.1f}s/page) ===")
    print(f"    OK: {stats['ok']}, Blank: {stats['blank']}, Failed: {stats['fail']}")
    print(f"    Total chars: {stats['total_chars']}, Recognized: {stats['total_rec']} ({pct:.0f}%)")


if __name__ == '__main__':
    main()
