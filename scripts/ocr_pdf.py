#!/usr/bin/env python3
"""
故宮珍本叢刊 PDF → Mandoku 格式分頁 txt

用法：
    python scripts/ocr_pdf.py --pdf /path/to/GGZBCK422.pdf --volume GGZBCK422 --book 淵海子平 --start 1 --end 200
    python scripts/ocr_pdf.py --pdf /path/to/GGZBCK422.pdf --volume GGZBCK422 --book 淵海子平 --pages 10,15,20-30
    python scripts/ocr_pdf.py --pdf /path/to/GGZBCK422.pdf --volume GGZBCK422 --book 淵海子平 --model v5-server

輸出：texts/GGZBCK422/淵海子平/raw/001.txt ~ NNN.txt（每頁一檔）

依賴：paddleocr >= 3.4.0 + paddlepaddle >= 3.0.0 + pdf2image (poppler)

模型選項：
    hybrid   — PP-OCRv4 mobile 偵測 + PP-OCRv5 server 識別（預設，CPU 最佳平衡）
    v4-mobile — PP-OCRv4 mobile（最快，品質略低）
    v5-server — PP-OCRv5 server（最準，CPU 極慢，建議搭配 GPU）
"""
import argparse
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ── PDF → 圖片 ──────────────────────────────────────────
try:
    from pdf2image import convert_from_path
except ImportError:
    print("需要安裝 pdf2image: pip install pdf2image")
    print("同時需要 poppler: brew install poppler (macOS)")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("需要安裝 PaddleOCR: pip install 'paddleocr>=3.4.0'")
    sys.exit(1)

MODEL_PRESETS = {
    "hybrid": {
        "text_detection_model_name": "PP-OCRv4_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_server_rec",
    },
    "v4-mobile": {
        "ocr_version": "PP-OCRv4",
    },
    "v5-server": {
        "ocr_version": "PP-OCRv5",
    },
}


def parse_page_spec(spec: str, total_pages: int) -> list[int]:
    """解析頁碼規格，支援 '1-10,15,20-30' 格式。回傳 0-based index list。"""
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            pages.extend(range(a - 1, min(b, total_pages)))
        else:
            p = int(part) - 1
            if 0 <= p < total_pages:
                pages.append(p)
    return sorted(set(pages))


def ocr_image(ocr_engine, img, min_text_len: int = 2, min_score: float = 0.3) -> list[tuple[str, float]]:
    """用 PaddleOCR 辨識單張圖片，回傳按 y 座標排序的文字行。

    使用 paddleocr >= 3.4.0 的 predict() API。
    過濾掉過短或低信心度的噪音（頁眉、邊注碎片等）。
    """
    import numpy as np
    import tempfile

    # predict() 需要檔案路徑，將 PIL Image 暫存
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        for result in ocr_engine.predict(tmp_path):
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            polys = result.get("dt_polys", [])

            if not texts:
                return []

            # 建立 (y_center, x_center, text, confidence) 列表
            lines_with_pos = []
            for text, score, poly in zip(texts, scores, polys):
                poly = np.array(poly)
                y_center = (poly[0][1] + poly[2][1]) / 2
                x_center = (poly[0][0] + poly[2][0]) / 2
                # 過濾噪音：太短、低信心度、或空白
                if len(text.strip()) < min_text_len and score < 0.9:
                    continue
                if score < min_score:
                    continue
                lines_with_pos.append((y_center, x_center, text, score))

            if not lines_with_pos:
                return []

            # 古籍直排文字：按 x 座標從右到左排序（閱讀順序）
            # 同一 x 帶（差 < 40px）的文字合併，合併後按 y 排序
            lines_with_pos.sort(key=lambda t: -t[1])  # x 降序（右到左）
            grouped_lines = []
            current_group = [lines_with_pos[0]]

            for item in lines_with_pos[1:]:
                # 比較與群組「錨點」的 x 差距（避免鏈式合併）
                anchor_x = current_group[0][1]
                if abs(item[1] - anchor_x) < 40:
                    current_group.append(item)
                else:
                    # 同欄內按 y 排序（上到下）
                    current_group.sort(key=lambda t: t[0])
                    merged = "".join(t[2] for t in current_group)
                    avg_conf = sum(t[3] for t in current_group) / len(current_group)
                    grouped_lines.append((merged, avg_conf))
                    current_group = [item]

            if current_group:
                current_group.sort(key=lambda t: t[0])
                merged = "".join(t[2] for t in current_group)
                avg_conf = sum(t[3] for t in current_group) / len(current_group)
                grouped_lines.append((merged, avg_conf))

            return grouped_lines
    finally:
        os.unlink(tmp_path)

    return []


def format_mandoku_page(
    lines: list[tuple[str, float]],
    title: str,
    source: str,
    page_num: int,
    model_name: str = "hybrid",
    low_conf_threshold: float = 0.7,
) -> str:
    """將 OCR 結果格式化為 Mandoku 風格 txt。"""
    header = (
        f"#+TITLE: {title}\n"
        f"#+SOURCE: {source}\n"
        f"#+DATE: {date.today().isoformat()}\n"
        f"#+PAGE: {page_num}\n"
        f"#+OCR: PaddleOCR-{model_name}\n"
        "\n"
    )

    body_lines = []
    for text, confidence in lines:
        if confidence < low_conf_threshold:
            # 低信心度：每個字都標記為不確定
            marked = ""
            for ch in text:
                if ch.strip():
                    marked += f"[?{ch}]"
                else:
                    marked += ch
            body_lines.append(marked)
        else:
            body_lines.append(text)

    body = "¶\n".join(body_lines)

    return header + body + "\n"


def main():
    parser = argparse.ArgumentParser(description="故宮珍本叢刊 PDF → Mandoku 格式 OCR")
    parser.add_argument("--pdf", required=True, help="PDF 檔案路徑")
    parser.add_argument("--volume", required=True, help="冊別代碼，如 GGZBCK422")
    parser.add_argument("--book", required=True, help="書名，如 淵海子平")
    parser.add_argument("--start", type=int, default=1, help="起始頁碼（含）")
    parser.add_argument("--end", type=int, default=None, help="結束頁碼（含）")
    parser.add_argument("--pages", type=str, default=None, help="指定頁碼，如 '1-10,15,20-30'")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 轉圖片解析度（預設 300）")
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=list(MODEL_PRESETS.keys()),
        help="OCR 模型組合（預設 hybrid = v4偵測+v5識別，CPU最佳平衡）",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="repo 根目錄（預設為腳本所在的上層目錄）",
    )
    args = parser.parse_args()

    # 確定 repo 根目錄
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).parent.parent

    output_dir = repo_root / "texts" / args.volume / args.book / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = f"故宮珍本叢刊第{args.volume.replace('GGZBCK', '')}冊"

    # 初始化 OCR
    preset = MODEL_PRESETS[args.model]
    print(f"初始化 PaddleOCR（模型: {args.model}）...")
    # hybrid 模式直接指定模型名稱，不需要 lang 參數
    ocr_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        **preset,
    }
    if "ocr_version" in preset:
        ocr_kwargs["lang"] = "ch"
    ocr = PaddleOCR(**ocr_kwargs)

    # 轉換 PDF
    print(f"讀取 PDF: {args.pdf}")
    print(f"DPI: {args.dpi}")

    # 先取得總頁數
    from pdf2image.pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(args.pdf)
    total_pages = info["Pages"]
    print(f"PDF 總頁數: {total_pages}")

    # 決定要處理的頁碼
    if args.pages:
        page_indices = parse_page_spec(args.pages, total_pages)
    else:
        start = args.start - 1
        end = (args.end if args.end else total_pages)
        page_indices = list(range(start, min(end, total_pages)))

    print(f"將處理 {len(page_indices)} 頁")

    # 逐頁處理（避免一次載入全部 PDF 佔太多記憶體）
    processed = 0
    for idx in page_indices:
        page_num = idx + 1
        out_file = output_dir / f"{page_num:03d}.txt"

        if out_file.exists():
            print(f"  跳過第 {page_num} 頁（已存在）")
            continue

        print(f"  處理第 {page_num}/{total_pages} 頁...", end=" ", flush=True)

        # 只轉換這一頁
        images = convert_from_path(
            args.pdf,
            dpi=args.dpi,
            first_page=page_num,
            last_page=page_num,
        )

        if not images:
            print("跳過（轉換失敗）")
            continue

        # OCR
        lines = ocr_image(ocr, images[0])

        if not lines:
            print("跳過（無文字）")
            # 寫入空頁標記
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(
                    f"#+TITLE: {args.book}\n"
                    f"#+SOURCE: {source}\n"
                    f"#+PAGE: {page_num}\n"
                    f"#+OCR: PaddleOCR-{args.model}\n"
                    f"\n"
                    f"\t[此頁無可辨識文字，可能為圖版或空白頁]\n"
                )
            continue

        # 格式化輸出
        content = format_mandoku_page(lines, args.book, source, page_num, args.model)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)

        avg_conf = sum(c for _, c in lines) / len(lines)
        print(f"OK ({len(lines)} 行, 平均信心度 {avg_conf:.2f})")
        processed += 1

    print(f"\n完成！處理 {processed} 頁，輸出至 {output_dir}")


if __name__ == "__main__":
    main()
