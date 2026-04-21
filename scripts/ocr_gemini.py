#!/usr/bin/env python3
"""
故宮珍本叢刊 PDF → Gemini Vision OCR → Mandoku 格式分頁 txt

用法：
    python scripts/ocr_gemini.py --pdf /tmp/GGZBCK416.pdf --volume GGZBCK416 --book 增刪卜易
    python scripts/ocr_gemini.py --pdf /tmp/GGZBCK416.pdf --volume GGZBCK416 --book 增刪卜易 --start 1 --end 50
    python scripts/ocr_gemini.py --pdf /tmp/GGZBCK416.pdf --volume GGZBCK416 --book 增刪卜易 --pages 10,15,20-30
    python scripts/ocr_gemini.py --pdf /tmp/GGZBCK416.pdf --volume GGZBCK416 --book 增刪卜易 --dpi 200

輸出：texts/GGZBCK416/增刪卜易/raw/001.txt ~ NNN.txt（每頁一檔）

依賴：google-genai + pdf2image (poppler)
    pip install google-genai pdf2image Pillow

Gemini 免費版限制：~15 RPM → 每頁間隔 4 秒
"""
import argparse
import base64
import io
import os
import sys
import time
from datetime import date
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("需要安裝 google-genai: pip install google-genai")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path
except ImportError:
    print("需要安裝 pdf2image: pip install pdf2image")
    print("同時需要 poppler: brew install poppler (macOS)")
    sys.exit(1)

from PIL import Image

# ── Gemini Vision OCR prompt ─────────────────────────────────
OCR_PROMPT = """你是一個專門辨識中國古籍（故宮珍本叢刊）的 OCR 專家。

請將此圖片中的文字轉錄為純文字，遵循以下規則：

1. **閱讀方向**：古籍是直排右到左。從右欄開始，逐欄向左，每欄從上到下。
2. **使用正體字（繁體中文）**：所有文字必須用繁體中文。
3. **分行**：每一直欄的文字用 ¶ 換行符分隔。
4. **不確定的字**：如果某個字辨識不確定，用 □ 標記。
5. **注釋/小字**：如果有夾注或小字注釋，用（）括起來。
6. **標題**：如果看到卷名或章名（通常字體較大或在頁首），用 * 開頭標記，如：* 卷一
7. **忽略**：忽略頁碼、書名邊框線、裝訂痕跡等非正文內容。
8. **空白頁/圖版頁**：如果此頁無文字（空白、全圖版），回覆 [無文字]

只輸出轉錄的文字，不要加任何解說。"""


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


def ocr_page_gemini(client, model_name: str, img: Image.Image) -> str:
    """用 Gemini Vision 辨識單頁圖片，回傳文字。失敗重試 3 次。"""
    # 轉成 JPEG bytes（比 PNG 小，省 bandwidth）
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=OCR_PROMPT),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            if response.text:
                return response.text.strip()
            return "[無文字]"
        except Exception as e:
            err_str = str(e)
            # 429 rate limit: 解析 retryDelay 或用較長等待
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                import re
                m = re.search(r"retryDelay.*?(\d+)", err_str)
                wait = int(m.group(1)) + 5 if m else 60
            else:
                wait = 4 * (2 ** attempt)  # 4, 8, 16 seconds
            print(f"\n    ⚠ Gemini API 錯誤 (attempt {attempt+1}/3): {e}")
            if attempt < 4:
                print(f"    等待 {wait} 秒後重試...", flush=True)
                time.sleep(wait)
            else:
                print(f"    ✗ 5 次重試皆失敗，跳過此頁", flush=True)
                return f"[OCR 失敗: {e}]"


def format_mandoku_page(
    text: str,
    title: str,
    source: str,
    page_num: int,
) -> str:
    """將 Gemini OCR 結果格式化為 Mandoku 風格 txt。"""
    header = (
        f"#+TITLE: {title}\n"
        f"#+SOURCE: {source}\n"
        f"#+DATE: {date.today().isoformat()}\n"
        f"#+PAGE: {page_num}\n"
        f"#+OCR: Gemini-Vision\n"
        "\n"
    )
    return header + text + "\n"


def main():
    parser = argparse.ArgumentParser(description="故宮珍本叢刊 PDF → Gemini Vision OCR → Mandoku 格式")
    parser.add_argument("--pdf", required=True, help="PDF 檔案路徑")
    parser.add_argument("--volume", required=True, help="冊別代碼，如 GGZBCK416")
    parser.add_argument("--book", required=True, help="書名，如 增刪卜易")
    parser.add_argument("--start", type=int, default=1, help="起始頁碼（含）")
    parser.add_argument("--end", type=int, default=None, help="結束頁碼（含）")
    parser.add_argument("--pages", type=str, default=None, help="指定頁碼，如 '1-10,15,20-30'")
    parser.add_argument("--dpi", type=int, default=200, help="PDF 轉圖片解析度（預設 200，省 bandwidth）")
    parser.add_argument("--delay", type=float, default=4.0, help="每頁間隔秒數（預設 4.0，免費版 15 RPM）")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini 模型名稱（預設 gemini-2.0-flash，免費 200 RPD）",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="repo 根目錄（預設為腳本所在的上層目錄）",
    )
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("錯誤：需要設定 GEMINI_API_KEY 環境變數")
        print("  export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # 確定 repo 根目錄
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).parent.parent

    output_dir = repo_root / "texts" / args.volume / args.book / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = f"故宮珍本叢刊第{args.volume.replace('GGZBCK', '')}冊"

    # 初始化 Gemini client
    print(f"初始化 Gemini client（模型: {args.model}）...")
    client = genai.Client(api_key=api_key)

    # 讀取 PDF 資訊
    print(f"讀取 PDF: {args.pdf}")
    info = pdfinfo_from_path(args.pdf)
    total_pages = info["Pages"]
    print(f"PDF 總頁數: {total_pages}")
    print(f"DPI: {args.dpi}")
    print(f"每頁間隔: {args.delay}s")

    # 決定要處理的頁碼
    if args.pages:
        page_indices = parse_page_spec(args.pages, total_pages)
    else:
        start = args.start - 1
        end = (args.end if args.end else total_pages)
        page_indices = list(range(start, min(end, total_pages)))

    # 過濾已存在的頁面
    remaining = []
    skipped = 0
    for idx in page_indices:
        page_num = idx + 1
        out_file = output_dir / f"{page_num:03d}.txt"
        if out_file.exists():
            skipped += 1
        else:
            remaining.append(idx)

    if skipped > 0:
        print(f"跳過 {skipped} 頁（已存在）")
    print(f"將處理 {len(remaining)} 頁")

    if not remaining:
        print("所有頁面已完成！")
        return

    # 逐頁處理
    processed = 0
    failed = 0
    start_time = time.time()

    for i, idx in enumerate(remaining):
        page_num = idx + 1
        out_file = output_dir / f"{page_num:03d}.txt"

        # 進度顯示
        elapsed = time.time() - start_time
        if processed > 0:
            eta = elapsed / processed * (len(remaining) - i)
            eta_str = f"ETA {eta/60:.0f}m"
        else:
            eta_str = ""

        print(f"  [{i+1}/{len(remaining)}] 第 {page_num}/{total_pages} 頁... {eta_str}", end=" ", flush=True)

        # 轉換此頁為圖片
        images = convert_from_path(
            args.pdf,
            dpi=args.dpi,
            first_page=page_num,
            last_page=page_num,
        )

        if not images:
            print("跳過（轉換失敗）")
            failed += 1
            continue

        # Gemini Vision OCR
        text = ocr_page_gemini(client, args.model, images[0])

        if text == "[無文字]" or text.startswith("[OCR 失敗"):
            print(f"({text})")
            # 仍寫入檔案作為記錄
            content = format_mandoku_page(
                f"\t{text}",
                args.book,
                source,
                page_num,
            )
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(content)
            if text.startswith("[OCR 失敗"):
                failed += 1
            continue

        # 格式化輸出
        content = format_mandoku_page(text, args.book, source, page_num)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(content)

        char_count = len(text.replace("¶", "").replace("\n", "").strip())
        print(f"OK ({char_count} 字)")
        processed += 1

        # Rate limit delay（最後一頁不用等）
        if i < len(remaining) - 1:
            time.sleep(args.delay)

    elapsed_total = time.time() - start_time
    print(f"\n完成！")
    print(f"  成功: {processed} 頁")
    print(f"  失敗: {failed} 頁")
    print(f"  耗時: {elapsed_total/60:.1f} 分鐘")
    print(f"  輸出: {output_dir}")


if __name__ == "__main__":
    main()
