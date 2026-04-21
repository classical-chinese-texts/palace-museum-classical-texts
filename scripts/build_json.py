#!/usr/bin/env python3
"""
將校對完成的 Mandoku 格式 txt 合併成知識庫 JSON。

用法：
    python scripts/build_json.py --output knowledge/palace_museum_classical.json
    python scripts/build_json.py --volume GGZBCK422 --output knowledge/ggzbck422.json

輸出格式與 MiaoXian 知識庫（bugua_classics.json）對齊：
{
  "source": "故宮珍本叢刊...",
  "category": "...",
  "articles": [
    {
      "id": "ggzbck422_yuanhai_ziping",
      "name": "淵海子平",
      "source_book": "GGZBCK422",
      "source": "故宮珍本叢刊第422冊",
      "keywords": [...],
      "chapters": [
        {"name": "卷一", "content": "..."}
      ]
    }
  ]
}
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

# 書名 → 英文 ID 和關鍵詞映射
BOOK_METADATA = {
    "梅花易數": {
        "id": "meihua_yishu",
        "keywords": ["梅花", "易數", "占卦", "先天", "後天", "邵康節", "體用", "八卦", "五行"],
    },
    "靈棋經": {
        "id": "lingqi_jing",
        "keywords": ["靈棋", "占卜", "黃石公", "課目", "占辭"],
    },
    "大六壬神課金口訣": {
        "id": "liuren_jinkoujue",
        "keywords": ["六壬", "金口訣", "天干", "地支", "神課", "占法"],
    },
    "邵子易數": {
        "id": "shaozi_yishu",
        "keywords": ["邵子", "易數", "伏羲", "太極", "八卦", "先天"],
    },
    "大六壬課經集": {
        "id": "liuren_kejingji",
        "keywords": ["六壬", "課經", "課式", "盤圖", "占例"],
    },
    "淵海子平": {
        "id": "yuanhai_ziping",
        "keywords": ["子平", "八字", "天干", "地支", "五行", "十神", "格局", "用神", "納音"],
    },
    "神相全編": {
        "id": "shenxiang_quanbian",
        "keywords": ["面相", "相法", "五官", "氣色", "骨法", "形體", "十二宮"],
    },
    "子平集要": {
        "id": "ziping_jiyao",
        "keywords": ["子平", "八字", "命理", "格局", "用神", "大運"],
    },
    "御定五星精義": {
        "id": "wuxing_jingyi",
        "keywords": ["五星", "七政", "四餘", "命宮", "大限", "星命", "十二宮"],
    },
    "星平集腋統宗": {
        "id": "xingping_jiye",
        "keywords": ["星平", "星命", "五星", "命理", "子平", "格局"],
    },
    "神相彙編": {
        "id": "shenxiang_huibian",
        "keywords": ["面相", "相法", "五官", "骨法", "氣色", "十二宮"],
    },
    "子平管見集解": {
        "id": "ziping_guanjian",
        "keywords": ["子平", "八字", "命理", "格局", "用神", "管見"],
    },
    "六壬眎斯": {
        "id": "liuren_shisi",
        "keywords": ["六壬", "眎斯", "課式", "占法"],
    },
    "火珠林": {
        "id": "huozhulin",
        "keywords": ["火珠林", "六爻", "占卜", "卜筮"],
    },
    "增刪卜易": {
        "id": "zengshan_buyi",
        "keywords": ["增刪卜易", "六爻", "野鶴老人", "占卜", "用神", "世應", "動爻", "飛伏", "月建", "日辰"],
    },
    "卜筮正宗": {
        "id": "bushi_zhengzong",
        "keywords": ["卜筮正宗", "六爻", "王洪緒", "占卜", "用神", "原神", "忌神", "仇神", "進神", "退神"],
    },
}


def parse_mandoku_file(filepath: Path) -> dict:
    """解析 Mandoku 格式文字檔，回傳 metadata + 正文。"""
    content = filepath.read_text(encoding="utf-8")
    lines = content.split("\n")

    metadata = {}
    body_lines = []
    in_header = True

    for line in lines:
        if in_header:
            if line.startswith("#+") and ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
            elif line.strip() == "":
                in_header = False
            else:
                in_header = False
                body_lines.append(line)
        else:
            # 跳過校對者註記（Tab 開頭）
            if not line.startswith("\t"):
                body_lines.append(line)

    return {
        "metadata": metadata,
        "body": "\n".join(body_lines).strip(),
    }


def detect_chapters(full_text: str, book_name: str) -> list[dict]:
    """將合併的正文按卷/章分割成 chapters。"""
    # 尋找標題標記（Mandoku 風格：* 開頭）
    title_pattern = re.compile(r"^(\*{1,4})\s*(.+)$", re.MULTILINE)

    matches = list(title_pattern.finditer(full_text))

    if not matches:
        # 沒有標題標記，嘗試用傳統卷標記分割
        vol_pattern = re.compile(
            r"^(.*(?:卷[一二三四五六七八九十百之上下]+|卷首|卷末|目錄|序).*)$",
            re.MULTILINE,
        )
        matches_vol = list(vol_pattern.finditer(full_text))

        if not matches_vol:
            return [{"name": book_name, "content": full_text}]

        chapters = []
        for i, m in enumerate(matches_vol):
            start = m.start()
            end = matches_vol[i + 1].start() if i + 1 < len(matches_vol) else len(full_text)
            segment = full_text[start:end].strip()
            if len(segment) > 50:
                chapters.append({"name": m.group(1).strip()[:40], "content": segment})
        return chapters if chapters else [{"name": book_name, "content": full_text}]

    # 有 Mandoku 標題標記
    chapters = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        name = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segment = full_text[start:end].strip()
        if len(segment) > 50:
            chapters.append({"name": name, "content": segment})

    return chapters if chapters else [{"name": book_name, "content": full_text}]


def process_book(volume_code: str, book_name: str, book_dir: Path) -> dict | None:
    """處理一本書的所有頁面，合併後分章。"""
    # 優先用 proofread/，沒有就用 raw/
    proofread_dir = book_dir / "proofread"
    raw_dir = book_dir / "raw"

    if proofread_dir.exists() and list(proofread_dir.glob("*.txt")):
        source_dir = proofread_dir
        ocr_status = "proofread"
    elif raw_dir.exists() and list(raw_dir.glob("*.txt")):
        source_dir = raw_dir
        ocr_status = "raw_ocr"
    else:
        return None

    # 按頁碼排序讀取
    txt_files = sorted(source_dir.glob("*.txt"), key=lambda f: f.stem)
    if not txt_files:
        return None

    all_body = []
    for f in txt_files:
        parsed = parse_mandoku_file(f)
        if parsed["body"]:
            all_body.append(parsed["body"])

    if not all_body:
        return None

    full_text = "\n\n".join(all_body)

    # 清理 OCR 雜訊
    # 移除連續重複字元行
    lines = full_text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(set(stripped.replace("¶", ""))) <= 2 and len(stripped) > 5:
            continue  # 疑似 OCR 雜訊行
        cleaned.append(line)
    full_text = "\n".join(cleaned)

    # 分章
    chapters = detect_chapters(full_text, book_name)

    # 查找 metadata
    meta = BOOK_METADATA.get(book_name, {})
    book_id = meta.get("id", book_name.lower().replace(" ", "_"))
    keywords = meta.get("keywords", [])
    volume_num = volume_code.replace("GGZBCK", "")

    return {
        "id": f"{volume_code.lower()}_{book_id}",
        "name": book_name,
        "source_book": volume_code,
        "source": f"故宮珍本叢刊第{volume_num}冊",
        "ocr_status": ocr_status,
        "keywords": keywords,
        "chapters": chapters,
    }


def main():
    parser = argparse.ArgumentParser(description="建構知識庫 JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="輸出 JSON 路徑",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default=None,
        help="只處理指定冊別（如 GGZBCK422）",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="repo 根目錄",
    )
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        repo_root = Path(__file__).parent.parent

    texts_dir = repo_root / "texts"

    if not texts_dir.exists():
        print(f"texts/ 目錄不存在: {texts_dir}")
        sys.exit(1)

    all_articles = []
    source_volumes = []

    # 遍歷所有冊別
    volume_dirs = sorted(texts_dir.iterdir())
    for vol_dir in volume_dirs:
        if not vol_dir.is_dir():
            continue

        volume_code = vol_dir.name
        if args.volume and volume_code != args.volume:
            continue

        # 遍歷該冊下的每本書
        book_dirs = sorted(vol_dir.iterdir())
        books_in_vol = []
        for book_dir in book_dirs:
            if not book_dir.is_dir():
                continue

            book_name = book_dir.name
            print(f"  處理 {volume_code}/{book_name}...", end=" ", flush=True)

            article = process_book(volume_code, book_name, book_dir)
            if article:
                all_articles.append(article)
                ch_count = len(article["chapters"])
                ch_chars = sum(len(c["content"]) for c in article["chapters"])
                print(f"{ch_count} 章, {ch_chars:,} 字")
                books_in_vol.append(book_name)
            else:
                print("跳過（無內容）")

        if books_in_vol:
            source_volumes.append(f"{volume_code} — {'、'.join(books_in_vol)}")

    if not all_articles:
        print("沒有找到任何可處理的內容")
        sys.exit(0)

    # 統計
    total_chars = sum(
        len(ch["content"]) for art in all_articles for ch in art["chapters"]
    )
    total_chapters = sum(len(art["chapters"]) for art in all_articles)

    knowledge = {
        "source": "故宮珍本叢刊（國家圖書館 / 故宮博物院珍藏善本）",
        "category": "故宮珍本叢刊古籍・占卜命理類",
        "format": "Mandoku/Kanripo 風格校對",
        "license": "CC BY-SA 4.0",
        "build_date": date.today().isoformat(),
        "meta": {
            "total_books": len(all_articles),
            "total_chapters": total_chapters,
            "total_characters": total_chars,
            "source_volumes": source_volumes,
        },
        "articles": all_articles,
    }

    # 寫入
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size
    print(f"\n完成！")
    print(f"  書籍數：{len(all_articles)}")
    print(f"  章節數：{total_chapters}")
    print(f"  總字數：{total_chars:,}")
    print(f"  檔案大小：{file_size / 1024:.1f} KB")
    print(f"  輸出路徑：{output_path}")


if __name__ == "__main__":
    main()
