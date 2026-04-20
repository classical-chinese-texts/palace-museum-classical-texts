# 故宮珍本叢刊古籍校對計畫

故宮珍本叢刊（國家圖書館藏 / 故宮博物院珍藏善本）占卜命理類古籍的 OCR + 人工校對開放文獻。

## 目標

將故宮珍本叢刊中 31 冊占卜命理類古籍數位化：

1. **PaddleOCR 粗讀** — 將掃描 PDF 轉為逐頁文字檔
2. **人工校對** — 對照原圖逐字修正 OCR 錯誤
3. **知識庫整合** — 轉為結構化 JSON，供 AI 服務使用

## 格式標準

採用 [Mandoku/Kanripo](https://github.com/kanripo) 風格標記，與台灣 TELDAP / 中研院數位人文標準相容：

| 標記 | 用途 | 範例 |
|------|------|------|
| `*` | 標題層級（`*` 書名、`**` 卷名、`***` 篇名） | `* 淵海子平` |
| `¶` | 原書換行（非段落斷行） | `天干者¶甲乙丙丁` |
| `□` | 不可辨識字 | `天□地支` |
| `[?X]` | 不確定字（X 為最佳猜測） | `[?運]氣所主` |
| `#+` | Dublin Core metadata header | `#+TITLE: 淵海子平` |
| `TAB` | 校對者註記（Tab 開頭，不混入正文） | `	此處原圖模糊` |

## 目錄結構

```
texts/
├── GGZBCK422/
│   ├── 淵海子平/
│   │   ├── raw/001.txt      ← PaddleOCR 粗讀（每頁一檔）
│   │   └── proofread/001.txt ← 人工校對後
│   └── 神相全編/ ...
├── GGZBCK415/ ...
└── GGZBCK423/ ...

scripts/                      ← OCR + 驗證 + JSON 建構腳本
inventory.json                ← 31 冊處理進度追蹤
knowledge/                    ← 最終 JSON 產物（release artifact）
```

## 目前進度

| 狀態 | 冊數 |
|------|------|
| 已整合進知識庫 JSON | 3（GGZBCK415、422、423） |
| 待處理 | 28 |
| **合計** | **31** |

詳見 [inventory.json](./inventory.json)。

## 使用方式

### OCR 粗讀

```bash
# 需要 PaddleOCR 環境
python scripts/ocr_pdf.py --pdf /path/to/GGZBCKXXX.pdf --volume GGZBCKXXX
```

### 格式驗證

```bash
python scripts/validate.py texts/GGZBCK422/淵海子平/proofread/
```

### 建構知識庫 JSON

```bash
python scripts/build_json.py --output knowledge/palace_museum_classical.json
```

## 授權

- **校對文字**：CC BY-SA 4.0（與 Kanripo 相同）
- **原始古籍**：公有領域（Public Domain）
- **工具腳本**：MIT License

## 致謝

- [Kanripo](https://github.com/kanripo) — 30,000+ 開放古籍，本計畫格式參考來源
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 中文 OCR 引擎
- 國家圖書館 / 故宮博物院 — 原始文獻來源
