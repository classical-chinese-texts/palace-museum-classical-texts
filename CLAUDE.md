# 故宮珍本叢刊校對平台 — CLAUDE.md

## ⚠️ 飛輪教訓

### #1 圖像演算法必須先分析再寫（2026-04-26，坎卦教訓）

**事件**：寫 CCA（Connected Component Analysis）字級邊界偵測，未分析實際圖片特徵就拍腦袋設參數（`merge_max_gap=15`、只搜中心點），部署後 bbox 框大量失準（「太」字高度只有 8px），用戶重新 OCR 後辨識品質暴跌。

**根因**：「水流而不盈」——有演算法流動但沒有數據積累支撐。

**規則**：
1. **寫圖像處理演算法前，必須先跑一張真實圖的統計分析**（component 數量、平均大小、間距分佈），用數據決定參數
2. **改動 OCR pipeline 的任何環節，必須在改動前後各跑一次同一頁，對比結果**（字數、bbox 尺寸統計、Playwright 截圖）
3. **新演算法必須有 fallback**：CCA 失敗時回退到 ink_tighten，不能讓整個 pipeline 掛掉
4. **一次只替換一個環節**：不要同時改 bbox 演算法 + 模板系統 + 前端，先確認 bbox 品質再往下

### #2 參數不能拍腦袋

**規則**：圖像處理的數值參數（kernel size、threshold、gap、padding）必須標註來源：
- `# 依據 P1 統計：median char height = 119px` ← 有依據
- `merge_max_gap = 15` ← 禁止，無依據

## 通用規則

- 遵循根目錄 `/Users/yui/CLAUDE.md` 的全域規則
- 先測試再 push
- 一次只改一個問題
- OCR 相關改動必須有前後對比數據
