# 貢獻指南

感謝您有興趣參與故宮珍本叢刊的校對工作！

## 校對流程

### 1. 認領工作

查看 [inventory.json](./inventory.json) 找到 `status: "pending"` 的冊別，在 Issues 中留言認領。

### 2. 取得原始 PDF

故宮珍本叢刊 PDF 需自行取得（檔案過大不存入 git）。

### 3. 校對格式

每頁一個 `.txt` 檔，放在 `texts/GGZBCKXXX/書名/proofread/` 目錄下。

檔案開頭必須有 Dublin Core metadata header：

```
#+TITLE: 淵海子平
#+SOURCE: 故宮珍本叢刊第422冊
#+AUTHOR: [宋] 徐子平
#+PROOFREADER: your-github-username
#+DATE: 2026-04-20
#+PAGE: 1
```

正文標記規則：

- `*` 書名、`**` 卷名、`***` 篇名
- `¶` 原書換行（不是段落分段，是原書版面的物理換行）
- `□` 完全無法辨識的字
- `[?X]` 不確定的字（X 是你的最佳猜測）
- Tab 開頭的行是校對者註記，不會進入正文

### 4. 提交 Pull Request

```bash
git checkout -b proofread/GGZBCK422-yuanhai-vol1
# 修改 proofread/ 下的檔案
git add texts/GGZBCK422/淵海子平/proofread/
git commit -m "校對：淵海子平卷一 pp.1-20"
git push origin proofread/GGZBCK422-yuanhai-vol1
# 開 PR
```

### 5. 驗證

提交前請跑驗證腳本：

```bash
python scripts/validate.py texts/GGZBCK422/淵海子平/proofread/
```

確認沒有格式錯誤。

## 品質標準

- 每頁至少對照原圖校對一遍
- 不確定的字用 `[?X]` 標記，不要猜測後直接寫入
- 保留原書的分行（用 `¶`），不要自行重排段落
- 異體字保留原書寫法，不要現代化
