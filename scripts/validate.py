#!/usr/bin/env python3
"""
驗證 Mandoku 格式文字檔。

用法：
    python scripts/validate.py texts/GGZBCK422/淵海子平/proofread/
    python scripts/validate.py texts/GGZBCK422/淵海子平/raw/001.txt
"""
import argparse
import re
import sys
from pathlib import Path

# 必要的 metadata header
REQUIRED_HEADERS = {"#+TITLE", "#+SOURCE", "#+PAGE"}
OPTIONAL_HEADERS = {"#+AUTHOR", "#+PROOFREADER", "#+DATE", "#+OCR", "#+EDITION"}

# 合法標記
VALID_MARKERS = {
    "¶",           # 原書換行
    "□",           # 不可辨識字
    "*",           # 書名
    "**",          # 卷名
    "***",         # 篇名
    "****",        # 節名
}


class ValidationError:
    def __init__(self, file: str, line: int, message: str, severity: str = "error"):
        self.file = file
        self.line = line
        self.message = message
        self.severity = severity  # "error" or "warning"

    def __str__(self):
        icon = "E" if self.severity == "error" else "W"
        return f"[{icon}] {self.file}:{self.line}: {self.message}"


def validate_file(filepath: Path) -> list[ValidationError]:
    """驗證單一 Mandoku 格式文字檔。"""
    errors = []
    fname = filepath.name

    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(ValidationError(fname, 0, "檔案不是有效的 UTF-8 編碼"))
        return errors

    lines = content.split("\n")

    if not lines:
        errors.append(ValidationError(fname, 0, "檔案為空"))
        return errors

    # 1. 檢查 metadata headers
    found_headers = set()
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("#+"):
            key = line.split(":")[0].strip() if ":" in line else line.strip()
            found_headers.add(key)
            header_end = i + 1

            # 檢查 header 格式
            if ":" not in line:
                errors.append(ValidationError(fname, i + 1, f"Header 缺少冒號: {line}"))
            elif not line.split(":", 1)[1].strip():
                errors.append(
                    ValidationError(
                        fname, i + 1, f"Header 值為空: {line}", "warning"
                    )
                )
        elif line.strip() == "":
            header_end = i + 1
            break
        else:
            break

    missing = REQUIRED_HEADERS - found_headers
    if missing:
        errors.append(
            ValidationError(fname, 1, f"缺少必要 header: {', '.join(sorted(missing))}")
        )

    unknown_headers = found_headers - REQUIRED_HEADERS - OPTIONAL_HEADERS
    if unknown_headers:
        errors.append(
            ValidationError(
                fname,
                1,
                f"未知的 header: {', '.join(sorted(unknown_headers))}",
                "warning",
            )
        )

    # 2. 檢查正文
    for i in range(header_end, len(lines)):
        line = lines[i]

        # Tab 開頭 = 校對者註記，跳過內容檢查
        if line.startswith("\t"):
            continue

        # 空行跳過
        if not line.strip():
            continue

        # 檢查 [?X] 標記格式
        uncertain = re.findall(r"\[\?[^\]]*\]", line)
        for mark in uncertain:
            inner = mark[2:-1]  # 去掉 [? 和 ]
            if len(inner) > 3:
                errors.append(
                    ValidationError(
                        fname,
                        i + 1,
                        f"不確定字標記過長（應為 1-3 字）: {mark}",
                        "warning",
                    )
                )

        # 檢查未閉合的方括號
        open_brackets = line.count("[") - line.count("[?")
        close_brackets = line.count("]") - len(uncertain)
        # 簡化檢查：[?X] 已處理，其他 [ ] 應該成對
        remaining = re.sub(r"\[\?[^\]]*\]", "", line)
        if remaining.count("[") != remaining.count("]"):
            errors.append(
                ValidationError(
                    fname, i + 1, f"未閉合的方括號: {line[:60]}...", "warning"
                )
            )

        # 檢查可疑的 OCR 雜訊：連續超過 5 個相同字元
        for match in re.finditer(r"(.)\1{4,}", line):
            errors.append(
                ValidationError(
                    fname,
                    i + 1,
                    f"可疑重複字元（可能是 OCR 雜訊）: '{match.group()[:10]}...'",
                    "warning",
                )
            )

    return errors


def validate_directory(dirpath: Path) -> list[ValidationError]:
    """驗證目錄下所有 .txt 檔案。"""
    all_errors = []
    txt_files = sorted(dirpath.glob("*.txt"))

    if not txt_files:
        all_errors.append(
            ValidationError(str(dirpath), 0, "目錄下沒有 .txt 檔案")
        )
        return all_errors

    # 檢查頁碼連續性
    page_nums = []
    for f in txt_files:
        try:
            num = int(f.stem)
            page_nums.append(num)
        except ValueError:
            all_errors.append(
                ValidationError(
                    f.name, 0, f"檔名不是純數字: {f.name}", "warning"
                )
            )

    if page_nums:
        page_nums.sort()
        for i in range(len(page_nums) - 1):
            if page_nums[i + 1] - page_nums[i] > 1:
                gap_start = page_nums[i] + 1
                gap_end = page_nums[i + 1] - 1
                all_errors.append(
                    ValidationError(
                        str(dirpath),
                        0,
                        f"頁碼不連續：缺少第 {gap_start}-{gap_end} 頁",
                        "warning",
                    )
                )

    # 驗證每個檔案
    for f in txt_files:
        all_errors.extend(validate_file(f))

    return all_errors


def main():
    parser = argparse.ArgumentParser(description="驗證 Mandoku 格式文字檔")
    parser.add_argument("path", help="要驗證的檔案或目錄")
    parser.add_argument("--strict", action="store_true", help="將 warning 也視為錯誤")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        errors = validate_file(target)
    elif target.is_dir():
        errors = validate_directory(target)
    else:
        print(f"路徑不存在: {args.path}")
        sys.exit(1)

    # 輸出結果
    error_count = sum(1 for e in errors if e.severity == "error")
    warning_count = sum(1 for e in errors if e.severity == "warning")

    for err in errors:
        print(err)

    print(f"\n結果：{error_count} 個錯誤, {warning_count} 個警告")

    if error_count > 0 or (args.strict and warning_count > 0):
        sys.exit(1)
    else:
        print("通過！")
        sys.exit(0)


if __name__ == "__main__":
    main()
