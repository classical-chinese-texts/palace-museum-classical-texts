import type { Page } from '../utils/api';

interface Props {
  page: Page | null;
  totalChars: number;
  confirmedChars: number;
  lowConfChars: number;
}

export function StatusBar({ page, totalChars, confirmedChars, lowConfChars }: Props) {
  const progress = totalChars > 0 ? Math.round((confirmedChars / totalChars) * 100) : 0;

  return (
    <div className="h-8 bg-gray-800 border-t border-gray-700 flex items-center px-4 gap-6 text-xs text-gray-400">
      {page ? (
        <>
          <span>第 {page.page_number} 頁</span>
          <span className="text-gray-600">|</span>
          <span>{totalChars} 字</span>
          <span className="text-gray-600">|</span>
          <span className="text-yellow-400">{lowConfChars} 個低信心</span>
          <span className="text-gray-600">|</span>
          <span className="text-green-400">{confirmedChars} 個已確認</span>
          <span className="text-gray-600">|</span>
          <div className="flex items-center gap-2">
            <div className="w-24 bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-green-500 h-1.5 rounded-full transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span>{progress}%</span>
          </div>
          <div className="ml-auto text-gray-600">
            Tab:下一個低信心 | Enter:確認 | Space:跳過 | 1-9:選候選字
          </div>
        </>
      ) : (
        <span>請選擇頁面</span>
      )}
    </div>
  );
}
