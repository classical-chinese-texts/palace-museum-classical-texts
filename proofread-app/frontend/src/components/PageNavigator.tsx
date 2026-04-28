import type { Page } from '../utils/api';
import { getPageImageUrl } from '../utils/api';

interface Props {
  pages: Page[];
  currentPageId: number | null;
  onSelectPage: (page: Page) => void;
}

export function PageNavigator({ pages, currentPageId, onSelectPage }: Props) {
  return (
    <div className="w-40 bg-gray-850 border-r border-gray-700 overflow-auto flex flex-col">
      <div className="p-2 text-xs text-gray-400 font-medium border-b border-gray-700">
        頁面 ({pages.length})
      </div>
      <div className="flex-1 overflow-auto p-2 space-y-2">
        {pages.map(page => {
          const isActive = page.id === currentPageId;
          const statusColor =
            page.proofread_status === 'done' ? 'border-green-500' :
            page.ocr_status === 'done' ? 'border-yellow-500' :
            'border-gray-600';

          return (
            <button
              key={page.id}
              onClick={() => onSelectPage(page)}
              className={`w-full text-left rounded overflow-hidden border-2 transition-all
                ${isActive ? 'border-blue-500 ring-2 ring-blue-500/30' : statusColor}
                hover:border-blue-400`}
            >
              <img
                src={getPageImageUrl(page.id)}
                alt={`Page ${page.page_number}`}
                className="w-full h-auto bg-gray-800"
                loading="lazy"
              />
              <div className="px-2 py-1 bg-gray-800 flex items-center justify-between">
                <span className="text-xs text-gray-300">P{page.page_number}</span>
                <div className="flex gap-1">
                  {page.ocr_status === 'done' && (
                    <span className="text-[10px] text-yellow-400">OCR</span>
                  )}
                  {page.confirmed_chars > 0 && (
                    <span className="text-[10px] text-green-400">
                      {page.confirmed_chars}/{page.total_chars}
                    </span>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
