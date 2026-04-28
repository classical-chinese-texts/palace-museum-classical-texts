import { useMemo, useEffect, useState } from 'react';
import type { Character } from '../utils/api';
import { groupByColumn } from '../utils/readingOrder';
import { getExportMandoku } from '../utils/api';

interface Props {
  characters: Character[];
  selectedId: number | null;
  pageId: number | null;
  onSelect: (id: number | null) => void;
  onInsert?: (columnIndex: number, afterCharId?: number, beforeFirst?: boolean) => void;
}

export function SidePanel({ characters, selectedId, pageId, onSelect, onInsert }: Props) {
  const [mandokuPreview, setMandokuPreview] = useState('');
  const columns = useMemo(() => groupByColumn(characters), [characters]);

  useEffect(() => {
    if (!pageId) return;
    getExportMandoku(pageId).then(setMandokuPreview).catch(() => {});
  }, [pageId, characters]);

  return (
    <div className="w-64 bg-gray-800 border-l border-gray-700 flex flex-col overflow-hidden">
      {/* Legend */}
      <div className="p-3 border-b border-gray-700">
        <div className="text-xs text-gray-400 mb-2 font-medium">圖例</div>
        <div className="flex flex-col gap-1 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-green-500 inline-block" />
            <span className="text-gray-300">已確認</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-yellow-500 inline-block" />
            <span className="text-gray-300">低信心度</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded bg-red-500 inline-block" />
            <span className="text-gray-300">已修正</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded border border-dashed border-fuchsia-500 inline-block" />
            <span className="text-gray-300">待辨識 (空隙發現)</span>
          </div>
        </div>
      </div>

      {/* Column preview */}
      <div className="flex-1 overflow-auto p-3">
        <div className="text-xs text-gray-400 mb-2 font-medium">文字預覽 (直排)</div>
        <div className="flex gap-2 flex-row-reverse justify-end">
          {Array.from(columns.entries()).map(([colIdx, chars]) => (
            <div key={colIdx} className="group/col flex flex-col items-center">
              <div className="text-[10px] text-gray-600 mb-1">欄{colIdx}</div>
              {onInsert && (
                <button
                  onClick={() => onInsert(colIdx, undefined, true)}
                  className="w-7 h-4 hidden group-hover/col:flex items-center justify-center
                    text-[10px] text-gray-600 hover:text-white hover:bg-gray-600
                    border border-dashed border-gray-700 hover:border-gray-400 rounded mb-0.5"
                  title="在此欄頂部新增字元"
                >
                  +
                </button>
              )}
              {chars.map((c, i) => {
                let bg = '';
                if (c.id === selectedId) bg = 'bg-blue-600';
                else if (c.is_confirmed) bg = 'bg-green-900/50';
                else if (c.corrected_text) bg = 'bg-red-900/50';
                else if (c.ocr_engine === 'gap_fill') bg = 'bg-fuchsia-900/50';
                else if (c.ocr_confidence < 0.7) bg = 'bg-yellow-900/50';

                return (
                  <div key={c.id} className="flex flex-col items-center">
                    <button
                      onClick={() => onSelect(c.id)}
                      className={`w-7 h-7 flex items-center justify-center text-sm text-white
                        hover:bg-gray-600 border border-transparent hover:border-gray-500
                        rounded ${bg}`}
                      title={`${c.display_text} (${Math.round(c.ocr_confidence * 100)}%)`}
                    >
                      {c.display_text}
                    </button>
                    {onInsert && i < chars.length - 1 && (
                      <button
                        onClick={() => onInsert(colIdx, c.id)}
                        className="w-5 h-3 hidden group-hover/col:flex items-center justify-center
                          text-[9px] text-gray-600 hover:text-white hover:bg-gray-600
                          rounded opacity-0 group-hover/col:opacity-50 hover:!opacity-100
                          transition-opacity"
                        title={`在「${c.display_text}」之後插入`}
                      >
                        +
                      </button>
                    )}
                  </div>
                );
              })}
              {onInsert && (
                <button
                  onClick={() => onInsert(colIdx, chars[chars.length - 1]?.id)}
                  className="w-7 h-5 hidden group-hover/col:flex items-center justify-center
                    text-[10px] text-gray-600 hover:text-white hover:bg-gray-600
                    border border-dashed border-gray-700 hover:border-gray-400 rounded mt-0.5"
                  title="在此欄底部新增字元"
                >
                  +
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Mandoku raw preview */}
      <div className="border-t border-gray-700 p-3 max-h-48 overflow-auto">
        <div className="text-xs text-gray-400 mb-1 font-medium">Mandoku 預覽</div>
        <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
          {mandokuPreview || '(尚無資料)'}
        </pre>
      </div>
    </div>
  );
}
