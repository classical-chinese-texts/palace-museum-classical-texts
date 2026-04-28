import { useState, useRef, useEffect, useCallback } from 'react';
import type { Character, CandidateResult } from '../utils/api';
import { getCandidates, getTemplateImageUrl } from '../utils/api';

interface Props {
  char: Character | null;
  onConfirm: (id: number) => Promise<void> | void;
  onCorrect: (id: number, text: string) => Promise<void> | void;
  onDelete: (id: number) => Promise<void> | void;
  onInsertAfter?: (charId: number, columnIndex: number) => void;
  onClose: () => void;
}

export function CharEditor({ char, onConfirm, onCorrect, onDelete, onInsertAfter, onClose }: Props) {
  const [inputText, setInputText] = useState('');
  const [candidates, setCandidates] = useState<CandidateResult | null>(null);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const [saving, setSaving] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const doCorrect = useCallback(async (id: number, text: string) => {
    setSaving(true);
    try {
      await onCorrect(id, text);
      setFlash(text);
      setTimeout(() => setFlash(null), 1500);
    } catch (e) {
      setFlash(`失敗: ${e instanceof Error ? e.message : '未知錯誤'}`);
      setTimeout(() => setFlash(null), 3000);
    } finally {
      setSaving(false);
    }
  }, [onCorrect]);

  useEffect(() => {
    if (char) {
      setInputText('');
      setCandidates(null);
      setFlash(null);
      // Fetch candidates (template matches + confusables + variants)
      setLoadingCandidates(true);
      getCandidates(char.id)
        .then(setCandidates)
        .catch(() => setCandidates(null))
        .finally(() => setLoadingCandidates(false));
    }
  }, [char?.id]);

  if (!char) return null;

  const confPercent = Math.round(char.ocr_confidence * 100);
  const confColor = char.ocr_confidence >= 0.7 ? 'bg-green-500' : 'bg-yellow-500';

  const hasTemplates = candidates?.template_matches && candidates.template_matches.length > 0;
  const hasConfusables = candidates?.confusables && candidates.confusables.length > 0;
  const hasVariants = candidates?.variants && candidates.variants.length > 0;

  return (
    <div className="absolute bottom-20 left-1/2 -translate-x-1/2 bg-gray-800 border border-gray-600 rounded-lg shadow-xl p-4 z-20 min-w-72 max-w-96">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm">
          [{char.column_index},{char.char_index}] {char.ocr_engine ?? ''}
        </span>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-lg">
          &times;
        </button>
      </div>

      {/* Gap-fill hint */}
      {char.ocr_engine === 'gap_fill' && !char.corrected_text && (
        <div className="text-xs text-fuchsia-400 mb-2 bg-fuchsia-900/30 rounded px-2 py-1">
          空隙發現：OCR 未偵測此字，請目視辨識後輸入
        </div>
      )}

      {/* Success/error flash */}
      {flash && (
        <div className={`text-xs mb-2 rounded px-2 py-1 ${
          flash.startsWith('失敗') ? 'text-red-400 bg-red-900/30' : 'text-green-400 bg-green-900/30'
        }`}>
          {flash.startsWith('失敗') ? flash : `已修正為「${flash}」`}
        </div>
      )}

      {/* Main OCR result */}
      <div className="flex items-center gap-4 mb-3">
        <span className="text-5xl font-serif text-white">{char.display_text}</span>
        <div className="flex-1">
          <div className="text-sm text-gray-400 mb-1">OCR: {char.ocr_text ?? '?'}</div>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${confColor}`}
                style={{ width: `${confPercent}%` }}
              />
            </div>
            <span className="text-xs text-gray-400">{confPercent}%</span>
          </div>
        </div>
      </div>

      {/* Alternatives from OCR engines */}
      {char.alternatives && char.alternatives.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">候選字:</div>
          <div className="flex gap-1 flex-wrap">
            {char.alternatives.map((alt, i) => (
              <button
                key={i}
                onClick={() => doCorrect(char.id, alt.text)}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-lg"
                title={`${alt.engine} (${Math.round(alt.confidence * 100)}%)`}
              >
                {alt.text}
                <span className="text-xs text-gray-400 ml-1">{i + 1}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Template matches */}
      {hasTemplates && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">模板建議:</div>
          <div className="flex gap-1 flex-wrap">
            {candidates!.template_matches.map((m) => {
              const isTextMatch = m.hamming_distance === -1;
              return (
                <button
                  key={m.template_id}
                  onClick={() => doCorrect(char.id, m.text)}
                  className={`flex items-center gap-1 px-2 py-1 text-white rounded ${
                    isTextMatch
                      ? 'bg-gray-700 hover:bg-gray-600'
                      : 'bg-indigo-900 hover:bg-indigo-800'
                  }`}
                  title={isTextMatch
                    ? `已確認模板「${m.text}」`
                    : `圖像相似度 ${Math.round(m.similarity * 100)}%`
                  }
                >
                  <img
                    src={getTemplateImageUrl(m.template_id)}
                    alt={m.text}
                    className="w-6 h-6 rounded border border-gray-600"
                  />
                  <span className="text-lg">{m.text}</span>
                  <span className="text-xs text-gray-400">
                    {isTextMatch ? '字' : `${Math.round(m.similarity * 100)}%`}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Confusables + Variants */}
      {(hasConfusables || hasVariants) && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">
            {hasConfusables && '形近字'}
            {hasConfusables && hasVariants && ' / '}
            {hasVariants && '異體字'}:
          </div>
          <div className="flex gap-1 flex-wrap">
            {candidates!.confusables.map((c, i) => (
              <button
                key={`c-${i}`}
                onClick={() => doCorrect(char.id, c)}
                className="px-3 py-1 bg-amber-900 hover:bg-amber-800 text-white rounded text-lg"
                title="形近字"
              >
                {c}
              </button>
            ))}
            {candidates!.variants.map((v, i) => (
              <button
                key={`v-${i}`}
                onClick={() => doCorrect(char.id, v)}
                className="px-3 py-1 bg-teal-900 hover:bg-teal-800 text-white rounded text-lg"
                title="異體字（正字）"
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      )}

      {loadingCandidates && (
        <div className="text-xs text-gray-500 mb-3">載入建議中...</div>
      )}

      {/* Manual input */}
      <div className="flex gap-2 mb-3">
        <input
          ref={inputRef}
          type="text"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && inputText && !saving) {
              doCorrect(char.id, inputText);
              setInputText('');
            }
          }}
          placeholder="輸入正確字..."
          className="flex-1 px-2 py-1 bg-gray-700 text-white border border-gray-600 rounded text-lg"
        />
        <button
          onClick={() => {
            if (inputText && !saving) {
              doCorrect(char.id, inputText);
              setInputText('');
            }
          }}
          disabled={!inputText || saving}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded"
        >
          {saving ? '...' : '修正'}
        </button>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={() => onConfirm(char.id)}
          className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-500 text-white rounded font-medium"
        >
          確認 (Enter)
        </button>
        <button
          onClick={onClose}
          className="flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded"
        >
          跳過 (Space)
        </button>
        {onInsertAfter && (
          <button
            onClick={() => onInsertAfter(char.id, char.column_index)}
            className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-sm"
            title="在此字下方插入新字元"
          >
            +插入
          </button>
        )}
        <button
          onClick={() => onDelete(char.id)}
          className="px-3 py-2 bg-red-600 hover:bg-red-500 text-white rounded"
        >
          刪除
        </button>
      </div>
    </div>
  );
}
