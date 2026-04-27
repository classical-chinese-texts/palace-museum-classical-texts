import { useState, useRef, useEffect } from 'react';
import type { Character, CandidateResult } from '../utils/api';
import { getCandidates, getTemplateImageUrl } from '../utils/api';

interface Props {
  char: Character | null;
  onConfirm: (id: number) => void;
  onCorrect: (id: number, text: string) => void;
  onDelete: (id: number) => void;
  onClose: () => void;
}

export function CharEditor({ char, onConfirm, onCorrect, onDelete, onClose }: Props) {
  const [inputText, setInputText] = useState('');
  const [candidates, setCandidates] = useState<CandidateResult | null>(null);
  const [loadingCandidates, setLoadingCandidates] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (char) {
      setInputText('');
      setCandidates(null);
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
                onClick={() => onCorrect(char.id, alt.text)}
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
                  onClick={() => onCorrect(char.id, m.text)}
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
                onClick={() => onCorrect(char.id, c)}
                className="px-3 py-1 bg-amber-900 hover:bg-amber-800 text-white rounded text-lg"
                title="形近字"
              >
                {c}
              </button>
            ))}
            {candidates!.variants.map((v, i) => (
              <button
                key={`v-${i}`}
                onClick={() => onCorrect(char.id, v)}
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
            if (e.key === 'Enter' && inputText) {
              onCorrect(char.id, inputText);
              setInputText('');
            }
          }}
          placeholder="輸入正確字..."
          className="flex-1 px-2 py-1 bg-gray-700 text-white border border-gray-600 rounded text-lg"
        />
        <button
          onClick={() => {
            if (inputText) {
              onCorrect(char.id, inputText);
              setInputText('');
            }
          }}
          disabled={!inputText}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded"
        >
          修正
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
