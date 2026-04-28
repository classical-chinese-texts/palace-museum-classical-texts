import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import type { Page as PageType, Character } from '../utils/api';
import {
  getProject, listPages, getPageImageUrl,
  triggerOCR, getOCRStatus, uploadPages, exportToRepo,
} from '../utils/api';
import { useCharacters } from '../hooks/useCharacters';
import { useKeyboard } from '../hooks/useKeyboard';
import { CanvasViewer } from '../components/CanvasViewer';
import { CharEditor } from '../components/CharEditor';
import { SidePanel } from '../components/SidePanel';
import { PageNavigator } from '../components/PageNavigator';
import { StatusBar } from '../components/StatusBar';

export function ProofreadPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const pid = parseInt(projectId ?? '0');

  const [projectName, setProjectName] = useState('');
  const [pages, setPages] = useState<PageType[]>([]);
  const [currentPage, setCurrentPage] = useState<PageType | null>(null);
  const [ocrRunning, setOcrRunning] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const zoomRef = useRef<((char: Character) => void) | null>(null);

  const {
    characters, selected, selectedId,
    setSelectedId, load: loadChars, confirmChar, correctChar,
    deleteChar, restoreChar, confirmAll, insertChar, reEvaluate,
  } = useCharacters(currentPage?.id ?? null);

  // Undo delete state
  const [undoInfo, setUndoInfo] = useState<{ id: number; text: string } | null>(null);
  const undoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleDelete = useCallback(async (id: number) => {
    const deleted = await deleteChar(id);
    if (!deleted) return;

    // Clear previous undo timer
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);

    setUndoInfo({ id: deleted.id, text: deleted.display_text });
    undoTimerRef.current = setTimeout(() => setUndoInfo(null), 5000);
  }, [deleteChar]);

  const handleRestore = useCallback(async () => {
    if (!undoInfo) return;
    if (undoTimerRef.current) clearTimeout(undoTimerRef.current);
    try {
      await restoreChar(undoInfo.id);
    } catch { /* char may already be hard-deleted */ }
    setUndoInfo(null);
  }, [undoInfo, restoreChar]);

  // Load project + pages
  useEffect(() => {
    if (!pid) return;
    getProject(pid).then(p => setProjectName(p.book_name));
    listPages(pid).then(setPages);
  }, [pid]);

  // Load characters when page changes
  useEffect(() => {
    if (currentPage) loadChars();
  }, [currentPage?.id, loadChars]);

  const handleSelectPage = useCallback((page: PageType) => {
    setCurrentPage(page);
    setSelectedId(null);
  }, [setSelectedId]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    await uploadPages(pid, e.target.files);
    const updated = await listPages(pid);
    setPages(updated);
    e.target.value = '';
  };

  const handleOCR = async () => {
    if (!currentPage) return;
    setOcrRunning(true);
    try {
      await triggerOCR(currentPage.id, ['kraken']);
      // Poll for completion
      const poll = async () => {
        const status = await getOCRStatus(currentPage.id);
        if (status.ocr_status === 'processing') {
          setTimeout(poll, 2000);
        } else {
          setOcrRunning(false);
          loadChars();
          const updated = await listPages(pid);
          setPages(updated);
          setCurrentPage(updated.find(p => p.id === currentPage.id) ?? null);
        }
      };
      poll();
    } catch {
      setOcrRunning(false);
    }
  };

  const handleExport = async () => {
    const result = await exportToRepo(pid);
    alert(`已匯出 ${result.exported_pages} 頁到 ${result.output_dir}`);
  };

  const handleZoomToChar = useCallback((char: Character) => {
    zoomRef.current?.(char);
  }, []);

  // Keyboard shortcuts
  useKeyboard({
    characters,
    selectedId,
    onSelect: setSelectedId,
    onConfirm: confirmChar,
    onDelete: handleDelete,
    onCorrect: correctChar,
    onZoomToChar: handleZoomToChar,
  });

  const imageUrl = currentPage ? getPageImageUrl(currentPage.id) : null;
  const totalChars = characters.length;
  const confirmedChars = characters.filter(c => c.is_confirmed).length;
  const lowConfChars = characters.filter(c => c.ocr_confidence < 0.7 && !c.is_confirmed).length;

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-white">
      {/* Header */}
      <div className="h-12 bg-gray-800 border-b border-gray-700 flex items-center px-4 gap-4">
        <Link to="/" className="text-gray-400 hover:text-white text-sm">&larr; 專案列表</Link>
        <span className="text-gray-600">|</span>
        <span className="font-medium">{projectName}</span>

        {currentPage && (
          <>
            <span className="text-gray-600">|</span>
            <span className="text-sm text-gray-400">
              頁 {currentPage.page_number} / {pages.length}
            </span>
            <div className="flex gap-2 ml-auto">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
              >
                上傳頁面
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,.pdf"
                onChange={handleUpload}
                className="hidden"
              />
              <button
                onClick={handleOCR}
                disabled={ocrRunning}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                           rounded text-sm"
              >
                {ocrRunning ? 'OCR 執行中...' : '執行 OCR'}
              </button>
              <button
                onClick={() => confirmAll()}
                className="px-3 py-1 bg-green-600 hover:bg-green-500 rounded text-sm"
              >
                批量確認
              </button>
              <button
                onClick={async () => {
                  const n = await reEvaluate();
                  if (n !== undefined) alert(`已重新評估 ${n} 個字元`);
                }}
                className="px-3 py-1 bg-amber-600 hover:bg-amber-500 rounded text-sm"
              >
                重新評估
              </button>
              <button
                onClick={handleExport}
                className="px-3 py-1 bg-purple-600 hover:bg-purple-500 rounded text-sm"
              >
                匯出 Mandoku
              </button>
            </div>
          </>
        )}

        {!currentPage && (
          <div className="flex gap-2 ml-auto">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
            >
              上傳頁面
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.pdf"
              onChange={handleUpload}
              className="hidden"
            />
          </div>
        )}
      </div>

      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Page navigator */}
        <PageNavigator
          pages={pages}
          currentPageId={currentPage?.id ?? null}
          onSelectPage={handleSelectPage}
        />

        {/* Center: Canvas */}
        <div className="flex-1 relative flex flex-col">
          {currentPage ? (
            <CanvasViewer
              imageUrl={imageUrl}
              characters={characters}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onZoomToChar={zoomRef}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              <div className="text-center">
                <p className="text-2xl mb-2">選擇或上傳頁面</p>
                <p className="text-sm">從左側選擇頁面，或點擊「上傳頁面」新增</p>
              </div>
            </div>
          )}

          {/* Character editor popup */}
          <CharEditor
            char={selected}
            onConfirm={confirmChar}
            onCorrect={correctChar}
            onDelete={handleDelete}
            onInsertAfter={(charId, colIdx) => insertChar(colIdx, charId)}
            onClose={() => setSelectedId(null)}
          />
        </div>

        {/* Right: Side panel */}
        <SidePanel
          characters={characters}
          selectedId={selectedId}
          pageId={currentPage?.id ?? null}
          onSelect={setSelectedId}
          onInsert={insertChar}
        />
      </div>

      {/* Undo delete toast */}
      {undoInfo && (
        <div className="fixed bottom-16 left-1/2 -translate-x-1/2 z-50
          bg-gray-700 border border-gray-600 rounded-lg shadow-xl px-4 py-2
          flex items-center gap-3 text-sm animate-fade-in">
          <span className="text-gray-300">
            已刪除「<span className="text-white font-medium">{undoInfo.text}</span>」
          </span>
          <button
            onClick={handleRestore}
            className="px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-medium"
          >
            復原
          </button>
          <button
            onClick={() => { if (undoTimerRef.current) clearTimeout(undoTimerRef.current); setUndoInfo(null); }}
            className="text-gray-500 hover:text-gray-300"
          >
            &times;
          </button>
        </div>
      )}

      {/* Status bar */}
      <StatusBar
        page={currentPage}
        totalChars={totalChars}
        confirmedChars={confirmedChars}
        lowConfChars={lowConfChars}
      />
    </div>
  );
}
