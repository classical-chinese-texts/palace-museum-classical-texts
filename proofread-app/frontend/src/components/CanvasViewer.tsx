import { useRef, useState, useCallback, useEffect } from 'react';
import type { Character } from '../utils/api';

const CONF_THRESHOLD = 0.7;

function getBboxBorder(char: Character, isSelected: boolean): string {
  if (isSelected) return '2px solid #3b82f6';
  if (char.is_confirmed) return '1px solid #22c55e';
  if (char.corrected_text) return '1px solid #ef4444';
  // Gap-fill chars (discovered by CCA, no OCR text): magenta dashed border
  if (char.ocr_engine === 'gap_fill') return '1px dashed #d946ef';
  if (char.ocr_confidence < CONF_THRESHOLD) return '1px solid #eab308';
  return '1px solid rgba(148, 163, 184, 0.4)';
}

function getBboxBg(char: Character, isSelected: boolean): string {
  if (isSelected) return 'rgba(59, 130, 246, 0.2)';
  if (char.is_confirmed) return 'rgba(34, 197, 94, 0.08)';
  if (char.corrected_text) return 'rgba(239, 68, 68, 0.08)';
  // Gap-fill chars: subtle magenta background
  if (char.ocr_engine === 'gap_fill') return 'rgba(217, 70, 239, 0.12)';
  if (char.ocr_confidence < CONF_THRESHOLD) return 'rgba(234, 179, 8, 0.1)';
  return 'transparent';
}

interface Props {
  imageUrl: string | null;
  characters: Character[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onZoomToChar: React.MutableRefObject<((char: Character) => void) | null>;
}

export function CanvasViewer({ imageUrl, characters, selectedId, onSelect, onZoomToChar }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef({ x: 0, y: 0, scrollX: 0, scrollY: 0 });
  const didPanRef = useRef(false);

  const handleImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    setZoom(1);
  }, []);

  const resetZoom = useCallback(() => {
    setZoom(1);
    if (containerRef.current) {
      containerRef.current.scrollTo(0, 0);
    }
  }, []);

  // Expose zoomToChar
  useEffect(() => {
    onZoomToChar.current = (char: Character) => {
      if (!naturalSize || !containerRef.current) return;
      const container = containerRef.current;
      const z = 3;
      setZoom(z);
      requestAnimationFrame(() => {
        const baseW = container.clientWidth;
        const scale = baseW / naturalSize.w;
        const cx = (char.bbox_x + char.bbox_w / 2) * scale * z;
        const cy = (char.bbox_y + char.bbox_h / 2) * scale * z;
        container.scrollTo({
          left: cx - container.clientWidth / 2,
          top: cy - container.clientHeight / 2,
          behavior: 'smooth',
        });
      });
    };
  }, [naturalSize, onZoomToChar]);

  // Reset on image change
  useEffect(() => {
    setNaturalSize(null);
    setZoom(1);
  }, [imageUrl]);

  // Pan (drag to scroll) handlers
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    // Only pan with left button on non-bbox areas, or middle button anywhere
    if (e.button !== 0 && e.button !== 1) return;
    const container = containerRef.current;
    if (!container) return;

    setIsPanning(true);
    didPanRef.current = false;
    panStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      scrollX: container.scrollLeft,
      scrollY: container.scrollTop,
    };
    container.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isPanning) return;
    const container = containerRef.current;
    if (!container) return;

    const dx = e.clientX - panStartRef.current.x;
    const dy = e.clientY - panStartRef.current.y;

    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      didPanRef.current = true;
    }

    container.scrollLeft = panStartRef.current.scrollX - dx;
    container.scrollTop = panStartRef.current.scrollY - dy;
  }, [isPanning]);

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    if (!isPanning) return;
    setIsPanning(false);
    containerRef.current?.releasePointerCapture(e.pointerId);
  }, [isPanning]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.8 : 1.25;
    setZoom(z => Math.max(0.5, Math.min(8, z * delta)));
  }, []);

  if (!imageUrl) {
    return <div className="flex-1 bg-gray-900 flex items-center justify-center text-gray-500">No image</div>;
  }

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-auto bg-gray-900"
      style={{ cursor: isPanning ? 'grabbing' : 'grab' }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onWheel={handleWheel}
    >
      {/* Controls — sticky */}
      <div className="sticky top-0 z-10 flex gap-1 justify-end p-2 pointer-events-none">
        <button
          onClick={(e) => { e.stopPropagation(); setZoom(z => Math.min(z * 1.5, 8)); }}
          onPointerDown={(e) => e.stopPropagation()}
          className="px-2 py-1 bg-gray-700/90 text-white text-xs rounded hover:bg-gray-600 pointer-events-auto cursor-pointer"
        >
          Zoom +
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); setZoom(z => Math.max(z / 1.5, 0.5)); }}
          onPointerDown={(e) => e.stopPropagation()}
          className="px-2 py-1 bg-gray-700/90 text-white text-xs rounded hover:bg-gray-600 pointer-events-auto cursor-pointer"
        >
          Zoom -
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); resetZoom(); }}
          onPointerDown={(e) => e.stopPropagation()}
          className="px-2 py-1 bg-gray-700/90 text-white text-xs rounded hover:bg-gray-600 pointer-events-auto cursor-pointer"
        >
          Reset
        </button>
      </div>

      {/* Image wrapper — width scales with zoom */}
      <div
        className="relative"
        style={{ width: `${zoom * 100}%` }}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          onLoad={handleImageLoad}
          alt="page"
          className="block w-full select-none"
          draggable={false}
        />

        {/* Bbox overlay */}
        {naturalSize && (
          <div className="absolute inset-0">
            {characters.map(char => {
              const isSelected = char.id === selectedId;
              const left = (char.bbox_x / naturalSize.w) * 100;
              const top = (char.bbox_y / naturalSize.h) * 100;
              const width = (char.bbox_w / naturalSize.w) * 100;
              const height = (char.bbox_h / naturalSize.h) * 100;

              return (
                <div
                  key={char.id}
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!didPanRef.current) onSelect(char.id);
                  }}
                  title={`${char.display_text} (${Math.round(char.ocr_confidence * 100)}%)`}
                  style={{
                    position: 'absolute',
                    left: `${left}%`,
                    top: `${top}%`,
                    width: `${width}%`,
                    height: `${height}%`,
                    border: getBboxBorder(char, isSelected),
                    backgroundColor: getBboxBg(char, isSelected),
                    cursor: 'pointer',
                    boxSizing: 'border-box',
                  }}
                />
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
