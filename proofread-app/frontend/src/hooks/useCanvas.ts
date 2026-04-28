import { useRef, useEffect, useCallback } from 'react';
import { Canvas, FabricImage, Rect } from 'fabric';
import type { Character } from '../utils/api';

const CONF_THRESHOLD = 0.7;

function getBboxBorder(char: Character, isSelected: boolean): string {
  if (isSelected) return '#3b82f6';
  if (char.is_confirmed) return '#22c55e';
  if (char.corrected_text) return '#ef4444';
  if (char.ocr_confidence < CONF_THRESHOLD) return '#eab308';
  return 'rgba(148, 163, 184, 0.5)';
}

function getBboxFill(char: Character, isSelected: boolean): string {
  if (isSelected) return 'rgba(59, 130, 246, 0.15)';
  if (char.is_confirmed) return 'rgba(34, 197, 94, 0.08)';
  if (char.corrected_text) return 'rgba(239, 68, 68, 0.08)';
  if (char.ocr_confidence < CONF_THRESHOLD) return 'rgba(234, 179, 8, 0.08)';
  return 'transparent';
}

const rectCharMap = new WeakMap<Rect, number>();

export function useCanvas(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  containerRef: React.RefObject<HTMLDivElement | null>,
  imageUrl: string | null,
  characters: Character[],
  selectedId: number | null,
  onSelect: (id: number | null) => void,
) {
  const fabricRef = useRef<Canvas | null>(null);
  const rectsRef = useRef<Map<number, Rect>>(new Map());
  const scaleRef = useRef<number>(1);

  // Initialize canvas
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = new Canvas(canvasRef.current, {
      selection: false,
      enableRetinaScaling: false,  // Disable retina — we handle scaling ourselves
    });
    fabricRef.current = canvas;

    return () => {
      canvas.dispose();
      fabricRef.current = null;
    };
  }, [canvasRef]);

  // Load image and draw bboxes
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas || !imageUrl) return;

    // Load image as a native HTML Image to get natural dimensions
    const htmlImg = new Image();
    htmlImg.crossOrigin = 'anonymous';
    htmlImg.onload = () => {
      canvas.clear();
      rectsRef.current.clear();

      const naturalW = htmlImg.naturalWidth;
      const naturalH = htmlImg.naturalHeight;

      // Size canvas to container width
      const containerW = containerRef.current?.clientWidth ?? 800;
      const scale = containerW / naturalW;
      scaleRef.current = scale;

      const canvasH = Math.round(naturalH * scale);

      // Set canvas dimensions (CSS pixels, not device pixels)
      canvas.setDimensions({ width: containerW, height: canvasH });

      // Create Fabric image from the loaded HTML image
      const fabricImg = new FabricImage(htmlImg, {
        scaleX: scale,
        scaleY: scale,
        left: 0,
        top: 0,
        selectable: false,
        evented: false,
      });
      canvas.add(fabricImg);

      // Add character bboxes
      for (const char of characters) {
        const isSelected = char.id === selectedId;
        const rect = new Rect({
          left: char.bbox_x * scale,
          top: char.bbox_y * scale,
          width: char.bbox_w * scale,
          height: char.bbox_h * scale,
          fill: getBboxFill(char, isSelected),
          stroke: getBboxBorder(char, isSelected),
          strokeWidth: isSelected ? 2.5 : 0.8,
          selectable: false,
          hoverCursor: 'pointer',
        });
        rectCharMap.set(rect, char.id);
        canvas.add(rect);
        rectsRef.current.set(char.id, rect);
      }

      canvas.renderAll();
    };
    htmlImg.src = imageUrl;
  }, [imageUrl, characters, selectedId, containerRef]);

  // Handle click
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    const handler = (e: { target?: unknown }) => {
      const target = e.target;
      if (target && target instanceof Rect) {
        const charId = rectCharMap.get(target);
        if (charId !== undefined) {
          onSelect(charId);
          return;
        }
      }
      onSelect(null);
    };

    canvas.on('mouse:down', handler);
    return () => { canvas.off('mouse:down', handler); };
  }, [onSelect]);

  // Highlight selected
  useEffect(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;

    rectsRef.current.forEach((rect, id) => {
      const isSelected = id === selectedId;
      rect.set({
        strokeWidth: isSelected ? 2.5 : 0.8,
        stroke: isSelected ? '#3b82f6' : (rect.stroke as string),
        fill: isSelected ? 'rgba(59, 130, 246, 0.15)' : (rect.fill as string),
      });
    });
    canvas.renderAll();
  }, [selectedId]);

  const zoomToChar = useCallback((char: Character) => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    const rect = rectsRef.current.get(char.id);
    if (!rect) return;

    const vpt = canvas.viewportTransform;
    if (!vpt) return;

    const zoom = 2.5;
    canvas.setZoom(zoom);
    const cx = (rect.left ?? 0) + (rect.width ?? 0) / 2;
    const cy = (rect.top ?? 0) + (rect.height ?? 0) / 2;
    vpt[4] = canvas.getWidth() / 2 - cx * zoom;
    vpt[5] = canvas.getHeight() / 2 - cy * zoom;
    canvas.setViewportTransform(vpt);
    canvas.renderAll();
  }, []);

  const resetZoom = useCallback(() => {
    const canvas = fabricRef.current;
    if (!canvas) return;
    canvas.setZoom(1);
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);
    canvas.renderAll();
  }, []);

  return { zoomToChar, resetZoom, canvas: fabricRef };
}
