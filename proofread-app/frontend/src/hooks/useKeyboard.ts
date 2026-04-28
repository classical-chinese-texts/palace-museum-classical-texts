import { useEffect } from 'react';
import type { Character } from '../utils/api';
import { sortByReadingOrder, findNextLowConfidence } from '../utils/readingOrder';

interface KeyboardActions {
  characters: Character[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onConfirm: (id: number) => void;
  onDelete: (id: number) => void;
  onCorrect: (id: number, text: string) => void;
  onZoomToChar: (char: Character) => void;
}

export function useKeyboard(actions: KeyboardActions) {
  useEffect(() => {
    const sorted = sortByReadingOrder(actions.characters);
    const currentIdx = actions.selectedId
      ? sorted.findIndex(c => c.id === actions.selectedId)
      : -1;

    const handler = (e: KeyboardEvent) => {
      // Skip if typing in an input
      if ((e.target as HTMLElement).tagName === 'INPUT' ||
          (e.target as HTMLElement).tagName === 'TEXTAREA') {
        return;
      }

      switch (e.key) {
        case 'Tab': {
          e.preventDefault();
          const next = findNextLowConfidence(
            actions.characters, actions.selectedId
          );
          if (next) {
            actions.onSelect(next.id);
            actions.onZoomToChar(next);
          }
          break;
        }

        case 'ArrowDown': {
          e.preventDefault();
          if (currentIdx < sorted.length - 1) {
            const next = sorted[currentIdx + 1];
            if (next.column_index === sorted[currentIdx]?.column_index) {
              actions.onSelect(next.id);
            }
          }
          break;
        }

        case 'ArrowUp': {
          e.preventDefault();
          if (currentIdx > 0) {
            const prev = sorted[currentIdx - 1];
            if (prev.column_index === sorted[currentIdx]?.column_index) {
              actions.onSelect(prev.id);
            }
          }
          break;
        }

        case 'ArrowRight': {
          e.preventDefault();
          // Move to previous column (right in vertical text = column_index - 1)
          if (currentIdx >= 0) {
            const curCol = sorted[currentIdx].column_index;
            const prev = sorted.find(c => c.column_index === curCol - 1);
            if (prev) actions.onSelect(prev.id);
          }
          break;
        }

        case 'ArrowLeft': {
          e.preventDefault();
          // Move to next column
          if (currentIdx >= 0) {
            const curCol = sorted[currentIdx].column_index;
            const next = sorted.find(c => c.column_index === curCol + 1);
            if (next) actions.onSelect(next.id);
          }
          break;
        }

        case 'Enter': {
          e.preventDefault();
          if (actions.selectedId) {
            actions.onConfirm(actions.selectedId);
            // Auto-advance
            if (currentIdx < sorted.length - 1) {
              actions.onSelect(sorted[currentIdx + 1].id);
            }
          }
          break;
        }

        case ' ': {
          e.preventDefault();
          // Skip — advance to next
          if (currentIdx < sorted.length - 1) {
            actions.onSelect(sorted[currentIdx + 1].id);
          }
          break;
        }

        case 'Delete':
        case 'Backspace': {
          if (actions.selectedId) {
            e.preventDefault();
            actions.onDelete(actions.selectedId);
          }
          break;
        }

        case 'Escape': {
          actions.onSelect(null);
          break;
        }

        default: {
          // Number keys 1-9: select alternative
          if (/^[1-9]$/.test(e.key) && actions.selectedId) {
            const char = actions.characters.find(c => c.id === actions.selectedId);
            if (char?.alternatives) {
              const idx = parseInt(e.key) - 1;
              if (idx < char.alternatives.length) {
                actions.onCorrect(actions.selectedId, char.alternatives[idx].text);
              }
            }
          }
          break;
        }
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [actions.characters, actions.selectedId]);
}
