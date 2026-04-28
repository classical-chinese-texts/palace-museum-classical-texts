import type { Character } from './api';

/**
 * Sort characters in traditional Chinese reading order:
 * Right to left (columns), top to bottom (within column).
 */
export function sortByReadingOrder(chars: Character[]): Character[] {
  return [...chars].sort((a, b) => {
    if (a.column_index !== b.column_index) return a.column_index - b.column_index;
    return a.char_index - b.char_index;
  });
}

/**
 * Group characters by column for Mandoku preview.
 */
export function groupByColumn(chars: Character[]): Map<number, Character[]> {
  const sorted = sortByReadingOrder(chars);
  const groups = new Map<number, Character[]>();
  for (const c of sorted) {
    const col = groups.get(c.column_index) ?? [];
    col.push(c);
    groups.set(c.column_index, col);
  }
  return groups;
}

/**
 * Find next low-confidence character after current index.
 */
export function findNextLowConfidence(
  chars: Character[],
  currentId: number | null,
  threshold: number = 0.7,
): Character | null {
  const sorted = sortByReadingOrder(chars);
  const currentIdx = currentId ? sorted.findIndex(c => c.id === currentId) : -1;

  // Search from current position forward
  for (let i = currentIdx + 1; i < sorted.length; i++) {
    if (!sorted[i].is_confirmed && sorted[i].ocr_confidence < threshold) {
      return sorted[i];
    }
  }
  // Wrap around
  for (let i = 0; i <= currentIdx; i++) {
    if (!sorted[i].is_confirmed && sorted[i].ocr_confidence < threshold) {
      return sorted[i];
    }
  }
  return null;
}
