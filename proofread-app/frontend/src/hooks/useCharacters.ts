import { useState, useCallback } from 'react';
import type { Character } from '../utils/api';
import {
  listCharacters,
  updateCharacter,
  deleteCharacter as apiDeleteChar,
  restoreCharacter as apiRestore,
  mergeCharacters as apiMerge,
  splitCharacter as apiSplit,
  confirmAllAboveThreshold,
  insertCharacter as apiInsert,
  reEvaluateCharacters as apiReEvaluate,
} from '../utils/api';

export function useCharacters(pageId: number | null) {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const selected = characters.find(c => c.id === selectedId) ?? null;

  const load = useCallback(async () => {
    if (!pageId) return;
    setLoading(true);
    try {
      const chars = await listCharacters(pageId);
      setCharacters(chars);
    } finally {
      setLoading(false);
    }
  }, [pageId]);

  const confirmChar = useCallback(async (id: number) => {
    const updated = await updateCharacter(id, { is_confirmed: true });
    setCharacters(prev => prev.map(c => c.id === id ? updated : c));
  }, []);

  const correctChar = useCallback(async (id: number, text: string) => {
    const updated = await updateCharacter(id, { corrected_text: text });
    setCharacters(prev => prev.map(c => c.id === id ? updated : c));
  }, []);

  const deleteChar = useCallback(async (id: number): Promise<Character | null> => {
    const deleted = characters.find(c => c.id === id) ?? null;
    await apiDeleteChar(id);
    setCharacters(prev => prev.filter(c => c.id !== id));
    if (selectedId === id) setSelectedId(null);
    return deleted;
  }, [selectedId, characters]);

  const restoreChar = useCallback(async (id: number) => {
    const restored = await apiRestore(id);
    setCharacters(prev =>
      [...prev, restored].sort(
        (a, b) => a.column_index - b.column_index || a.char_index - b.char_index
      )
    );
  }, []);

  const merge = useCallback(async (ids: number[], text?: string) => {
    const merged = await apiMerge({ character_ids: ids, merged_text: text });
    setCharacters(prev => {
      const remaining = prev.filter(c => !ids.includes(c.id) || c.id === merged.id);
      return remaining.map(c => c.id === merged.id ? merged : c);
    });
  }, []);

  const split = useCallback(async (id: number, position: number) => {
    const result = await apiSplit(id, position);
    setCharacters(prev => {
      const filtered = prev.filter(c => c.id !== id);
      return [...filtered, ...result].sort(
        (a, b) => a.column_index - b.column_index || a.char_index - b.char_index
      );
    });
  }, []);

  const confirmAll = useCallback(async () => {
    if (!pageId) return 0;
    const result = await confirmAllAboveThreshold(pageId);
    await load();
    return result.confirmed;
  }, [pageId, load]);

  const moveBbox = useCallback(async (id: number, bbox: {
    bbox_x?: number; bbox_y?: number; bbox_w?: number; bbox_h?: number;
  }) => {
    const updated = await updateCharacter(id, bbox);
    setCharacters(prev => prev.map(c => c.id === id ? updated : c));
  }, []);

  const insertChar = useCallback(async (columnIndex: number, afterCharId?: number, beforeFirst?: boolean) => {
    if (!pageId) return;
    const newChar = await apiInsert(pageId, {
      column_index: columnIndex,
      after_char_id: afterCharId,
      before_first: beforeFirst,
    });
    setCharacters(prev =>
      [...prev, newChar].sort(
        (a, b) => a.column_index - b.column_index || a.char_index - b.char_index
      )
    );
    setSelectedId(newChar.id);
  }, [pageId]);

  const reEvaluate = useCallback(async () => {
    if (!pageId) return 0;
    const result = await apiReEvaluate(pageId);
    await load();
    return result.updated;
  }, [pageId, load]);

  return {
    characters, selected, selectedId, loading,
    setSelectedId, load, confirmChar, correctChar,
    deleteChar, restoreChar, merge, split, confirmAll, moveBbox,
    insertChar, reEvaluate,
  };
}
