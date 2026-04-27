const API_BASE = '/api';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// Projects
export interface Project {
  id: number;
  volume_code: string;
  book_name: string;
  total_pages: number;
  completed_pages: number;
  created_at: string;
}

export const listProjects = () => request<Project[]>('/projects');
export const createProject = (data: { volume_code: string; book_name: string }) =>
  request<Project>('/projects', { method: 'POST', body: JSON.stringify(data) });
export const getProject = (id: number) => request<Project>(`/projects/${id}`);
export const deleteProject = (id: number) =>
  request<void>(`/projects/${id}`, { method: 'DELETE' });

// Pages
export interface Page {
  id: number;
  project_id: number;
  page_number: number;
  image_path: string;
  width: number | null;
  height: number | null;
  ocr_status: string;
  proofread_status: string;
  ocr_engine: string | null;
  total_chars: number;
  confirmed_chars: number;
  low_confidence_chars: number;
}

export const listPages = (projectId: number) =>
  request<Page[]>(`/projects/${projectId}/pages`);
export const getPage = (id: number) => request<Page>(`/pages/${id}`);
export const getPageImageUrl = (id: number) => `${API_BASE}/pages/${id}/image`;

export const uploadPages = async (projectId: number, files: FileList): Promise<Page[]> => {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }
  const res = await fetch(`${API_BASE}/projects/${projectId}/pages/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
};

// OCR
export const triggerOCR = (pageId: number, engines: string[] = ['paddle']) =>
  request<{ status: string }>(`/pages/${pageId}/ocr`, {
    method: 'POST',
    body: JSON.stringify({ engines }),
  });
export const getOCRStatus = (pageId: number) =>
  request<{ page_id: number; ocr_status: string; total_chars: number }>(
    `/pages/${pageId}/ocr/status`
  );

// Characters
export interface Character {
  id: number;
  page_id: number;
  bbox_x: number;
  bbox_y: number;
  bbox_w: number;
  bbox_h: number;
  column_index: number;
  char_index: number;
  ocr_text: string | null;
  ocr_confidence: number;
  ocr_engine: string | null;
  alternatives: { text: string; confidence: number; engine: string }[];
  is_confirmed: boolean;
  corrected_text: string | null;
  is_deleted: boolean;
  display_text: string;
}

export const listCharacters = (pageId: number) =>
  request<Character[]>(`/pages/${pageId}/characters`);
export const createCharacter = (pageId: number, data: {
  bbox_x: number; bbox_y: number; bbox_w: number; bbox_h: number;
  column_index: number; char_index: number; ocr_text?: string;
}) =>
  request<Character>(`/pages/${pageId}/characters`, {
    method: 'POST', body: JSON.stringify(data),
  });
export const updateCharacter = (id: number, data: {
  corrected_text?: string; is_confirmed?: boolean;
  bbox_x?: number; bbox_y?: number; bbox_w?: number; bbox_h?: number;
}) =>
  request<Character>(`/characters/${id}`, {
    method: 'PATCH', body: JSON.stringify(data),
  });
export const deleteCharacter = (id: number) =>
  request<void>(`/characters/${id}`, { method: 'DELETE' });
export const mergeCharacters = (data: { character_ids: number[]; merged_text?: string }) =>
  request<Character>('/characters/merge', {
    method: 'POST', body: JSON.stringify(data),
  });
export const splitCharacter = (id: number, split_position: number) =>
  request<Character[]>(`/characters/${id}/split`, {
    method: 'POST', body: JSON.stringify({ split_position }),
  });
export const confirmAllAboveThreshold = (pageId: number) =>
  request<{ confirmed: number }>(`/pages/${pageId}/characters/confirm-all`, {
    method: 'POST',
  });
export const reorderCharacters = (pageId: number) =>
  request<{ reordered: number }>(`/pages/${pageId}/characters/reorder`, {
    method: 'POST',
  });

// Batch OCR
export interface BatchOCRStatus {
  total: number;
  pending: number;
  processing: number;
  done: number;
  failed: number;
}

export const triggerBatchOCR = (pageIds?: number[], engines: string[] = ['kraken'], maxPages = 50) =>
  request<{ status: string; queued: number[]; count: number }>('/ocr/batch', {
    method: 'POST',
    body: JSON.stringify({ page_ids: pageIds ?? null, engines, max_pages: maxPages }),
  });
export const getBatchOCRStatus = () =>
  request<BatchOCRStatus>('/ocr/batch/status');

// Templates
export interface TemplateMatch {
  text: string;
  similarity: number;
  template_id: number;
  hamming_distance: number;
}

export interface CandidateResult {
  template_matches: TemplateMatch[];
  confusables: string[];
  variants: string[];
}

export const getCandidates = (charId: number, topK = 5) =>
  request<CandidateResult>(`/characters/${charId}/candidates?top_k=${topK}`);
export const matchTemplatesByCharId = (charId: number, topK = 5) =>
  request<TemplateMatch[]>(`/characters/${charId}/template-matches?top_k=${topK}`);
export const getTemplateImageUrl = (templateId: number) =>
  `${API_BASE}/templates/${templateId}/image`;
export const getTemplateStats = () =>
  request<{ total_templates: number; unique_characters: number }>('/templates/stats');

// Export
export const getExportMandoku = (pageId: number) =>
  fetch(`${API_BASE}/pages/${pageId}/export/mandoku`).then(r => r.text());
export const exportToRepo = (projectId: number) =>
  request<{ exported_pages: number; output_dir: string }>(
    `/projects/${projectId}/export/to-repo`, { method: 'POST' }
  );
