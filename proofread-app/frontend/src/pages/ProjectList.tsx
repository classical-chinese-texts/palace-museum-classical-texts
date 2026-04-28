import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Project } from '../utils/api';
import { listProjects, createProject, deleteProject } from '../utils/api';

export function ProjectList() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [volumeCode, setVolumeCode] = useState('');
  const [bookName, setBookName] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const loadProjects = async () => {
    setLoading(true);
    try {
      const list = await listProjects();
      setProjects(list);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadProjects(); }, []);

  const handleCreate = async () => {
    if (!volumeCode || !bookName) return;
    await createProject({ volume_code: volumeCode, book_name: bookName });
    setVolumeCode('');
    setBookName('');
    loadProjects();
  };

  const handleDelete = async (id: number) => {
    if (!confirm('確定刪除此專案？')) return;
    await deleteProject(id);
    loadProjects();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="max-w-4xl mx-auto p-8">
        <h1 className="text-3xl font-bold mb-2">OCR 校對平台</h1>
        <p className="text-gray-400 mb-8">故宮珍本叢刊 — 字級 OCR 校對工具</p>

        {/* Create project */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8 border border-gray-700">
          <h2 className="text-lg font-medium mb-4">建立新專案</h2>
          <div className="flex gap-3">
            <input
              type="text"
              value={volumeCode}
              onChange={e => setVolumeCode(e.target.value)}
              placeholder="冊別代碼 (如 GGZBCK421)"
              className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
            />
            <input
              type="text"
              value={bookName}
              onChange={e => setBookName(e.target.value)}
              placeholder="書名 (如 六壬眎斯)"
              className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
            />
            <button
              onClick={handleCreate}
              disabled={!volumeCode || !bookName}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                         disabled:text-gray-500 rounded font-medium"
            >
              建立
            </button>
          </div>
        </div>

        {/* Project list */}
        <div className="space-y-3">
          {loading && <p className="text-gray-500">載入中...</p>}
          {!loading && projects.length === 0 && (
            <p className="text-gray-500 text-center py-8">尚無專案，請建立第一個</p>
          )}
          {projects.map(p => (
            <div
              key={p.id}
              className="bg-gray-800 rounded-lg p-4 border border-gray-700
                         hover:border-gray-600 flex items-center justify-between cursor-pointer"
              onClick={() => navigate(`/project/${p.id}`)}
            >
              <div>
                <div className="font-medium text-lg">{p.book_name}</div>
                <div className="text-sm text-gray-400">
                  {p.volume_code} | {p.total_pages} 頁 | 已完成 {p.completed_pages} 頁
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={(e) => { e.stopPropagation(); navigate(`/project/${p.id}`); }}
                  className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm"
                >
                  開始校對
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); handleDelete(p.id); }}
                  className="px-3 py-2 bg-red-600/20 hover:bg-red-600 text-red-400
                             hover:text-white rounded text-sm"
                >
                  刪除
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
