import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ProjectList } from './pages/ProjectList';
import { ProofreadPage } from './pages/ProofreadPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProjectList />} />
        <Route path="/project/:projectId" element={<ProofreadPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
