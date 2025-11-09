import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import HomePage from './pages/HomePage';
import PlaygroundPage from './pages/PlaygroundPage';
import AgentDetailPage from './pages/AgentDetailPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<HomePage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route path="/agent/:agentId" element={<AgentDetailPage />} />
      </Routes>
    </BrowserRouter>
  );
}
