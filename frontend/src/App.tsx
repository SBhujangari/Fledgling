import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import HomePage from './pages/HomePage';
import PlaygroundPage from './pages/PlaygroundPage';
import AgentDetailPage from './pages/AgentDetailPage';
import TracesPage from './pages/TracesPage';
import OperationsConsolePage from './pages/OperationsConsolePage';
import MetricsPage from './pages/MetricsPage';
import SLMDashboardPage from './pages/SLMDashboardPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<HomePage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route path="/agent/:agentId" element={<AgentDetailPage />} />
        <Route path="/traces" element={<TracesPage />} />
        <Route path="/ops" element={<OperationsConsolePage />} />
        <Route path="/metrics" element={<MetricsPage />} />
        <Route path="/slm-dashboard" element={<SLMDashboardPage />} />
      </Routes>
    </BrowserRouter>
  );
}
