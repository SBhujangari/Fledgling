import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import HistoryPage from './pages/HistoryPage';
import TuningPage from './pages/TuningPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/testing" element={<HistoryPage />} />
        <Route path="/tuning" element={<TuningPage />} />
      </Routes>
    </BrowserRouter>
  );
}
