import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import ComparePage from './pages/ComparePage';
import ResearchPage from './pages/ResearchPage';

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="main">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/result" element={<ResultsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/research" element={<ResearchPage />} />
        </Routes>
      </main>
      <footer className="footer">
        <div className="container">
          Menu OCR System - EasyOCR + Rule-based Classification
        </div>
      </footer>
    </div>
  );
}
