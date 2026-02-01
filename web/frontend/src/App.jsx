import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import ComparePage from './pages/ComparePage';

export default function App() {
  return (
    <div className="app">
      <Header />
      <main className="main">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/result" element={<ResultPage />} />
          <Route path="/compare" element={<ComparePage />} />
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
