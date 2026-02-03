import React, { useState, useCallback, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { 
  ArrowLeft, ArrowRight, Table, Download, Image, ImageOff, 
  Maximize2, X, ChevronLeft, ChevronRight, List, Minimize2
} from 'lucide-react';
import JsonModal from '../components/JsonModal';

export default function ResultsPage() {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();
  const [showModal, setShowModal] = useState(false);
  const [activeImage, setActiveImage] = useState('annotated');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);
  
  const results = state.uploads;
  const currentResult = results[currentIndex];
  
  // Navigate to specific result
  useEffect(() => {
    if (state.currentResult && results.length > 0) {
      const idx = results.findIndex(r => r.id === state.currentResult.id);
      if (idx >= 0) {
        setCurrentIndex(idx);
      }
    }
  }, [state.currentResult, results]);
  
  const goNext = useCallback(() => {
    if (currentIndex < results.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  }, [currentIndex, results.length]);
  
  const goPrev = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  }, [currentIndex]);
  
  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowRight') goNext();
      if (e.key === 'ArrowLeft') goPrev();
      if (e.key === 'Escape' && fullscreen) setFullscreen(false);
      if (e.key === 'f' && !showModal) setFullscreen(f => !f);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goNext, goPrev, fullscreen, showModal]);
  
  if (!results.length || !currentResult) {
    return (
      <div className="container">
        <div style={{ textAlign: 'center', padding: 64 }}>
          <ImageOff size={48} style={{ color: '#a3a3a3', marginBottom: 16 }} />
          <h2 style={{ marginBottom: 8 }}>No Results Available</h2>
          <p style={{ color: '#737373', marginBottom: 24 }}>
            Upload and process menu images to see results
          </p>
          <Link to="/" className="btn btn-primary">
            Go to Upload
          </Link>
        </div>
      </div>
    );
  }
  
  const countItems = (menu) => {
    let count = 0;
    for (const section of menu?.menu || []) {
      for (const group of section.groups || []) {
        count += (group.items || []).length;
      }
    }
    return count;
  };
  
  const countSections = (menu) => (menu?.menu || []).length;
  
  // Fullscreen view
  if (fullscreen) {
    return (
      <div className="fullscreen-view">
        <div className="fullscreen-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button className="btn btn-ghost" onClick={() => setFullscreen(false)} style={{ color: '#fff' }}>
              <X size={20} />
            </button>
            <span style={{ fontWeight: 500, color: '#fff' }}>{currentResult.filename}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 14, color: 'rgba(255,255,255,0.7)' }}>
              {currentIndex + 1} / {results.length}
            </span>
            <button 
              className={`btn ${activeImage === 'annotated' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveImage('annotated')}
              style={{ padding: '6px 12px', fontSize: 12 }}
            >
              Annotated
            </button>
            <button 
              className={`btn ${activeImage === 'original' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveImage('original')}
              style={{ padding: '6px 12px', fontSize: 12 }}
            >
              Original
            </button>
          </div>
        </div>
        
        <div className="fullscreen-content">
          <button 
            className="fullscreen-nav prev"
            onClick={goPrev}
            disabled={currentIndex === 0}
          >
            <ChevronLeft size={32} />
          </button>
          
          <div className="fullscreen-image">
            <img 
              src={activeImage === 'annotated' ? currentResult.images.annotated : currentResult.images.original}
              alt={currentResult.filename}
            />
          </div>
          
          <button 
            className="fullscreen-nav next"
            onClick={goNext}
            disabled={currentIndex === results.length - 1}
          >
            <ChevronRight size={32} />
          </button>
        </div>
        
        <div className="fullscreen-footer">
          <div className="thumbnail-strip">
            {results.map((result, idx) => (
              <div 
                key={result.id}
                className={`thumbnail ${idx === currentIndex ? 'active' : ''}`}
                onClick={() => setCurrentIndex(idx)}
              >
                <img src={result.images.annotated} alt={result.filename} />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <Link to="/" className="btn btn-ghost">
            <ArrowLeft size={16} />
            Back
          </Link>
          <div>
            <h1 className="page-title" style={{ marginBottom: 0 }}>
              {results.length > 1 ? 'Results' : currentResult.filename}
            </h1>
            <p style={{ color: '#737373', fontSize: 14 }}>
              {results.length > 1 
                ? `${results.length} images processed`
                : `Processed in ${currentResult.processingTime}ms`
              }
            </p>
          </div>
        </div>
        
        {results.length > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button className="btn btn-secondary" onClick={goPrev} disabled={currentIndex === 0}>
              <ChevronLeft size={16} />
            </button>
            <span style={{ fontSize: 14, minWidth: 60, textAlign: 'center' }}>
              {currentIndex + 1} / {results.length}
            </span>
            <button className="btn btn-secondary" onClick={goNext} disabled={currentIndex === results.length - 1}>
              <ChevronRight size={16} />
            </button>
          </div>
        )}
      </div>
      
      {/* Result card */}
      <div style={{ marginBottom: 24, padding: '12px 16px', background: 'var(--color-gray-50)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontWeight: 500 }}>{currentResult.filename}</div>
          <div style={{ fontSize: 13, color: 'var(--color-gray-500)' }}>
            {currentResult.processingTime}ms • {currentResult.modelUsed}
          </div>
        </div>
        {results.length > 1 && (
          <button 
            className="btn btn-ghost" 
            onClick={() => setFullscreen(true)}
            style={{ padding: 8 }}
          >
            <Maximize2 size={16} />
          </button>
        )}
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{countSections(currentResult.menu)}</div>
          <div className="stat-label">Sections</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{countItems(currentResult.menu)}</div>
          <div className="stat-label">Items</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{currentResult.processingTime}ms</div>
          <div className="stat-label">Processing Time</div>
        </div>
        {results.length > 1 && (
          <div className="stat-card">
            <div className="stat-value">{results.length}</div>
            <div className="stat-label">Total Images</div>
          </div>
        )}
      </div>
      
      <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
          <Table size={16} />
          View Data
        </button>
        <button 
          className="btn btn-secondary"
          onClick={() => {
            const blob = new Blob([JSON.stringify(currentResult.menu, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${currentResult.filename.replace(/\.[^.]+$/, '')}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
        >
          <Download size={16} />
          Download JSON
        </button>
        <button className="btn btn-secondary" onClick={() => setFullscreen(true)}>
          <Maximize2 size={16} />
          Fullscreen
        </button>
      </div>
      
      <div className="card">
        <div className="card-header">
          <span className="card-title">Extracted Image</span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button 
              className={`btn ${activeImage === 'annotated' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveImage('annotated')}
              style={{ padding: '6px 12px', fontSize: 13 }}
            >
              Annotated
            </button>
            <button 
              className={`btn ${activeImage === 'original' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveImage('original')}
              style={{ padding: '6px 12px', fontSize: 13 }}
            >
              Original
            </button>
          </div>
        </div>
        <div className="card-body">
          <div 
            className="image-container" 
            style={{ cursor: 'pointer' }}
            onClick={() => setFullscreen(true)}
          >
            <img 
              src={activeImage === 'annotated' ? currentResult.images.annotated : currentResult.images.original}
              alt={activeImage === 'annotated' ? 'Annotated menu' : 'Original menu'}
            />
          </div>
        </div>
      </div>
      
      {/* Thumbnails for multiple results */}
      {results.length > 1 && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-header">
            <span className="card-title">All Results ({results.length})</span>
          </div>
          <div style={{ padding: 16, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {results.map((result, idx) => (
              <div 
                key={result.id}
                onClick={() => setCurrentIndex(idx)}
                style={{ 
                  cursor: 'pointer',
                  border: idx === currentIndex ? '2px solid var(--color-gray-900)' : '2px solid var(--color-gray-200)',
                  borderRadius: 8,
                  overflow: 'hidden',
                  width: 120,
                  transition: 'border-color 150ms ease'
                }}
              >
                <img 
                  src={result.images.annotated} 
                  alt={result.filename}
                  style={{ width: '100%', height: 90, objectFit: 'cover' }}
                />
                <div style={{ padding: '6px 8px', fontSize: 11, background: 'var(--color-gray-50)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {result.filename}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {showModal && (
        <JsonModal data={currentResult.menu} onClose={() => setShowModal(false)} />
      )}
    </div>
  );
}
