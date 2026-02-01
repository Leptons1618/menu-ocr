import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import JsonModal from '../components/JsonModal';
import { ArrowLeft, Table, Download, Image, ImageOff } from 'lucide-react';

export default function ResultPage() {
  const { state } = useApp();
  const [showModal, setShowModal] = useState(false);
  const [activeImage, setActiveImage] = useState('annotated');
  
  const result = state.currentResult;
  
  if (!result) {
    return (
      <div className="container">
        <div style={{ textAlign: 'center', padding: 64 }}>
          <ImageOff size={48} style={{ color: '#a3a3a3', marginBottom: 16 }} />
          <h2 style={{ marginBottom: 8 }}>No Result Selected</h2>
          <p style={{ color: '#737373', marginBottom: 24 }}>
            Upload an image first to see extraction results
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
  
  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 32 }}>
        <Link to="/" className="btn btn-ghost">
          <ArrowLeft size={16} />
          Back
        </Link>
        <div>
          <h1 className="page-title" style={{ marginBottom: 0 }}>{result.filename}</h1>
          <p style={{ color: '#737373', fontSize: 14 }}>
            Processed in {result.processingTime}ms
          </p>
        </div>
      </div>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{countSections(result.menu)}</div>
          <div className="stat-label">Sections</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{countItems(result.menu)}</div>
          <div className="stat-label">Items Extracted</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{result.processingTime}ms</div>
          <div className="stat-label">Processing Time</div>
        </div>
      </div>
      
      <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
          <Table size={16} />
          View Data
        </button>
        <button 
          className="btn btn-secondary"
          onClick={() => {
            const blob = new Blob([JSON.stringify(result.menu, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${result.filename.replace(/\.[^.]+$/, '')}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
        >
          <Download size={16} />
          Download JSON
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
          <div className="image-container">
            <img 
              src={activeImage === 'annotated' ? result.images.annotated : result.images.original}
              alt={activeImage === 'annotated' ? 'Annotated menu' : 'Original menu'}
            />
          </div>
        </div>
      </div>
      
      {showModal && (
        <JsonModal data={result.menu} onClose={() => setShowModal(false)} />
      )}
    </div>
  );
}
