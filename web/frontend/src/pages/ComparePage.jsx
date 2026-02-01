import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import JsonModal from '../components/JsonModal';
import { ArrowLeft, Table, ImageOff, ChevronDown } from 'lucide-react';

export default function ComparePage() {
  const { state } = useApp();
  const [leftId, setLeftId] = useState(state.uploads[0]?.id || '');
  const [rightId, setRightId] = useState(state.uploads[1]?.id || '');
  const [showModal, setShowModal] = useState(null);
  
  const leftResult = state.uploads.find(u => u.id === leftId);
  const rightResult = state.uploads.find(u => u.id === rightId);
  
  if (state.uploads.length < 2) {
    return (
      <div className="container">
        <div style={{ textAlign: 'center', padding: 64 }}>
          <ImageOff size={48} style={{ color: '#a3a3a3', marginBottom: 16 }} />
          <h2 style={{ marginBottom: 8 }}>Need More Images</h2>
          <p style={{ color: '#737373', marginBottom: 24 }}>
            Upload at least 2 images to compare results
          </p>
          <Link to="/" className="btn btn-primary">
            Go to Upload
          </Link>
        </div>
      </div>
    );
  }
  
  const renderSelect = (value, onChange, excludeId) => (
    <div style={{ position: 'relative' }}>
      <select 
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{
          width: '100%',
          padding: '10px 36px 10px 12px',
          border: '1px solid #d4d4d4',
          borderRadius: 8,
          fontSize: 14,
          appearance: 'none',
          background: '#fff',
          cursor: 'pointer',
        }}
      >
        <option value="">Select image...</option>
        {state.uploads
          .filter(u => u.id !== excludeId)
          .map(u => (
            <option key={u.id} value={u.id}>{u.filename}</option>
          ))
        }
      </select>
      <ChevronDown 
        size={16} 
        style={{ 
          position: 'absolute', 
          right: 12, 
          top: '50%', 
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
          color: '#737373'
        }} 
      />
    </div>
  );
  
  const renderResult = (result) => {
    if (!result) {
      return (
        <div style={{ 
          background: '#fafafa', 
          border: '2px dashed #d4d4d4', 
          borderRadius: 8, 
          padding: 48, 
          textAlign: 'center',
          color: '#737373'
        }}>
          Select an image to compare
        </div>
      );
    }
    
    return (
      <div className="card">
        <div className="card-header">
          <span className="card-title">{result.filename}</span>
          <button 
            className="btn btn-ghost" 
            style={{ padding: '4px 8px' }}
            onClick={() => setShowModal(result)}
          >
            <Table size={14} />
          </button>
        </div>
        <div className="card-body" style={{ padding: 0 }}>
          <div className="image-container">
            <img src={result.images.annotated} alt="Annotated" />
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 32 }}>
        <Link to="/" className="btn btn-ghost">
          <ArrowLeft size={16} />
          Back
        </Link>
        <div>
          <h1 className="page-title" style={{ marginBottom: 0 }}>Compare Results</h1>
          <p style={{ color: '#737373', fontSize: 14 }}>
            Compare extraction results side by side
          </p>
        </div>
      </div>
      
      <div className="compare-grid" style={{ marginBottom: 24 }}>
        <div>
          {renderSelect(leftId, setLeftId, rightId)}
        </div>
        <div>
          {renderSelect(rightId, setRightId, leftId)}
        </div>
      </div>
      
      <div className="compare-grid">
        {renderResult(leftResult)}
        {renderResult(rightResult)}
      </div>
      
      {showModal && (
        <JsonModal data={showModal.menu} onClose={() => setShowModal(null)} />
      )}
    </div>
  );
}
