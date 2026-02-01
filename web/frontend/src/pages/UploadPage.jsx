import React, { useCallback, useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../context/AppContext';
import { extractMenu, getModels } from '../utils/api';
import UploadZone from '../components/UploadZone';
import { Clock, FileText, Trash2, ChevronDown, Cpu, CheckCircle, AlertCircle, Loader, X } from 'lucide-react';

export default function UploadPage() {
  const navigate = useNavigate();
  const { state, dispatch } = useApp();
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('rule_based');
  const processingRef = useRef(false);
  
  useEffect(() => {
    getModels()
      .then(data => {
        setModels(data.models || []);
      })
      .catch(err => console.warn('Could not fetch models:', err));
  }, []);
  
  // Process queue
  useEffect(() => {
    const processNext = async () => {
      if (processingRef.current) return;
      
      const nextItem = state.processingQueue.find(item => item.status === 'queued');
      if (!nextItem) return;
      
      processingRef.current = true;
      
      dispatch({ type: 'UPDATE_QUEUE_ITEM', payload: { id: nextItem.id, status: 'processing' } });
      
      try {
        const result = await extractMenu(nextItem.file, selectedModel);
        
        const upload = {
          id: nextItem.id,
          filename: nextItem.filename,
          timestamp: new Date().toISOString(),
          menu: result.menu,
          images: result.images,
          processingTime: result.processingTime,
          modelUsed: result.modelUsed,
        };
        
        dispatch({ type: 'ADD_RESULT', payload: upload });
        dispatch({ type: 'UPDATE_QUEUE_ITEM', payload: { id: nextItem.id, status: 'completed' } });
      } catch (error) {
        dispatch({ type: 'UPDATE_QUEUE_ITEM', payload: { id: nextItem.id, status: 'error', error: error.message } });
      }
      
      processingRef.current = false;
    };
    
    processNext();
  }, [state.processingQueue, dispatch, selectedModel]);
  
  const handleUpload = useCallback((files) => {
    const queueItems = files.map((file, index) => ({
      id: `${Date.now()}-${index}`,
      filename: file.name,
      file: file,
      status: 'queued',
    }));
    
    dispatch({ type: 'ADD_TO_QUEUE', payload: queueItems });
  }, [dispatch]);
  
  const handleRemove = useCallback((id) => {
    dispatch({ type: 'REMOVE_RESULT', payload: id });
  }, [dispatch]);
  
  const handleRemoveFromQueue = useCallback((id) => {
    dispatch({ type: 'REMOVE_FROM_QUEUE', payload: id });
  }, [dispatch]);
  
  const isProcessing = state.processingQueue.some(item => item.status === 'processing');
  const hasQueue = state.processingQueue.length > 0;
  const completedCount = state.processingQueue.filter(item => item.status === 'completed').length;
  const totalCount = state.processingQueue.length;
  
  return (
    <div className="container">
      <h1 className="page-title">Upload Menu Image</h1>
      <p className="page-subtitle">
        Extract structured data from restaurant menu images
      </p>
      
      {/* Model Selection */}
      <div style={{ marginBottom: 24 }}>
        <label style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 8, 
          fontSize: 14, 
          fontWeight: 500,
          marginBottom: 8 
        }}>
          <Cpu size={16} />
          Select Model
        </label>
        <div style={{ position: 'relative', maxWidth: 300 }}>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isProcessing}
            style={{
              width: '100%',
              padding: '10px 36px 10px 12px',
              fontSize: 14,
              border: '1px solid #d4d4d4',
              borderRadius: 6,
              background: '#fff',
              appearance: 'none',
              cursor: isProcessing ? 'not-allowed' : 'pointer',
              opacity: isProcessing ? 0.6 : 1,
            }}
          >
            {models.length > 0 ? (
              models.map(m => (
                <option key={m.name} value={m.name}>
                  {m.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  {m.accuracy ? ` (${(m.accuracy * 100).toFixed(0)}% acc)` : ''}
                </option>
              ))
            ) : (
              <option value="rule_based">Rule Based</option>
            )}
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
        <p style={{ fontSize: 12, color: '#737373', marginTop: 4 }}>
          Rule-based is recommended for best menu extraction
        </p>
      </div>
      
      {state.error && (
        <div style={{ 
          background: '#fef2f2', 
          border: '1px solid #fecaca', 
          borderRadius: 8, 
          padding: 16, 
          marginBottom: 24,
          color: '#991b1b'
        }}>
          {state.error}
        </div>
      )}
      
      <UploadZone onUpload={handleUpload} disabled={false} />
      
      {/* Processing Queue */}
      {hasQueue && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {isProcessing && <Loader size={16} className="spinner-inline" />}
              Processing Queue
              <span style={{ 
                fontSize: 12, 
                fontWeight: 400, 
                color: '#737373',
                marginLeft: 8 
              }}>
                {completedCount}/{totalCount} completed
              </span>
            </span>
            <button 
              className="btn btn-ghost"
              onClick={() => dispatch({ type: 'CLEAR_QUEUE' })}
              style={{ fontSize: 12, padding: '4px 8px' }}
            >
              Clear All
            </button>
          </div>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            {state.processingQueue.map((item) => (
              <div 
                key={item.id}
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  padding: '12px 20px',
                  borderBottom: '1px solid #e5e5e5',
                  gap: 12
                }}
              >
                {item.status === 'queued' && (
                  <Clock size={16} style={{ color: '#a3a3a3' }} />
                )}
                {item.status === 'processing' && (
                  <Loader size={16} className="spinner-inline" style={{ color: '#171717' }} />
                )}
                {item.status === 'completed' && (
                  <CheckCircle size={16} style={{ color: '#22c55e' }} />
                )}
                {item.status === 'error' && (
                  <AlertCircle size={16} style={{ color: '#ef4444' }} />
                )}
                
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ 
                    fontWeight: 500, 
                    fontSize: 14,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {item.filename}
                  </div>
                  <div style={{ fontSize: 12, color: '#737373' }}>
                    {item.status === 'queued' && 'Waiting in queue...'}
                    {item.status === 'processing' && 'Processing...'}
                    {item.status === 'completed' && 'Completed'}
                    {item.status === 'error' && (
                      <span style={{ color: '#ef4444' }}>{item.error || 'Processing failed'}</span>
                    )}
                  </div>
                </div>
                
                {item.status === 'completed' && (
                  <button 
                    className="btn btn-secondary"
                    style={{ fontSize: 12, padding: '4px 12px' }}
                    onClick={() => {
                      const result = state.uploads.find(u => u.id === item.id);
                      if (result) {
                        dispatch({ type: 'SET_CURRENT', payload: result });
                        navigate('/result');
                      }
                    }}
                  >
                    View
                  </button>
                )}
                
                {(item.status === 'queued' || item.status === 'error') && (
                  <button 
                    className="btn btn-ghost"
                    onClick={() => handleRemoveFromQueue(item.id)}
                    style={{ padding: 4 }}
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {state.uploads.length > 0 && (
        <div style={{ marginTop: 32 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 16 }}>
            Recent Uploads
          </h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {state.uploads.map((upload) => (
              <div 
                key={upload.id}
                className="card"
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  padding: 16,
                  cursor: 'pointer'
                }}
                onClick={() => {
                  dispatch({ type: 'SET_CURRENT', payload: upload });
                  navigate('/result');
                }}
              >
                <FileText size={20} style={{ marginRight: 12, color: '#737373' }} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 500 }}>{upload.filename}</div>
                  <div style={{ fontSize: 13, color: '#737373', display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
                    <Clock size={12} />
                    {new Date(upload.timestamp).toLocaleString()}
                    <span style={{ color: '#a3a3a3' }}>|</span>
                    {upload.processingTime}ms
                    {upload.modelUsed && (
                      <>
                        <span style={{ color: '#a3a3a3' }}>|</span>
                        <Cpu size={12} />
                        {upload.modelUsed}
                      </>
                    )}
                  </div>
                </div>
                <button 
                  className="btn btn-ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemove(upload.id);
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
