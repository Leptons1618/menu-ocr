import React, { useState, useCallback } from 'react';
import { X, Image, Trash2, Play, FileText, AlertCircle } from 'lucide-react';

export default function UploadModal({ files, onClose, onProcess, onRemoveFile }) {
  const [selectedIndices, setSelectedIndices] = useState(new Set(files.map((_, i) => i)));
  
  const toggleFile = useCallback((index) => {
    setSelectedIndices(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }, []);
  
  const selectAll = useCallback(() => {
    setSelectedIndices(new Set(files.map((_, i) => i)));
  }, [files]);
  
  const deselectAll = useCallback(() => {
    setSelectedIndices(new Set());
  }, []);
  
  const selectedFiles = files.filter((_, i) => selectedIndices.has(i));
  
  const handleProcess = useCallback(() => {
    if (selectedFiles.length > 0) {
      onProcess(selectedFiles);
    }
  }, [selectedFiles, onProcess]);
  
  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 700 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">
            <FileText size={20} style={{ marginRight: 8 }} />
            {files.length} File{files.length !== 1 ? 's' : ''} Selected
          </h2>
          <button className="modal-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <div style={{ padding: '12px 24px', borderBottom: '1px solid var(--color-gray-200)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <button 
              className="btn btn-ghost" 
              style={{ padding: '4px 10px', fontSize: 12 }}
              onClick={selectAll}
            >
              Select All
            </button>
            <button 
              className="btn btn-ghost" 
              style={{ padding: '4px 10px', fontSize: 12 }}
              onClick={deselectAll}
            >
              Deselect All
            </button>
          </div>
          <span style={{ fontSize: 13, color: 'var(--color-gray-500)' }}>
            {selectedIndices.size} selected
          </span>
        </div>
        
        <div className="modal-body" style={{ padding: 0, maxHeight: 400, overflowY: 'auto' }}>
          {files.map((file, index) => (
            <FilePreviewItem
              key={file.name + index}
              file={file}
              selected={selectedIndices.has(index)}
              onToggle={() => toggleFile(index)}
              onRemove={() => {
                onRemoveFile(index);
                selectedIndices.delete(index);
              }}
              formatSize={formatSize}
            />
          ))}
        </div>
        
        {selectedFiles.length === 0 && (
          <div style={{ padding: '16px 24px', background: 'var(--color-gray-50)', display: 'flex', alignItems: 'center', gap: 8, color: 'var(--color-gray-600)', fontSize: 13 }}>
            <AlertCircle size={16} />
            Select at least one file to process
          </div>
        )}
        
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button 
            className="btn btn-primary" 
            onClick={handleProcess}
            disabled={selectedFiles.length === 0}
          >
            <Play size={16} />
            Process {selectedFiles.length > 0 ? `${selectedFiles.length} File${selectedFiles.length !== 1 ? 's' : ''}` : 'Files'}
          </button>
        </div>
      </div>
    </div>
  );
}

function FilePreviewItem({ file, selected, onToggle, onRemove, formatSize }) {
  const [preview, setPreview] = useState(null);
  
  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  
  return (
    <div 
      style={{ 
        display: 'flex', 
        alignItems: 'center', 
        padding: '12px 24px',
        borderBottom: '1px solid var(--color-gray-100)',
        cursor: 'pointer',
        background: selected ? 'var(--color-gray-50)' : 'transparent',
        transition: 'background 150ms ease'
      }}
      onClick={onToggle}
    >
      <input 
        type="checkbox" 
        checked={selected} 
        onChange={onToggle}
        onClick={(e) => e.stopPropagation()}
        style={{ marginRight: 16, width: 18, height: 18, cursor: 'pointer' }}
      />
      
      <div style={{ 
        width: 64, 
        height: 64, 
        borderRadius: 6, 
        overflow: 'hidden', 
        marginRight: 16,
        background: 'var(--color-gray-100)',
        flexShrink: 0
      }}>
        {preview && (
          <img 
            src={preview} 
            alt={file.name}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        )}
      </div>
      
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ 
          fontWeight: 500, 
          fontSize: 14,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap'
        }}>
          {file.name}
        </div>
        <div style={{ fontSize: 12, color: 'var(--color-gray-500)', marginTop: 2 }}>
          {formatSize(file.size)} • {file.type.split('/')[1]?.toUpperCase() || 'Image'}
        </div>
      </div>
      
      <button 
        className="btn btn-ghost"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        style={{ padding: 8 }}
      >
        <Trash2 size={16} />
      </button>
    </div>
  );
}
