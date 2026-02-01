import React, { useCallback, useState } from 'react';
import { Upload } from 'lucide-react';

export default function UploadZone({ onUpload, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);
  
  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);
  
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (disabled) return;
    
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (files.length > 0) {
      onUpload(files);
    }
  }, [disabled, onUpload]);
  
  const handleClick = useCallback(() => {
    if (disabled) return;
    
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    input.onchange = (e) => {
      const files = Array.from(e.target.files);
      if (files.length > 0) {
        onUpload(files);
      }
    };
    input.click();
  }, [disabled, onUpload]);
  
  return (
    <div
      className={`upload-zone ${isDragging ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <div className="upload-icon">
        <Upload size={48} />
      </div>
      <p className="upload-text">
        Drop menu images here, or click to browse
      </p>
      <p className="upload-hint">
        Supports JPG, PNG, WebP up to 10MB. Select multiple files for batch processing.
      </p>
    </div>
  );
}
