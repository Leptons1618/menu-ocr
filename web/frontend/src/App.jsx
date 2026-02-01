import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const API_URL = '';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('annotated');

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch(`${API_URL}/api/extract`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Extraction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1
  });

  return (
    <div className="app">
      <header>
        <h1>📋 Menu OCR</h1>
        <p>Extract structured data from restaurant menu images</p>
      </header>

      <main>
        <section className="upload-section">
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                <p>Processing menu...</p>
              </div>
            ) : (
              <>
                <div className="icon">📸</div>
                <p>Drag & drop a menu image, or click to select</p>
                <span className="hint">Supports JPG, PNG, WEBP (max 10MB)</span>
              </>
            )}
          </div>
        </section>

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <section className="results">
            <div className="tabs">
              <button 
                className={activeTab === 'annotated' ? 'active' : ''}
                onClick={() => setActiveTab('annotated')}
              >
                Annotated Image
              </button>
              <button 
                className={activeTab === 'original' ? 'active' : ''}
                onClick={() => setActiveTab('original')}
              >
                Original
              </button>
              <button 
                className={activeTab === 'json' ? 'active' : ''}
                onClick={() => setActiveTab('json')}
              >
                JSON Output
              </button>
            </div>

            <div className="tab-content">
              {activeTab === 'annotated' && (
                <div className="image-container">
                  <img 
                    src={result.images.annotated} 
                    alt="Annotated menu"
                  />
                  <div className="legend">
                    <span className="legend-item section">Section</span>
                    <span className="legend-item group">Group</span>
                    <span className="legend-item item">Item</span>
                    <span className="legend-item price">Price</span>
                    <span className="legend-item desc">Description</span>
                  </div>
                </div>
              )}
              
              {activeTab === 'original' && (
                <div className="image-container">
                  <img 
                    src={result.images.original} 
                    alt="Original menu"
                  />
                </div>
              )}
              
              {activeTab === 'json' && (
                <div className="json-container">
                  <div className="json-header">
                    <span>Extracted Menu Data</span>
                    <button 
                      onClick={() => {
                        navigator.clipboard.writeText(
                          JSON.stringify(result.menu, null, 2)
                        );
                      }}
                    >
                      Copy
                    </button>
                    <button
                      onClick={() => {
                        const blob = new Blob(
                          [JSON.stringify(result.menu, null, 2)], 
                          { type: 'application/json' }
                        );
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'menu.json';
                        a.click();
                      }}
                    >
                      Download
                    </button>
                  </div>
                  <pre>{JSON.stringify(result.menu, null, 2)}</pre>
                </div>
              )}
            </div>

            {result.processingTime && (
              <div className="stats">
                Processing time: {result.processingTime}ms
              </div>
            )}
          </section>
        )}
      </main>

      <footer>
        <p>Menu OCR System • Built with EasyOCR + Rule-based Classification</p>
      </footer>
    </div>
  );
}

export default App;
