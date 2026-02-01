import React, { useState } from 'react';
import { X, Copy, Download, List, Code } from 'lucide-react';

function flattenMenu(menu) {
  const items = [];
  for (const section of menu?.menu || []) {
    for (const group of section.groups || []) {
      for (const item of group.items || []) {
        items.push({
          section: section.label,
          group: group.label,
          name: item.name,
          price: item.price,
          description: item.description,
        });
      }
    }
  }
  return items;
}

export default function JsonModal({ data, onClose }) {
  const [view, setView] = useState('table');
  
  if (!data) return null;
  
  const items = flattenMenu(data);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  };
  
  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'menu.json';
    a.click();
    URL.revokeObjectURL(url);
  };
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Extracted Menu Data</h2>
          <button className="modal-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>
        
        <div className="tabs" style={{ padding: '0 24px' }}>
          <button 
            className={`tab ${view === 'table' ? 'active' : ''}`}
            onClick={() => setView('table')}
          >
            <List size={16} style={{ marginRight: 6 }} />
            Table View
          </button>
          <button 
            className={`tab ${view === 'json' ? 'active' : ''}`}
            onClick={() => setView('json')}
          >
            <Code size={16} style={{ marginRight: 6 }} />
            JSON View
          </button>
        </div>
        
        <div className="modal-body">
          {view === 'table' ? (
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Section</th>
                    <th>Group</th>
                    <th>Item Name</th>
                    <th>Price</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, idx) => (
                    <tr key={idx}>
                      <td>{item.section || '-'}</td>
                      <td>{item.group || '-'}</td>
                      <td>{item.name}</td>
                      <td>{item.price != null ? item.price : '-'}</td>
                      <td>{item.description || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {items.length === 0 && (
                <p style={{ textAlign: 'center', padding: 32, color: '#737373' }}>
                  No items extracted
                </p>
              )}
            </div>
          ) : (
            <div className="json-container">
              <div className="json-content">
                <pre>{JSON.stringify(data, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
        
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={handleCopy}>
            <Copy size={16} />
            Copy JSON
          </button>
          <button className="btn btn-primary" onClick={handleDownload}>
            <Download size={16} />
            Download
          </button>
        </div>
      </div>
    </div>
  );
}
