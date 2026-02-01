const API_BASE = 'http://localhost:3001';

export async function extractMenu(file, model = 'rule_based') {
  const formData = new FormData();
  formData.append('image', file);
  
  const response = await fetch(`${API_BASE}/api/extract?model=${encodeURIComponent(model)}&use_gpu=true`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.error || 'Extraction failed');
  }
  
  const data = await response.json();
  return {
    menu: data.menu,
    images: {
      original: `${API_BASE}${data.images.original}`,
      annotated: `${API_BASE}${data.images.annotated}`,
    },
    processingTime: Math.round(data.processing_time_ms),
    modelUsed: data.model_used,
  };
}

export async function getModels() {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) {
    throw new Error('Failed to fetch models');
  }
  return response.json();
}

export async function healthCheck() {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}
