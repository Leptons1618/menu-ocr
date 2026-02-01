import React, { createContext, useContext, useReducer, useEffect } from 'react';

const AppContext = createContext(null);

const SESSION_KEY = 'menu-ocr-session';

const initialState = {
  uploads: [],
  currentResult: null,
  processingQueue: [], // { id, filename, status: 'queued' | 'processing' | 'completed' | 'error', error? }
  error: null,
};

function loadFromSession() {
  try {
    const data = sessionStorage.getItem(SESSION_KEY);
    if (data) {
      const state = JSON.parse(data);
      // Reset processing queue on page load (don't resume processing)
      return { ...state, processingQueue: [], error: null };
    }
  } catch (e) {
    console.error('Failed to load session:', e);
  }
  return initialState;
}

function saveToSession(state) {
  try {
    // Don't save processingQueue to session (transient state)
    const { processingQueue, ...persistState } = state;
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(persistState));
  } catch (e) {
    console.error('Failed to save session:', e);
  }
}

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_TO_QUEUE':
      return {
        ...state,
        processingQueue: [...state.processingQueue, ...action.payload],
      };
    
    case 'UPDATE_QUEUE_ITEM':
      return {
        ...state,
        processingQueue: state.processingQueue.map(item =>
          item.id === action.payload.id ? { ...item, ...action.payload } : item
        ),
      };
    
    case 'REMOVE_FROM_QUEUE':
      return {
        ...state,
        processingQueue: state.processingQueue.filter(item => item.id !== action.payload),
      };
    
    case 'CLEAR_QUEUE':
      return {
        ...state,
        processingQueue: [],
      };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'ADD_RESULT':
      const newUploads = [...state.uploads, action.payload];
      return {
        ...state,
        uploads: newUploads,
        currentResult: action.payload,
        error: null,
      };
    
    case 'SET_CURRENT':
      return { ...state, currentResult: action.payload };
    
    case 'REMOVE_RESULT':
      const filtered = state.uploads.filter(u => u.id !== action.payload);
      return {
        ...state,
        uploads: filtered,
        currentResult: state.currentResult?.id === action.payload ? null : state.currentResult,
      };
    
    case 'CLEAR_ALL':
      return initialState;
    
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, null, loadFromSession);
  
  // Save to session on changes (sessionStorage persists on refresh, clears on tab close)
  useEffect(() => {
    saveToSession(state);
  }, [state]);
  
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
}
