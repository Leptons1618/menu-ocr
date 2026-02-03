import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:3001',
      '/uploads': 'http://localhost:3001',
      '/output': 'http://localhost:3001',
      '/paper': 'http://localhost:3001'
    },
    // Allow all hosts for ngrok tunneling
    allowedHosts: ['all','dusti-xylographic-jere.ngrok-free.dev']
  }
});
