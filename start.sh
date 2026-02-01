#!/bin/bash
# Start Menu OCR web application

# Kill any existing processes
pkill -f "node.*server.js" 2>/dev/null
pkill -f "vite" 2>/dev/null

# Start backend
echo "Starting backend on http://localhost:3001..."
cd "$(dirname "$0")/web/backend"
node server.js &
BACKEND_PID=$!

# Wait for backend
sleep 2

# Start frontend
echo "Starting frontend on http://localhost:3000..."
cd "../frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Menu OCR is running!"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop"

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
