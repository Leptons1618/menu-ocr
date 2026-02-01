#!/bin/bash
# Start Menu OCR web application

set -e
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Start Python server
echo "Starting Python API server on http://localhost:3001..."
python server.py &
SERVER_PID=$!

sleep 3

# Start frontend
echo "Starting frontend on http://localhost:3000..."
cd "web/frontend"
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "Menu OCR is running!"
echo "  Frontend: http://localhost:3000"
echo "  API:      http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop"

# Wait for interrupt
trap "kill $SERVER_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
