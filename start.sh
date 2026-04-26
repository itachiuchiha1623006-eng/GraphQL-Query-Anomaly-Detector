#!/bin/bash

# Configuration
MIDDLEWARE_DIR="middleware"
ML_SERVICE_DIR="ml-service"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Starting GraphQL Anomaly Detector Project...${NC}"

# Start ML Service in the background
echo -e "${GREEN}Starting ML Service...${NC}"
cd $ML_SERVICE_DIR
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "myenv/bin/activate" ]; then
    source myenv/bin/activate
else
    echo -e "${RED}Virtual environment not found! Please ensure venv/myenv exists.${NC}"
    exit 1
fi
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
ML_PID=$!
cd ..

# Start Middleware in the background
echo -e "${GREEN}Starting Middleware...${NC}"
cd $MIDDLEWARE_DIR
npm start &
MIDDLEWARE_PID=$!
cd ..

# Cleanup function to kill background processes on script exit
cleanup() {
    echo -e "\n${RED}Shutting down services...${NC}"
    kill $ML_PID
    kill $MIDDLEWARE_PID
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}All services started!${NC}"
echo -e "${GREEN}ML Service: http://localhost:8000${NC}"
echo -e "${GREEN}Middleware/Dashboard: Check the npm start output for the port.${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services.${NC}"

# Wait for background processes to remain open
wait $ML_PID $MIDDLEWARE_PID
