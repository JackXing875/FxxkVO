#!/bin/bash
# ---------------------------------------------------------------------------
# DeepVO Industrial Launch Script
# Orchestrates the C++ backend and Python 3D visualization frontend.
# ---------------------------------------------------------------------------

# Define explicit paths based on project structure
VIDEO_PATH="./data/dataset/test_video.mp4"
POSE_DIR="./data/poses"
POSE_FILE="${POSE_DIR}/trajectory.csv"
CPP_EXECUTABLE="./build/deepvo_app"
PYTHON_VIEWER="./scripts/viewer_3d.py"
PYTHON_EXECUTABLE="./venv/bin/python"

# Ensure the output directory exists
mkdir -p ${POSE_DIR}

# Clear previous trajectory data to prevent plotting old ghost paths
if [ -f "$POSE_FILE" ]; then
    rm "$POSE_FILE"
fi
touch "$POSE_FILE"

echo "  Starting DeepVO Launch Sequence      "

# Function to gracefully kill the Python background process on exit
cleanup() {
    echo -e "\n[Launch Script] Shutting down DeepVO system..."
    # Kill the Python process started in the background
    kill $PYTHON_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (Ctrl+C) and EXIT signals to trigger the cleanup function
trap cleanup SIGINT EXIT

# 1. Launch the Python 3D Viewer in the background (&)
echo "[Launch Script] Starting Python 3D Visualizer in background..."
${PYTHON_EXECUTABLE} ${PYTHON_VIEWER} ${POSE_FILE} &
PYTHON_PID=$!

# Give Python a second to initialize the Matplotlib window
sleep 1 

# 2. Launch the C++ Core Engine in the foreground
echo "[Launch Script] Igniting C++ VO Engine..."
${CPP_EXECUTABLE} ${VIDEO_PATH} ${POSE_FILE}

# Wait for background processes to finish (if C++ exits naturally)
wait $PYTHON_PID