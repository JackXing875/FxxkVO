# FxxkVO: Asynchronous Monocular Visual Odometry

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![C++14](https://img.shields.io/badge/C++-14%2B-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Ubuntu](https://img.shields.io/badge/Platform-Ubuntu%20%7C%20WSL2-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV_4.0%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Eigen3](https://img.shields.io/badge/Math-Eigen3-8B0000.svg)](https://eigen.tuxfamily.org/)

[中文版本 (Chinese Version)](README_ch.md)

**FxxkVO** is a lightweight, industrial-grade Monocular Visual Odometry (VO) system built from scratch in C++. It features a strictly decoupled multithreaded architecture, separating high-speed frontend tracking from heavy backend non-linear optimization (Bundle Adjustment).


## Key Features

* **Multithreaded Architecture:** Frontend and Backend run in isolated `std::thread` environments with strictly managed `std::mutex` and `std::condition_variable` synchronization, ensuring zero-blocking video processing.
* **Robust Frontend:** Fast and reliable feature tracking using OpenCV's KLT (Kanade-Lucas-Tomasi) optical flow.
* **Industrial Backend Optimization:** Local Bundle Adjustment (LBA) powered by **Ceres Solver** over a sliding window, minimizing reprojection errors to maintain trajectory consistency.
* **Dynamic Map Point Culling:** Intelligent depth filtering to eliminate "small baseline" artifacts and infinity points.
* **Real-Time 3D Visualization:** A decoupled Python viewer using Matplotlib/Open3D for real-time camera trajectory plotting and automated high-resolution static PNG exports.

---

## System Architecture

The pipeline is divided into three completely decoupled modules:

1. **Frontend (Tracking Thread):** Ingests video frames, extracts keypoints, computes optical flow, and estimates the initial $SE(3)$ transformation.
2. **Backend (Mapping Thread):** Wakes up asynchronously when the sliding window queue is filled. Triangulates 3D points and optimizes both camera poses and map points using Ceres.
3. **Visualization (Python Inter-process):** Reads the aggressively flushed CSV and PLY data to render 3D trajectories and point clouds in real-time.

---

## Dependencies

Ensure you have the following libraries installed on your Linux system (Ubuntu/WSL2 recommended):

### C++ Core

* **OpenCV (4.0+)**: Image processing and feature extraction.
* **Eigen3**: Fast linear algebra and matrix operations.
* **Ceres Solver**: Non-linear least squares optimization.

### Python Visualization

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Build & Run

### 1. Build the C++ Engine

Clone the repository and build the project using CMake:

```bash
mkdir build && cd build
cmake ..
make -j4
```

### 2. Run the Pipeline

We provide a unified shell script to launch both the C++ engine and the Python visualization tool:

```bash
# Execute from the project root directory
./scripts/run.sh <path_to_your_video.mp4>
```

*(Note: The system will automatically output `trajectory.csv` and `sparse_map.ply` to the `data/poses/` directory, along with an auto-generated high-resolution trajectory PNG).*

---

## Project Structure

```text
FxxkVO/
├── app/
│   └── main.cpp                  # C++ Engine entry point
├── include/deepvo/
│   ├── tracker.h                 # Frontend KLT tracker
│   ├── backend.h                 # Asynchronous Ceres optimizer
│   ├── map.h                     # Global 3D point cloud map
│   └── visualizer.h              # 2D OpenCV UI wrapper
├── src/
│   ├── tracker.cpp
│   ├── backend.cpp
│   ├── map.cpp
│   └── visualizer.cpp
├── scripts/
│   ├── run_vo.sh                 # Unified launch script
│   └── view_map.py               # Real-time Matplotlib/Open3D visualizer
├── data/
│   └── poses/                    # Auto-generated CSV, PLY, and PNG outputs
├── CMakeLists.txt                # Build configuration
└── README.md
```


---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
