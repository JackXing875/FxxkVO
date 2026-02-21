
# DeepVO: Deep Learning-Driven Monocular Visual Odometry

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)

[中文版本 (Chinese Version)](README_ch.md)

DeepVO is an industrial-grade Monocular Visual Odometry (VO) pipeline that combines the precision of self-supervised deep learning with the rigor of classical epipolar geometry. By leveraging **SuperPoint** for feature tracking and robust geometric solvers, DeepVO estimates 3D camera trajectories from a single monocular video stream.



## System Architecture

The pipeline follows a decoupled frontend-backend design:
* **Frontend**: SuperPoint-based feature extraction (keypoints & descriptors).
* **Backend**: Pose recovery using the Essential Matrix ($E$) and RANSAC-based outlier rejection.
* **Keyframe Logic**: Parallax-based triggering to ensure geometric stability and handle monocular scale.

## Key Features
* **Deep Learning Frontend**: Integrated SuperPoint model for robust feature tracking across illumination changes.
* **Keyframe Management**: Adaptive keyframe selection to mitigate zero-drift and ensure wide-baseline matching.
* **Pseudo-scale Heuristic**: Advanced pixel-parallax mapping to address monocular scale ambiguity.
* **Interactive 3D Visualizer**: Real-time trajectory rendering with interactive rotation and high-resolution export.

## Project Structure
```text
DeepVO/
├── configs/          # Camera intrinsics and system hyperparameters
├── data/             # Input video datasets and output trajectory logs
├── src/
│   ├── feature/      # SuperPoint implementation
│   ├── geometry/     # Epipolar solvers and pose recovery
│   ├── tracker.py    # VO state machine and keyframe logic
│   └── visualizer.py # Interactive 3D rendering engine
├── weights/          # Model weights (superpoint_v1.pth)
└── main.py           # Application entry point
```

## Quick Start

- **Clone & Dependencies**:
```bash
git clone https://github.com/JackXing875/DeepVO.git
pip install -r requirements.txt
```


- **Execute**:
```bash
python3 main.py
```



## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

