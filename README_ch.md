# FxxkVO: 异步单目视觉里程计

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![C++14](https://img.shields.io/badge/C++-14%2B-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Ubuntu](https://img.shields.io/badge/Platform-Ubuntu%20%7C%20WSL2-E95420?logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV_4.0%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Eigen3](https://img.shields.io/badge/Math-Eigen3-8B0000.svg)](https://eigen.tuxfamily.org/)

[English Version](README.md)

**FxxkVO** 是一个从零开始使用 C++ 构建的轻量级、工业级单目视觉里程计 (VO) 系统。它采用严格解耦的多线程架构，将高速的前端追踪与计算密集的后端非线性优化 (光束法平差 / Bundle Adjustment) 彻底分离。

## 核心特性

* **多线程架构：** 前端和后端运行在完全独立的 `std::thread` 线程环境中，通过严格管理的 `std::mutex` 和 `std::condition_variable` 进行同步，确保视频处理达到零阻塞的极速体验。
* **鲁棒的前端追踪：** 使用 OpenCV 的 KLT (Kanade-Lucas-Tomasi) 光流法进行快速、可靠的特征点追踪。
* **工业级后端优化：** 基于 **Ceres Solver** 在滑动窗口上执行局部光束法平差 (Local Bundle Adjustment, LBA)，有效最小化重投影误差，保持轨迹的全局一致性。
* **动态地图点修剪：** 引入智能深度滤波机制，无情剔除由“短基线”引起的数学伪影和无穷远噪点。
* **实时 3D 可视化：** 采用完全解耦的 Python 跨进程视图器，结合 Matplotlib 和 Open3D 实现相机轨迹的实时 3D 渲染，并支持自动导出高分辨率的静态 PNG 轨迹图。

---

## 系统架构

本系统的管线 (Pipeline) 被划分为三个完全解耦的模块：

1. **前端 (追踪线程)：** 摄入视频帧，提取关键点，计算光流，并初步估计相机的 $SE(3)$ 变换矩阵。
2. **后端 (建图线程)：** 当滑动窗口队列填满时被异步唤醒。利用 Ceres 引擎执行 3D 空间点三角化，并同时对相机位姿和地图点进行联合优化。
3. **可视化 (Python 跨进程)：** 实时读取极速刷盘的 CSV 和 PLY 数据流，渲染 3D 轨迹和环境点云。

---

## 环境依赖 (Dependencies)

请确保您的 Linux 系统（强烈推荐 Ubuntu 或 WSL2）已安装以下依赖库：

### C++ 核心算法端

* **OpenCV (4.0+)**: 用于图像处理和特征提取。
* **Eigen3**: 提供高性能的线性代数与矩阵运算支持。
* **Ceres Solver**: 用于非线性最小二乘优化。

### Python 可视化端

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 编译与运行 (Build & Run)

### 1. 编译 C++ 引擎

克隆本仓库，并使用 CMake 构建项目：

```bash
mkdir build && cd build
cmake ..
make -j4
```

### 2. 运行系统管线

我们提供了一个统一的 Shell 脚本，用于一键启动 C++ 引擎和 Python 可视化工具：

```bash
# 请在项目根目录下执行
./scripts/run.sh <您的视频路径.mp4>
```

*(注意：系统运行结束后，会自动将 `trajectory.csv` 和 `sparse_map.ply` 输出至 `data/poses/` 目录，并自动生成一张高分辨率的 3D 轨迹图 PNG)。*

---

## 项目结构 (Project Structure)

```text
FxxkVO/
├── app/
│   └── main.cpp                  # C++ 引擎主入口
├── include/deepvo/
│   ├── tracker.h                 # 前端 KLT 追踪器头文件
│   ├── backend.h                 # 异步 Ceres 优化器头文件
│   ├── map.h                     # 全局 3D 点云地图头文件
│   └── visualizer.h              # 2D OpenCV UI 封装头文件
├── src/
│   ├── tracker.cpp
│   ├── backend.cpp
│   ├── map.cpp
│   └── visualizer.cpp
├── scripts/
│   ├── run_vo.sh                 # 统一启动脚本
│   └── view_map.py               # 实时 Matplotlib/Open3D 可视化脚本
├── data/
│   └── poses/                    # 自动生成的 CSV, PLY 及 PNG 输出目录
├── CMakeLists.txt                # CMake 构建配置
└── README.md
```

---

## 开源协议 (License)

本项目遵循 **GNU General Public License v3.0** 开源协议。详情请查阅 [LICENSE](https://www.google.com/search?q=LICENSE) 文件。
