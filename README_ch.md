
# DeepVO: 深度学习驱动的单目视觉里程计

[English Version](README.md)

DeepVO 是一款单目视觉里程计（VO）系统。它将自监督深度学习的精确性与传统极线几何的严谨性相结合，利用 **SuperPoint** 进行特征追踪，并配合稳健的几何求解器，实现从单路视频流中估算相机 3D 运动轨迹。



## 系统架构

系统采用前后端解耦设计：
* **前端**：基于 SuperPoint 的关键点提取与描述子生成。
* **后端**：利用本质矩阵 ($E$) 结合 RANSAC 算法进行位姿恢复。
* **关键帧逻辑**：基于视差的触发机制，确保几何解算的稳定性并处理单目尺度问题。

## 核心特性
* **深度学习前端**：集成 SuperPoint 模型，在不同光照条件下均能保持稳健的特征追踪。
* **关键帧管理**：自适应关键帧选择，有效抑制零点漂移，确保宽基线匹配。
* **伪尺度启发算法**：先进的像素视差映射技术，解决单目相机固有的尺度模糊问题。
* **交互式 3D 可视化**：实时轨迹渲染，支持鼠标旋转交互及高清结果导出。

## 项目结构
```text
DeepVO/
├── configs/          # 相机内参及系统超参数配置
├── data/             # 输入视频数据及输出轨迹日志
├── src/
│   ├── feature/      # SuperPoint 网络实现 (Google 代码规范)
│   ├── geometry/     # 极线几何求解与位姿恢复
│   ├── tracker.py    # VO 状态机与关键帧逻辑
│   └── visualizer.py # 交互式 3D 渲染引擎
├── weights/          # 预训练模型权重
└── main.py           # 程序入口

```

## 快速开始

- **克隆与环境配置**：
```bash
git clone [https://github.com/JackXing875/DeepVO.git](https://github.com/JackXing875/DeepVO.git)
pip install -r requirements.txt

```


- **运行**：
```bash
python main.py

```



## 许可证

本项目采用 **GNU General Public License v3.0** 开源协议。详情请参阅 [LICENSE](https://www.google.com/search?q=LICENSE) 文件。

