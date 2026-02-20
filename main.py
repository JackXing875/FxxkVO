import cv2
import yaml
import os
from src.tracker import VisualOdometryTracker
from src.visualizer import TrajectoryVisualizer3D  # 引入我们刚写的 3D 渲染器

def main():
    print("🚀 DeepVO 系统启动...")
    
    with open("configs/kitti_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    tracker = VisualOdometryTracker(config, "weights/superpoint_v1.pth")
    
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
        
    print(f"✅ 成功加载视频: {video_path}")
    print("开始逐帧分析... (3D 弹窗即将出现！)")

    # 1. 初始化 3D 渲染引擎
    viz3d = TrajectoryVisualizer3D()
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕！")
            break
            
        frame = cv2.resize(frame, (config['image']['width'], config['image']['height']))

        # 获取当前帧的 3D 坐标 (X, Y, Z)
        pos, debug_img = tracker.process_frame(frame)
        x, y, z = pos[0][0], pos[1][0], pos[2][0]
        
        # 2. 实时更新 3D 轨迹图！
        viz3d.update(x, y, z)
        
        # 我们依然保留 OpenCV 的窗口，用来实时看神经网络提取特征点的工作状态
        cv2.imshow("DeepVO Feature Tracker (Press 'q' to quit)", debug_img)

        frame_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("💾 计算完成，请在 3D 窗口中自由拖拽查看轨迹！(关闭图形窗口以结束程序)")
    # 3. 保持 3D 窗口开启
    viz3d.close()

if __name__ == "__main__":
    main()