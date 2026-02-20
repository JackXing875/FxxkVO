import cv2
import yaml
import os
import numpy as np
from src.tracker import VisualOdometryTracker

def main():
    print("🚀 DeepVO 系统启动...")
    
    # 加载配置
    with open("configs/kitti_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 初始化 Tracker
    weights_path = "weights/superpoint_v1.pth"
    tracker = VisualOdometryTracker(config, weights_path)
    
    # 打开视频流
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
        
    print(f"成功加载视频: {video_path}")
    print("开始逐帧分析... (按下 'q' 键退出)")

    # 创建一个黑色的画布，用于画出上帝视角的轨迹图 (X-Z 平面)
    traj_img = np.ones((480, 480, 3), dtype=np.uint8) * 40
    
    # 【核心修改】：把原点大幅往上挪！Z 从 400 改成 80
    # 这样上方只留 80 像素，下方留出整整 400 像素的空间！
    origin_x, origin_z = 240, 80 
    
    # 重新画十字坐标系，让十字线跟着原点走
    cv2.line(traj_img, (origin_x, 0), (origin_x, 480), (100, 100, 100), 1) # 竖线
    cv2.line(traj_img, (0, origin_z), (480, origin_z), (100, 100, 100), 1) # 横线
    
    # 缩放比例保持不变
    draw_scale = 15.0

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取完毕！")
            break
            
        # 缩小画面加速处理 (假设你的 MP4 很大，我们固定缩放到 640x480)
        frame = cv2.resize(frame, (config['image']['width'], config['image']['height']))

        # 核心：处理一帧，拿到相机的 3D 世界坐标系坐标 (X, Y, Z)
        pos, debug_img = tracker.process_frame(frame)
        
        # 提取 X 和 Z 坐标用于俯视平面图绘制
        x, y, z = pos[0][0], pos[1][0], pos[2][0]
        
        # 映射到画布像素坐标上
        draw_x = int(x * draw_scale) + origin_x
        draw_z = origin_z - int(z * draw_scale) # OpenCV y轴向下，所以用减法
        
        # 在轨迹图上画一个红色的点
        cv2.circle(traj_img, (draw_x, draw_z), 1, (0, 0, 255), 2)
        
        # 将视频特征图和轨迹图横向拼接在一起展示
        combined = np.hstack((debug_img, traj_img))
        cv2.imshow("Deep Visual Odometry", combined)

        frame_id += 1
        
        # 1 毫秒延迟，按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # 工业习惯：最后把画好的轨迹图保存到硬盘，方便在服务器上脱机查看
    os.makedirs(config['output_dir'], exist_ok=True)
    cv2.imwrite(os.path.join(config['output_dir'], "trajectory.png"), traj_img)
    print("💾 轨迹图已保存到 data/poses/trajectory.png")

if __name__ == "__main__":
    main()