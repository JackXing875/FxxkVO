import numpy as np
import torch
import cv2
from .feature.superpoint import SuperPoint
from .geometry.epipolar import EpipolarGeometry

class VisualOdometryTracker:
    def __init__(self, config, weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sp = SuperPoint(config['superpoint']).to(self.device)
        self.sp.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.sp.eval()
        
        self.geo = EpipolarGeometry(config['cam_intrinsics'])
        
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        
        # 记忆上一“关键帧”的信息
        self.keyframe_kpts = None
        self.keyframe_desc = None
        
        # 关键帧触发阈值：平均像素移动距离
        self.pixel_movement_threshold = 2.0

    def process_frame(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            out = self.sp(img_tensor)
            
        kpts = out['keypoints'][0].cpu().numpy()
        desc_map = out['descriptors'][0].cpu().numpy() 
        C, H_out, W_out = desc_map.shape
        
        kpts_scaled = kpts / 8.0
        kpts_scaled = np.clip(kpts_scaled, 0, [W_out-1, H_out-1]).astype(int)
        desc = desc_map[:, kpts_scaled[:, 1], kpts_scaled[:, 0]].T
        
        debug_img = frame_bgr.copy()
        for x, y in kpts:
            cv2.circle(debug_img, (int(x), int(y)), 3, (0, 255, 0), -1)

        # 初始第一帧，直接设为关键帧
        if self.keyframe_kpts is None:
            self.keyframe_kpts = kpts
            self.keyframe_desc = desc
            return self.cur_t.copy(), debug_img

        # 匹配当前帧与“上一个关键帧”
        idx1, idx2 = self.geo.match_features(self.keyframe_desc, desc)
        
        if len(idx1) > 8:
            matched_kpts1 = self.keyframe_kpts[idx1]
            matched_kpts2 = kpts[idx2]
            
            # 计算特征点在画面上的平均移动像素距离
            distances = np.linalg.norm(matched_kpts1 - matched_kpts2, axis=1)
            avg_dist = np.mean(distances)
            
            for pt1, pt2 in zip(matched_kpts1, matched_kpts2):
                cv2.line(debug_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 1)

            # 【核心补丁】：只有画面变化足够大（基线足够宽），才进行矩阵解算
            # 【核心补丁】：阈值降低，且注入伪尺度
            if avg_dist > self.pixel_movement_threshold:
                R, t, inlier_mask = self.geo.estimate_pose(matched_kpts1, matched_kpts2)
                
                if R is not None and t is not None:
                    # 伪尺度 (Pseudo-scale) 技巧：
                    # 画面像素变动越大，我们就假设物理世界走得越远
                    fake_scale = avg_dist / 20.0 
                    
                    # 使用带有假尺度的平移来更新坐标，这样转弯的细节就能显现出来
                    self.cur_t = self.cur_t + self.cur_R.dot(t) * fake_scale
                    self.cur_R = self.cur_R.dot(R)
                    
                    self.keyframe_kpts = kpts
                    self.keyframe_desc = desc

        return self.cur_t.copy(), debug_img