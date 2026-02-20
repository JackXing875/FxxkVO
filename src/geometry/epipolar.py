import cv2
import numpy as np

class EpipolarGeometry:
    def __init__(self, camera_intrinsics: dict):
        """
        初始化时传入相机内参 K
        """
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # 构建 3x3 的相机内参矩阵 K
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray):
        """
        特征匹配：在两帧之间寻找相同的物理点
        desc1, desc2: 形状为 (N, 256) 的特征描述子
        返回: 匹配对的索引 (match_idx1, match_idx2)
        """
        # 使用 OpenCV 的暴力匹配器 (Brute-Force Matcher)
        # 因为 SuperPoint 的描述子已经做了 L2 归一化，使用 L2 距离即可
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # 使用 KNN 匹配 (k=2)，为每个点找两个最相似的点
        # 为了进行 Lowe's Ratio Test (滤除那些模棱两可的错误匹配)
        knn_matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in knn_matches:
            # Ratio Test: 如果第一名比第二名明显好很多 (距离小于 0.8 倍)，才认为是可靠匹配
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
                
        # 提取匹配点的索引
        idx1 = np.array([m.queryIdx for m in good_matches])
        idx2 = np.array([m.trainIdx for m in good_matches])
        
        return idx1, idx2

    def estimate_pose(self, kpts1: np.ndarray, kpts2: np.ndarray):
        """
        核心数学解算：通过 2D 匹配点对，计算相机的 3D 运动 (R, t)
        kpts1, kpts2: 形状为 (M, 2) 的匹配点坐标
        """
        # 至少需要 8 个点才能计算本质矩阵 (八点法)
        if len(kpts1) < 8 or len(kpts2) < 8:
            return None, None, None
            
        # 1. 计算本质矩阵 E (使用 RANSAC 算法剔除误匹配的离群点)
        E, mask = cv2.findEssentialMat(
            kpts1, kpts2, self.K, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is None:
            return None, None, None
            
        # 2. 从本质矩阵 E 中恢复旋转 R 和平移 t (内部使用 SVD 分解)
        # 这一步 OpenCV 会自动帮我们验证 4 种可能的数学解，并选出物理上合理(点在相机前方)的那个
        _, R, t, mask_pose = cv2.recoverPose(E, kpts1, kpts2, self.K, mask=mask)
        
        # 最终被认为是有效内点 (Inliers) 的布尔掩码
        inlier_mask = mask_pose.ravel() > 0
        
        return R, t, inlier_mask

# === 本地测试代码 ===
if __name__ == "__main__":
    # 模拟相机内参
    intrinsics = {'fx': 800, 'fy': 800, 'cx': 320, 'cy': 240}
    geo = EpipolarGeometry(intrinsics)
    
    # 模拟 10 个完美的物理点在前移时的像素坐标变化 (纯平移 t_z)
    pts1 = np.array([[320, 240], [330, 240], [320, 250], [310, 240], [320, 230],
                     [400, 300], [250, 180], [450, 200], [200, 400], [350, 350]], dtype=np.float64)
                     
    # 假设相机往前走，画面中的点会向四周扩散
    pts2 = pts1 + (pts1 - np.array([320, 240])) * 0.1 
    
    R, t, mask = geo.estimate_pose(pts1, pts2)
    
    print("✅ 极线几何模块测试通过！")
    print("解算出的旋转矩阵 R (应接近单位矩阵):\n", R)
    print("解算出的平移向量 t (应主要在 Z 轴方向):\n", t)