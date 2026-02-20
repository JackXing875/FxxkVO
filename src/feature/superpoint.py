import torch
import torch.nn as nn

def simple_nms(scores, nms_radius: int):
    """
    非极大值抑制 (Non-Maximum Suppression)
    作用：防止提取出的特征点全部扎堆在一起。在半径 nms_radius 内，只保留分数最高的那一个点。
    """
    assert(nms_radius >= 0)
    
    def max_pool(x):
        return nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

class SuperPoint(nn.Module):
    """SuperPoint 卷积神经网络"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # 1. 共享编码器 (Shared Encoder)
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # 2. 特征点解码器 (Detector Head)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # 3. 描述子解码器 (Descriptor Head)
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """ 
        输入 x: (Batch, 1, H, W) 的灰度图张量
        """
        # --- 共享编码器前向传播 ---
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # --- 特征点头 (Detector) ---
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1] # 抛弃最后一维的"无特征点"类别
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius']) # 应用非极大值抑制

        # 提取满足阈值的点坐标
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores
        ]
        
        # 提取对应的置信度分数
        scores_list = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # 如果点太多，按分数截断
        for i in range(b):
            k = keypoints[i]
            s = scores_list[i]
            if len(k) > self.config['max_keypoints']:
                # 按照分数降序排列，只取前 max_keypoints 个
                indices = torch.argsort(s, descending=True)[:self.config['max_keypoints']]
                keypoints[i] = k[indices]
                scores_list[i] = s[indices]

        # --- 描述子头 (Descriptor) ---
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        # 对描述子进行 L2 归一化，这样后续计算两个点的相似度时，直接算点积就可以了
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # 把坐标从 (y, x) 翻转为 (x, y) 格式，适配 OpenCV 习惯
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        return {
            'keypoints': keypoints,       # 提取出的 2D 坐标列表 [ (N, 2) ]
            'scores': scores_list,        # 每个点的置信度 [ (N,) ]
            'descriptors': descriptors    # 整张图的密集描述子特征图 (B, 256, H/8, W/8)
        }

# === 本地测试代码 ===
if __name__ == "__main__":
    import yaml
    import os
    
    # 模拟读取配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "..", "configs", "kitti_config.yaml")
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        
    sp_config = full_config['superpoint']
    
    # 初始化模型
    model = SuperPoint(sp_config)
    
    # 加载权重
    weight_path = os.path.join(base_dir, "..", "weights", "superpoint_v1.pth")
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    
    print("✅ SuperPoint 网络构建并加载权重成功！")
    
    # 模拟一张 640x480 的灰度输入图片 (BatchSize=1, Channel=1, H=480, W=640)
    dummy_image = torch.randn(1, 1, 480, 640)
    
    with torch.no_grad():
        out = model(dummy_image)
        
    print(f"输入图像形状: {dummy_image.shape}")
    print(f"提取到的特征点数量: {out['keypoints'][0].shape[0]} 个")
    print(f"特征点坐标张量形状: {out['keypoints'][0].shape}")
    print(f"描述子特征图形状: {out['descriptors'].shape}")