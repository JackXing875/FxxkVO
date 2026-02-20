import torch
import torch.nn as nn

def simple_nms(scores, nms_radius: int):
    """Applies Non-Maximum Suppression (NMS) to extract local maxima.

    Performs NMS on the score map to ensure keypoints are spatially distributed.
    Retains only the maximum score within the defined radius.

    Args:
        scores (torch.Tensor): The input score map tensor.
        nms_radius (int): The radius for non-maximum suppression. Must be >= 0.

    Returns:
        torch.Tensor: The suppressed score map tensor where non-maxima are zeroed.
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
    """SuperPoint Convolutional Neural Network architecture.

    Implements a fully convolutional network that jointly extracts
    interest points and their corresponding descriptors from a single image.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters.
    """
    def __init__(self, config):
        """Initializes the SuperPoint network modules.

        Args:
            config (dict): A dictionary containing model configurations such as
                'nms_radius', 'keypoint_threshold', and 'max_keypoints'.
        """
        super().__init__()
        self.config = config
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        # Shared Encoder.
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Computes the forward pass for keypoint detection and description.

        Args:
            x (torch.Tensor): A batch of grayscale images with shape (B, 1, H, W).

        Returns:
            dict: A dictionary containing the following keys:
                - 'keypoints' (list[torch.Tensor]): List of extracted 2D coordinates
                  with shape (N, 2) for each image in the batch.
                - 'scores' (list[torch.Tensor]): List of confidence scores
                  with shape (N,) for each keypoint.
                - 'descriptors' (torch.Tensor): Dense descriptor map
                  with shape (B, 256, H/8, W/8).
        """
        # Shared encoder forward pass.
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

        # Detector head forward pass.
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        # Remove the 'no interest point' dustbin class.
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        # Apply Non-Maximum Suppression.
        scores = simple_nms(scores, self.config['nms_radius']) 

        # Extract coordinates of points exceeding the confidence threshold.
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores
        ]
        
        # Extract corresponding confidence scores.
        scores_list = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Truncate keypoints based on scores if the maximum limit is exceeded.
        for i in range(b):
            k = keypoints[i]
            s = scores_list[i]
            if len(k) > self.config['max_keypoints']:
                # Sort in descending order and retain the top max_keypoints.
                indices = torch.argsort(s, descending=True)[:self.config['max_keypoints']]
                keypoints[i] = k[indices]
                scores_list[i] = s[indices]

        # Descriptor head forward pass.
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        # Apply L2 normalization to descriptors for subsequent dot-product similarity computation.
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Convert coordinates from (y, x) to (x, y) format to align with OpenCV conventions.
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        return {
            'keypoints': keypoints,       
            'scores': scores_list,        
            'descriptors': descriptors    
        }

# Module verification block.
if __name__ == "__main__":
    import yaml
    import os
    
    # Load configuration.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "..", "configs", "kitti_config.yaml")
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        
    sp_config = full_config['superpoint']
    
    # Initialize SuperPoint model.
    model = SuperPoint(sp_config)
    
    # Load pre-trained weights.
    weight_path = os.path.join(base_dir, "..", "weights", "superpoint_v1.pth")
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    
    print("SuperPoint model initialized and weights loaded successfully.")
    
    # Generate a dummy grayscale input tensor.
    dummy_image = torch.randn(1, 1, 480, 640)
    
    with torch.no_grad():
        out = model(dummy_image)
        
    print(f"Input image shape: {dummy_image.shape}")
    print(f"Number of extracted keypoints: {out['keypoints'][0].shape[0]}")
    print(f"Keypoints tensor shape: {out['keypoints'][0].shape}")
    print(f"Descriptors tensor shape: {out['descriptors'].shape}")