import numpy as np
import torch
import cv2
from .feature.superpoint import SuperPoint
from .geometry.epipolar import EpipolarGeometry

class VisualOdometryTracker:
    """Core tracking system for Monocular Visual Odometry.

    Integrates the SuperPoint deep learning frontend for feature extraction
    and the epipolar geometry backend for camera pose estimation. Manages
    global trajectory states and keyframe selection based on parallax.
    """
    def __init__(self, config, weights_path):
        """Initializes the visual odometry tracker.

        Args:
            config (dict): Configuration dictionary containing model and camera parameters.
            weights_path (str): File path to the pre-trained SuperPoint weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sp = SuperPoint(config['superpoint']).to(self.device)
        self.sp.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.sp.eval()
        
        self.geo = EpipolarGeometry(config['cam_intrinsics'])
        
        # Initialize global pose (Rotation and Translation matrices).
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        
        # Store the state of the last registered keyframe.
        self.keyframe_kpts = None
        self.keyframe_desc = None
        
        # Parallax threshold for keyframe generation (in pixels).
        self.pixel_movement_threshold = 2.0

    def process_frame(self, frame_bgr):
        """Processes a single video frame to update the global camera trajectory.

        Extracts features, matches them against the last keyframe, evaluates
        parallax, and computes the relative pose if the baseline is sufficient.

        Args:
            frame_bgr (np.ndarray): The current BGR image frame from the video stream.

        Returns:
            tuple: A tuple containing:
                - current_translation (np.ndarray): The updated 3x1 global translation vector.
                - debug_image (np.ndarray): The frame annotated with tracked features.
        """
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

        # Register the first frame as the initial keyframe.
        if self.keyframe_kpts is None:
            self.keyframe_kpts = kpts
            self.keyframe_desc = desc
            return self.cur_t.copy(), debug_img

        # Match features between the current frame and the last keyframe.
        idx1, idx2 = self.geo.match_features(self.keyframe_desc, desc)
        
        if len(idx1) > 8:
            matched_kpts1 = self.keyframe_kpts[idx1]
            matched_kpts2 = kpts[idx2]
            
            # Calculate average optical flow magnitude (parallax) for tracked features.
            distances = np.linalg.norm(matched_kpts1 - matched_kpts2, axis=1)
            avg_dist = np.mean(distances)
            
            for pt1, pt2 in zip(matched_kpts1, matched_kpts2):
                cv2.line(debug_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 1)

            # Process geometry only if sufficient parallax (wide baseline) is observed.
            if avg_dist > self.pixel_movement_threshold:
                R, t, inlier_mask = self.geo.estimate_pose(matched_kpts1, matched_kpts2)
                
                if R is not None and t is not None:
                    # Inject pseudo-scale heuristic: map average pixel disparity to physical translation magnitude.
                    fake_scale = avg_dist / 20.0 
                    
                    # Update global pose using scaled translation and accumulated rotation.
                    self.cur_t = self.cur_t + self.cur_R.dot(t) * fake_scale
                    self.cur_R = self.cur_R.dot(R)
                    
                    # Update keyframe state.
                    self.keyframe_kpts = kpts
                    self.keyframe_desc = desc

        return self.cur_t.copy(), debug_img