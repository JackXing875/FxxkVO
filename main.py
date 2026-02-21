import os
import cv2
import yaml
from src.tracker import VisualOdometryTracker
from src.visualizer import TrajectoryVisualizer3D

def main():
    """Main entry point for the DeepVO pipeline.

    Orchestrates the visual odometry process by loading configurations, 
    initializing the tracking engine, and managing the real-time 
    visualization loop.
    """
    print("DeepVO System Initializing...")
    
    # Load system configurations from the YAML file.
    config_path = "configs/kitti_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize the Visual Odometry Tracker with pre-trained SuperPoint weights.
    weights_path = "weights/superpoint_v1.pth"
    tracker = VisualOdometryTracker(config, weights_path)
    
    # Initialize the video stream.
    video_path = config['video_path']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file at: {video_path}")
        
    print(f"Video source loaded: {video_path}")
    print("Starting frame-by-frame analysis. Interactive 3D visualizer will be active.")

    # Initialize the 3D trajectory visualization engine.
    viz3d = TrajectoryVisualizer3D()
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Video stream processing completed.")
            break
            
        # Standardize frame resolution based on configuration.
        frame = cv2.resize(frame, (config['image']['width'], config['image']['height']))

        # Execute tracking pipeline to retrieve global 3D coordinates (X, Y, Z).
        pos, debug_img = tracker.process_frame(frame)
        x, y, z = pos[0][0], pos[1][0], pos[2][0]
        
        # Real-time update of the 3D trajectory plot.
        viz3d.update(x, y, z)
        
        # Display the feature tracking dashboard for visual inspection.
        cv2.imshow("DeepVO Feature Tracker (Press 'q' to quit)", debug_img)

        frame_count += 1
        
        # Terminate loop if user presses 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release hardware resources.
    cap.release()
    cv2.destroyAllWindows()
    
    print("Computation finished. 3D trajectory can be manipulated in the interactive window.")
    
    # Define the output directory and file path for the final result.
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "trajectory_3d.png")
    
    # Finalize the visualization session and export high-resolution capture.
    viz3d.close(save_path=save_path)

if __name__ == "__main__":
    main()