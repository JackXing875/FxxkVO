/**
 * @file main.cpp
 * @brief Main entry point for the C++ DeepVO pipeline.
 * Orchestrates video capturing, KLT tracking, and real-time trajectory logging.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "deepvo/tracker.h"
#include "deepvo/visualizer.h"

int main(int argc, char* argv[]) {
    std::cout << "  OptiVO C++ Engine Initialized   " << std::endl;

    if (argc < 3) {
        std::cerr << "[Error] Invalid arguments." << std::endl;
        std::cerr << "Usage: ./deepvo_app <input_video_path> <output_csv_path>" << std::endl;
        return -1;
    }

    std::string video_source = argv[1];
    std::string output_csv = argv[2];

    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        1000.0,    0.0,  640.0,
           0.0, 1000.0,  360.0,
           0.0,    0.0,    1.0);

    deepvo::tracker::VisualOdometryTracker tracker(K);

    deepvo::visualization::Visualizer2D viewer("DeepVO - KLT Tracker");

    cv::VideoCapture cap;
    if (video_source == "0") {
        cap.open(0);
    } else {
        cap.open(video_source);
    }

    if (!cap.isOpened()) {
        std::cerr << "[Error] Cannot open video stream: " << video_source << std::endl;
        return -1;
    }

    std::ofstream traj_file(output_csv);
    if (!traj_file.is_open()) {
        std::cerr << "[Error] Could not open trajectory CSV for writing: " << output_csv << std::endl;
        return -1;
    }
    traj_file << "x,y,z\n"; 

    cv::Mat frame;

    // 主循环
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "[Info] End of video stream." << std::endl;
            break;
        }

        bool success = tracker.ProcessFrame(frame);

        if (success) {
            cv::Mat t = tracker.GetCurrentT();
            
            traj_file << t.at<double>(0) << "," 
                      << t.at<double>(1) << "," 
                      << t.at<double>(2) << "\n";
            traj_file.flush(); 

            // 【核心重构】：把画图和退出逻辑全部甩给 Viewer
            std::vector<cv::Point2f> tracked_points = tracker.GetTrackedPoints();
            if (!viewer.ShowFrame(frame, tracked_points, t)) {
                break; // 如果返回 false (按下了 ESC)，直接跳出循环
            }
        }
    }

    cap.release();
    traj_file.close();
    std::cout << "[Info] Shutting down gracefully." << std::endl;
    
    return 0;
}