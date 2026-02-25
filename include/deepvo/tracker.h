#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "deepvo/geometry/epipolar.h"

namespace deepvo {
namespace tracker {

/**
 * @class VisualOdometryTracker
 * @brief Industrial-grade monocular visual odometry frontend utilizing KLT optical flow.
 * * This class maintains the state of the tracked features, accumulates the global 
 * camera ego-motion, and autonomously manages feature detection and re-initialization.
 */
class VisualOdometryTracker {
public:
    /**
     * @brief Constructor initializes the tracker with camera intrinsics.
     * @param K Camera intrinsic matrix (3x3).
     */
    explicit VisualOdometryTracker(const cv::Mat& K);

    /**
     * @brief Processes the next video frame to estimate relative camera motion.
     * @param frame The current RGB or grayscale image frame.
     * @return True if tracking is successful and pose is updated; false otherwise.
     */
    bool ProcessFrame(const cv::Mat& frame);

    /**
     * @brief Retrieves the accumulated global rotation matrix.
     * @return 3x3 double-precision rotation matrix (CV_64F).
     */
    cv::Mat GetCurrentR() const;

    /**
     * @brief Retrieves the accumulated global translation vector.
     * @return 3x1 double-precision translation vector (CV_64F).
     */
    cv::Mat GetCurrentT() const;

    /**
     * @brief Retrieves the successfully tracked keypoints in the current frame.
     * @return Vector of 2D sub-pixel coordinates.
     */
    std::vector<cv::Point2f> GetTrackedPoints() const;

private:
    // Tracking hyperparameters (Industrial Defaults)
    static constexpr int kMinFeatures = 150;
    static constexpr int kMaxFeatures = 1000;
    static constexpr double kMinFeatureDistance = 10.0;
    static constexpr double kPixelMovementThreshold = 2.0;

    // Core geometric solver
    geometry::EpipolarGeometry geo_solver_;

    // Internal state variables
    bool is_initialized_;
    cv::Mat K_;
    cv::Mat prev_frame_gray_;
    std::vector<cv::Point2f> prev_kpts_;

    // Global pose accumulation (Absolute coordinates)
    cv::Mat cur_R_;
    cv::Mat cur_t_;
    double current_scale_;
    double prev_avg_dist_;

    /**
     * @brief Extracts high-quality Shi-Tomasi corners to replenish the feature pool.
     * @param frame_gray The single-channel grayscale image.
     * @param kpts The vector to populate with newly detected points.
     */
    void DetectNewFeatures(const cv::Mat& frame_gray, std::vector<cv::Point2f>& kpts);
};

}  // namespace tracker
}  // namespace deepvo