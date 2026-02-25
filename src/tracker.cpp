#include "deepvo/tracker.h"
#include <numeric>
#include <algorithm>
#include <iostream>

namespace deepvo {
namespace tracker {

VisualOdometryTracker::VisualOdometryTracker(const cv::Mat& K)
    : geo_solver_(K), is_initialized_(false), K_(K.clone()), 
      current_scale_(1.0), prev_avg_dist_(-1.0) {
    
    // Initialize global pose at the origin facing forward
    cur_R_ = cv::Mat::eye(3, 3, CV_64F);
    cur_t_ = cv::Mat::zeros(3, 1, CV_64F);
}

void VisualOdometryTracker::DetectNewFeatures(const cv::Mat& frame_gray, std::vector<cv::Point2f>& kpts) {
    // Utilize Shi-Tomasi corner detection, which provides excellent stability for optical flow
    cv::goodFeaturesToTrack(
        frame_gray, kpts, 
        kMaxFeatures, 
        0.01,                 // Quality level
        kMinFeatureDistance,  // Enforce uniform spatial distribution
        cv::Mat(), 
        3,                    // Block size
        false, 
        0.04
    );
}

bool VisualOdometryTracker::ProcessFrame(const cv::Mat& frame) {
    cv::Mat frame_gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    } else {
        frame_gray = frame.clone();
    }

    // ---------------------------------------------------------
    // Phase 1: System Initialization
    // ---------------------------------------------------------
    if (!is_initialized_) {
        DetectNewFeatures(frame_gray, prev_kpts_);
        prev_frame_gray_ = frame_gray;
        is_initialized_ = true;
        std::cout << "[Tracker] System initialized with " << prev_kpts_.size() << " features." << std::endl;
        return true;
    }

    // ---------------------------------------------------------
    // Phase 2: KLT Optical Flow Tracking
    // ---------------------------------------------------------
    std::vector<cv::Point2f> cur_kpts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Calculate sparse optical flow utilizing image pyramids
    cv::calcOpticalFlowPyrLK(
        prev_frame_gray_, frame_gray, 
        prev_kpts_, cur_kpts, 
        status, err, 
        cv::Size(21, 21), 
        3 // Max pyramid level
    );

    // Filter out points that the KLT algorithm failed to track
    std::vector<cv::Point2f> matched_prev, matched_cur;
    double total_dist = 0.0;
    
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            matched_prev.push_back(prev_kpts_[i]);
            matched_cur.push_back(cur_kpts[i]);
            
            // Accumulate pixel displacement to estimate relative physical scale later
            total_dist += cv::norm(cur_kpts[i] - prev_kpts_[i]);
        }
    }

    if (matched_cur.empty()) {
        std::cerr << "[Tracker] Fatal: Complete loss of tracking." << std::endl;
        is_initialized_ = false;
        return false;
    }

    double avg_dist = total_dist / matched_cur.size();

    // ---------------------------------------------------------
    // Phase 3: Motion Estimation and Implicit Keyframe Management
    // ---------------------------------------------------------
    
    // Require a minimum average pixel displacement to trigger pose estimation.
    // This threshold filters out micro-jitter and ensures a sufficiently wide 
    // baseline for accurate epipolar geometry computation.
    if (avg_dist > 2.0) {
        cv::Mat R, t, inlier_mask;
        std::tie(R, t, inlier_mask) = geo_solver_.EstimatePose(matched_prev, matched_cur);

        // Ensure the pose recovery succeeded and is supported by a robust number of inliers.
        // This prevents aggressive trajectory jumps caused by severe motion blur or dynamic objects.
        if (!R.empty() && !t.empty() && cv::countNonZero(inlier_mask) > 15) {
            
            // Inject pseudo-scale heuristic: map average pixel disparity to physical translation magnitude.
            // This dynamic scaling allows the monocular system to approximate velocity changes.
            double fake_scale = avg_dist / 20.0; 

            // Convert matrices to double precision (CV_64F) prior to multiplication 
            // to maintain strict numerical stability.
            R.convertTo(R, CV_64F);
            t.convertTo(t, CV_64F);

            // Update the global trajectory.
            // Strict rotation limits are deliberately omitted here to accommodate 
            // aggressive cornering movements in the dataset.
            cur_t_ = cur_t_ + fake_scale * (cur_R_ * t);
            cur_R_ = cur_R_ * R;

            // Implicit Keyframe Mechanism:
            // Only update the reference features (prev_kpts_) when a valid camera movement is recorded.
            // If the camera is nearly stationary, the KLT tracker will safely continue tracking 
            // from the last valid keyframe, accumulating disparity until it crosses the threshold.
            prev_kpts_ = matched_cur;
            prev_frame_gray_ = frame_gray;
        }
    }

    // ---------------------------------------------------------
    // Phase 4: Feature Pool Management
    // ---------------------------------------------------------
    
    // Replenish the feature pool if a significant number of points are lost 
    // due to occlusion, severe rotation, or objects exiting the field of view.
    if (prev_kpts_.size() < kMinFeatures) {
        DetectNewFeatures(frame_gray, prev_kpts_);
        prev_frame_gray_ = frame_gray;
        std::cout << "[Tracker] Feature pool replenished. Current size: " << prev_kpts_.size() << std::endl;
    }

    return true;
}

cv::Mat VisualOdometryTracker::GetCurrentR() const { return cur_R_.clone(); }
cv::Mat VisualOdometryTracker::GetCurrentT() const { return cur_t_.clone(); }
std::vector<cv::Point2f> VisualOdometryTracker::GetTrackedPoints() const { return prev_kpts_; }

}  // namespace tracker
}  // namespace deepvo