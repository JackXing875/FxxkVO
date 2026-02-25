#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace deepvo {
namespace visualization {

class Visualizer2D {
 public:
    explicit Visualizer2D(const std::string& window_name);
    
    ~Visualizer2D();

    bool ShowFrame(cv::Mat& frame, 
                   const std::vector<cv::Point2f>& tracked_points, 
                   const cv::Mat& t);

 private:
    std::string window_name_;
};

} // namespace visualization
} // namespace deepvo