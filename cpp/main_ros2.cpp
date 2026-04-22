// ROS2 NCNN Hand Tracker Native Node
// Subscribes to /cam1/image_raw, performs SimCC Inference, and overlays 3D Hand PnP Angles

#ifndef HAS_NCNN
#define HAS_NCNN 0
#endif

#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#if HAS_NCNN
#include <net.h>
#endif

#include "OneEuroFilter.hpp"

// ROS 2 dependencies
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

static const int kSize = 160;
static const int kJoints = 10;
static const int kBins = 320;

// SLAM Calib (cam1) -> Pinhole Approximation for ds model
static const double FX = 148.390;
static const double FY = 148.675;
static const double CX = 151.987;
static const double CY = 115.147;

static void letterbox(const cv::Mat& src, cv::Mat& dst160, float& scale, float& pad_x, float& pad_y) {
    int w = src.cols, h = src.rows;
    scale = std::min(kSize / float(w), kSize / float(h));
    int nw = int(std::round(w * scale));
    int nh = int(std::round(h * scale));
    pad_x = (kSize - nw) * 0.5f;
    pad_y = (kSize - nh) * 0.5f;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    dst160 = cv::Mat(kSize, kSize, CV_8UC3, cv::Scalar(114, 114, 114));
    int x0 = int(std::round(pad_x));
    int y0 = int(std::round(pad_y));
    resized.copyTo(dst160(cv::Rect(x0, y0, nw, nh)));
}

#if HAS_NCNN
static void fill_input_nchw_imagenet(const cv::Mat& bgr160, ncnn::Mat& in) {
    in = ncnn::Mat::from_pixels(bgr160.data, ncnn::Mat::PIXEL_BGR2RGB, bgr160.cols, bgr160.rows);
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1.0f / (0.229f * 255.f), 1.0f / (0.224f * 255.f), 1.0f / (0.225f * 255.f)};
    in.substract_mean_normalize(mean_vals, norm_vals);
}

static void softmax_inplace(float* v, int n) {
    float m = v[0];
    for (int i = 1; i < n; ++i) m = std::max(m, v[i]);
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        v[i] = std::exp(v[i] - m);
        s += v[i];
    }
    for (int i = 0; i < n; ++i) v[i] /= (s + 1e-8f);
}

static void decode_simcc(const ncnn::Mat& outx, const ncnn::Mat& outy, float coords[kJoints][2]) {
    const float span = float(kSize);
    for (int j = 0; j < kJoints; ++j) {
        float px[kBins];
        float py[kBins];
        const float* rowx = outx.row(j);
        const float* rowy = outy.row(j);
        for (int b = 0; b < kBins; ++b) {
            px[b] = rowx[b];
            py[b] = rowy[b];
        }
        softmax_inplace(px, kBins);
        softmax_inplace(py, kBins);
        float cx = 0, cy = 0;
        for (int b = 0; b < kBins; ++b) {
            float t = b * (span / float(kBins));
            cx += px[b] * t;
            cy += py[b] * t;
        }
        coords[j][0] = cx;
        coords[j][1] = cy;
    }
}
#endif

// 3D Object Space for solvePnP
std::vector<cv::Point3f> getCanonicalObjectPoints() {
    float scale = 100.0f;
    return {
        {0.0f, 0.0f, 0.0f},
        {scale * 0.9f, 0.0f, 0.0f},
        {scale * 0.95f, scale * 0.35f, 0.0f},
        {scale * 0.75f, scale * 0.65f, 0.0f},
        {scale * 0.45f, scale * 0.85f, 0.0f}
    };
}


class HandSimCCNCNNNode : public rclcpp::Node {
public:
    HandSimCCNCNNNode(const std::string& param_file, const std::string& bin_file)
    : Node("hand_simcc_ncnn")
    #if HAS_NCNN
    , fx_(kJoints, OneEuroFilter1D(90.0, 1.0, 0.007, 1.0))
    , fy_(kJoints, OneEuroFilter1D(90.0, 1.0, 0.007, 1.0)) 
    #endif
    {
#if HAS_NCNN
        omp_set_num_threads(4);
        ncnn::Option opt;
        opt.num_threads = 4;
        opt.use_fp16_arithmetic = false;
        opt.use_fp16_storage = false;
        net_.opt = opt;

        if (net_.load_param(param_file.c_str()) != 0 || net_.load_model(bin_file.c_str()) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load NCNN model!");
            rclcpp::shutdown();
        }

        // OpenCV Camera Matrix
        cam_matrix_ = (cv::Mat_<double>(3, 3) << FX, 0, CX, 0, FY, CY, 0, 0, 1);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);

        // Subscribe to Stereo Right
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/cam1/image_raw", 10, std::bind(&HandSimCCNCNNNode::imageCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Hand tracker node listening on /cam1/image_raw");
#else
        RCLCPP_ERROR(this->get_logger(), "Built without NCNN Support!");
        rclcpp::shutdown();
#endif
    }

private:
#if HAS_NCNN
    ncnn::Net net_;
    std::vector<OneEuroFilter1D> fx_;
    std::vector<OneEuroFilter1D> fy_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    cv::Mat cam_matrix_;
    cv::Mat dist_coeffs_;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            // Stereo splitter outputs 'mono8', but net expects BGR 
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        cv::Mat bgr160;
        float sc, px, py;
        letterbox(frame, bgr160, sc, px, py);

        ncnn::Mat in;
        fill_input_nchw_imagenet(bgr160, in);

        auto t0 = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net_.create_extractor();
        ex.input("input", in);
        ncnn::Mat outx, outy;
        if (ex.extract("simcc_x", outx) != 0 || ex.extract("simcc_y", outy) != 0) {
            RCLCPP_ERROR(this->get_logger(), "Extraction failed. Blob names mismatch?");
            return;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        float coords[kJoints][2];
        decode_simcc(outx, outy, coords);

        std::vector<cv::Point2f> image_pts;

        for (int j = 0; j < kJoints; ++j) {
            float smoothed_x = fx_[j](coords[j][0]);
            float smoothed_y = fy_[j](coords[j][1]);
            
            float real_x = (smoothed_x - px) / sc;
            float real_y = (smoothed_y - py) / sc;
            
            // Push Wrist and first 4 MCPs for PnP Matrix
            if (j <= 4) {
                image_pts.push_back(cv::Point2f(real_x, real_y));
            }

            if (j == 0) {
                cv::circle(frame, cv::Point((int)real_x, (int)real_y), 6, cv::Scalar(0, 0, 255), -1);
            } else if (j <= 4) {
                cv::circle(frame, cv::Point((int)real_x, (int)real_y), 5, cv::Scalar(0, 255, 255), -1);
            } else {
                cv::circle(frame, cv::Point((int)real_x, (int)real_y), 4, cv::Scalar(0, 255, 0), -1);
            }
        }

        // PnP Solver
        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(getCanonicalObjectPoints(), image_pts, cam_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        
        double splay = 0, curl = 0;
        if (ok) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            double ex_x = R.at<double>(0,0), ex_y = R.at<double>(1,0); 
            double ez_x = R.at<double>(0,2), ez_z = R.at<double>(2,2);
            splay = std::atan2(ex_y, ex_x + 1e-8) * 180.0 / M_PI;
            curl = std::atan2(ez_z, ez_x + 1e-8) * 180.0 / M_PI;
        }

        cv::putText(frame, "Latency: " + std::to_string(ms).substr(0, 4) + "ms", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);
        cv::putText(frame, "Splay: " + std::to_string(splay).substr(0, 5), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
        cv::putText(frame, "Curl:  " + std::to_string(curl).substr(0, 5), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);

        cv::imshow("ROS2 Hand Tracker", frame);
        cv::waitKey(1);
    }
#endif
};

int main(int argc, char** argv) {
#if !HAS_NCNN
    std::cerr << "Built without NCNN (set HAS_NCNN=1 and link ncnn).\n";
    return 1;
#else
    rclcpp::init(argc, argv);
    std::string param = argc > 1 ? argv[1] : "../models/ncnn/hand_simcc.opt.param";
    std::string bin   = argc > 2 ? argv[2] : "../models/ncnn/hand_simcc.opt.bin";
    
    auto node = std::make_shared<HandSimCCNCNNNode>(param, bin);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
#endif
}
