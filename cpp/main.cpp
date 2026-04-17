// Phase 4: OpenCV V4L2 capture + letterbox, NCNN inference (4 threads), 1€ filter, latency log.
// Usage: ./hand_ncnn_demo [camera_id] [model.param] [model.bin] [max_frames]

#ifndef HAS_NCNN
#define HAS_NCNN 0
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <omp.h>
#include <opencv2/opencv.hpp>

#if HAS_NCNN
#include <net.h>
#endif

#include "OneEuroFilter.hpp"

static const int kSize = 160;
static const int kJoints = 10;
static const int kBins = 320;

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

struct FrameBuf {
    cv::Mat bgr160;
};
#endif

int main(int argc, char** argv) {
#if !HAS_NCNN
    (void)argc;
    (void)argv;
    std::cerr << "Built without NCNN (set HAS_NCNN=1 and link ncnn).\n";
    return 2;
#else
    int cam = argc > 1 ? std::atoi(argv[1]) : 0;
    std::string param = argc > 2 ? argv[2] : "../models/ncnn/hand_simcc.opt.param";
    std::string binf = argc > 3 ? argv[3] : "../models/ncnn/hand_simcc.opt.bin";
    int max_frames = (argc > 4 ? std::atoi(argv[4]) : 2000);

    omp_set_num_threads(4);
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_arithmetic = false;
    opt.use_fp16_storage = false;
    ncnn::Net net;
    net.opt = opt;
    if (net.load_param(param.c_str()) != 0 || net.load_model(binf.c_str()) != 0) {
        std::cerr << "Failed to load NCNN model: " << param << " / " << binf << "\n";
        return 1;
    }

    std::vector<OneEuroFilter1D> fx(kJoints, OneEuroFilter1D(90.0, 1.0, 0.007, 1.0));
    std::vector<OneEuroFilter1D> fy(kJoints, OneEuroFilter1D(90.0, 1.0, 0.007, 1.0));

    cv::VideoCapture cap(cam, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "V4L2 open failed, trying default\n";
        cap.open(cam);
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 90);

    int frames_done = 0;
    while (frames_done < max_frames) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        
        cv::Mat bgr160;
        float sc, px, py;
        letterbox(frame, bgr160, sc, px, py);

        ncnn::Mat in;
        fill_input_nchw_imagenet(bgr160, in);
        
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input("input", in);
        ncnn::Mat outx, outy;
        if (ex.extract("simcc_x", outx) != 0 || ex.extract("simcc_y", outy) != 0) {
            std::cerr << "extract failed; check blob names\n";
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "inference_time_ms=" << ms << std::endl;

        float coords[kJoints][2];
        decode_simcc(outx, outy, coords);
        
        // Draw the 10 joints on the original frame
        for (int j = 0; j < kJoints; ++j) {
            float smoothed_x = fx[j](coords[j][0]);
            float smoothed_y = fy[j](coords[j][1]);
            
            // Un-letterbox back to original `frame` pixel space
            int real_x = int((smoothed_x - px) / sc);
            int real_y = int((smoothed_y - py) / sc);
            
            std::cout << "[" << real_x << "," << real_y << "] ";

            if (j == 0) {
                // Wrist = Red
                cv::circle(frame, cv::Point(real_x, real_y), 6, cv::Scalar(0, 0, 255), -1);
            } else if (j <= 4) {
                // MCPs = Yellow
                cv::circle(frame, cv::Point(real_x, real_y), 5, cv::Scalar(0, 255, 255), -1);
            } else {
                // Tips = Green
                cv::circle(frame, cv::Point(real_x, real_y), 4, cv::Scalar(0, 255, 0), -1);
            }
        }

        cv::imshow("NCNN Hand Tracking", frame);
        if (cv::waitKey(1) == 27) { // Wait for 'ESC' key to exit
            break;
        }
        
        frames_done++;
    }

    return 0;
#endif
}
