#pragma once
#include <cmath>
#include <algorithm>

// 1€ filter (Casiez et al.) — per-coordinate 1D.
class OneEuroFilter1D {
public:
    OneEuroFilter1D(double rate_hz, double min_cutoff = 1.0, double beta = 0.007, double d_cutoff = 1.0)
        : rate_(rate_hz), min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff), x_prev_(0.0), dx_prev_(0.0), first_(true) {}

    void set_rate(double hz) { rate_ = hz; }

    double operator()(double x) {
        if (first_) {
            first_ = false;
            x_prev_ = x;
            dx_prev_ = 0.0;
            return x;
        }
        double te = 1.0 / rate_;
        double dx = (x - x_prev_) * rate_;
        double a_d = alpha(d_cutoff_, rate_);
        dx_prev_ = a_d * dx + (1.0 - a_d) * dx_prev_;
        double cutoff = min_cutoff_ + beta_ * std::abs(dx_prev_);
        double a = alpha(cutoff, rate_);
        double xh = a * x + (1.0 - a) * x_prev_;
        x_prev_ = xh;
        return xh;
    }

private:
    static double alpha(double cutoff, double rate) {
        double te = 1.0 / rate;
        double tau = 1.0 / (2.0 * M_PI * cutoff);
        return 1.0 / (1.0 + tau / te);
    }

    double rate_;
    double min_cutoff_;
    double beta_;
    double d_cutoff_;
    double x_prev_;
    double dx_prev_;
    bool first_;
};
