#pragma once

// Standalone C++17 header; no Eigen, Python or Kalman dependencies.
// Window-normalized horizontal NED velocity, matching velocity_flight_viewer.py.
// Acceleration is gravity-compensated NED acceleration BEFORE our bias removal.
// Quaternion: x,y,z,w. Body-Z bias is assumed zero; full attitude is used.
// Each update supplies position at the interval END and acceleration/attitude
// held over that interval. The first update initializes; its dt is not integrated.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <initializer_list>
#include <stdexcept>

namespace flight_velocity {

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;
using Rotation = std::array<std::array<double, 3>, 3>;
struct Quaternion { double x = 0, y = 0, z = 0, w = 1; };
enum class QuaternionDirection { BodyToNed, NedToBody };

inline Quaternion normalized(Quaternion q) {
    const double n = std::hypot(std::hypot(q.x, q.y), std::hypot(q.z, q.w));
    if (!std::isfinite(n) || n < 1e-8)
        throw std::invalid_argument("Quaternion has a zero/non-finite norm");
    return {q.x/n, q.y/n, q.z/n, q.w/n};
}

inline Rotation rotationMatrix(Quaternion input,
                               QuaternionDirection direction = QuaternionDirection::BodyToNed) {
    const auto q = normalized(input);
    const auto x = q.x, y = q.y, z = q.z, w = q.w;
    Rotation r{{
        {{1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)}},
        {{2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)}},
        {{2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}}
    }};
    if (direction == QuaternionDirection::NedToBody)
        for (int i = 0; i < 3; ++i)
            for (int j = i+1; j < 3; ++j) std::swap(r[i][j], r[j][i]);
    return r;
}

struct Parameters {
    double fade_in = 6, flat = 6, fade_out = 6; // seconds, oldest -> newest
    double lead = 1;                          // forward_position = scaled vision + lead*v
    double coordinate_scale = 1;
    double bias_warmup = 6;                   // minimum information time, seconds
    double bias_memory = 60;                  // exponential memory, seconds; 0 = whole flight
    Vec2 initial_velocity{};                 // NED, m/s
    Vec2 initial_bias_body{};                // BODY X/Y, m/s^2
    bool learn_bias = true;
    QuaternionDirection quaternion_direction = QuaternionDirection::BodyToNed;
    double windowSeconds() const noexcept { return fade_in + flat + fade_out; }
};

struct Measurement {
    Vec2 acceleration_ned{};
    Quaternion quaternion{};
    Vec2 visual_position{};
    double dt = 0;                           // seconds, >0 after initialization
};

struct Estimate {
    double time_s = 0, dt_s = 0, window_span_s = 0;
    double weight_sum = 0, weighted_duration_s = 0;
    double bias_prior_information_s = 0;     // decaying startup floor
    bool window_complete = false;
    Vec2 raw_position{}, measured_position{}, forward_position{}, velocity{};
    Vec2 mean_visual_velocity{}, mean_integrated_velocity{};
    Vec2 window_integral{}, normalization_offset{};
    Vec2 raw_acceleration{}, corrected_acceleration{}, integral_raw{};
    Vec2 bias_body{}, bias_body_used{}, bias_ned{}, bias_ned_used{}, bias_rhs{};
    std::array<double, 3> bias_normal{};       // xx,xy,yy; information, NOT covariance
    Mat2 bias_response{};                    // C: velocity response to BODY bias
    Quaternion quaternion{};                // normalized INPUT convention
};

class FlightVelocityEstimator {
public:
    explicit FlightVelocityEstimator(Parameters parameters = {}) { reset(parameters); }

    const Parameters& parameters() const noexcept { return parameters_; }
    const Estimate& state() const noexcept { return state_; }
    bool initialized() const noexcept { return initialized_; }
    std::size_t historySize() const noexcept { return history_.size(); }

    // Configuration changes deliberately start a new history. Replaying is
    // required to compare a whole flight under different parameters.
    void reset(const Parameters& parameters) {
        validate(parameters);
        parameters_ = parameters;
        reset();
    }
    void reset() noexcept {
        history_.clear();
        time_ = 0;
        initialized_ = false;
        state_ = Estimate{};
        state_.velocity = parameters_.initial_velocity;
        state_.bias_body = state_.bias_body_used = parameters_.initial_bias_body;
        state_.bias_prior_information_s = parameters_.bias_warmup;
    }

    Estimate update(Vec2 raw_acceleration_ned, Quaternion quaternion,
                    Vec2 visual_position, double dt) {
        return update(Measurement{raw_acceleration_ned, quaternion, visual_position, dt});
    }

    // Invalid input / numeric overflow throws without changing state or history.
    // O(samples in window) time and memory; deque may allocate. Not thread-safe.
    Estimate update(const Measurement& input) {
        finite(input.acceleration_ned); finite(input.visual_position);
        if (!std::isfinite(input.dt) || input.dt < 0 || (initialized_ && input.dt == 0))
            throw std::invalid_argument("dt must be finite and >0 after initialization");
        Estimate next = state_;
        next.quaternion = normalized(input.quaternion);
        const auto r = rotationMatrix(next.quaternion, parameters_.quaternion_direction);
        const Mat2 m{{{{r[0][0], r[0][1]}}, {{r[1][0], r[1][1]}}}};
        next.raw_position = input.visual_position;
        next.raw_acceleration = input.acceleration_ned;
        next.bias_body_used = state_.bias_body;
        next.bias_ned_used = multiply(m, next.bias_body_used);
        for (int j = 0; j < 2; ++j) {
            next.measured_position[j] = input.visual_position[j]*parameters_.coordinate_scale;
            next.corrected_acceleration[j] = input.acceleration_ned[j]-next.bias_ned_used[j];
        }
        if (!initialized_) {
            next.mean_visual_velocity = next.normalization_offset = next.velocity;
            finish(next, m);
            state_ = next;
            initialized_ = true;
            return state_;
        }

        const long double time = time_ + input.dt;
        if (!std::isfinite(time) || time <= time_)
            throw std::invalid_argument("Elapsed time is outside representable range");
        next.time_s = static_cast<double>(time);
        next.dt_s = input.dt;
        const long double left = std::max(0.0L, time-parameters_.windowSeconds());
        next.window_span_s = static_cast<double>(time-left);
        next.window_complete = time + 1e-12L*parameters_.windowSeconds() >= parameters_.windowSeconds();
        Vec2 displacement{};
        for (int j = 0; j < 2; ++j) {
            displacement[j] = next.measured_position[j]-state_.measured_position[j];
            next.integral_raw[j] += input.acceleration_ned[j]*input.dt;
        }
        finite(displacement);
        history_.push_back({time, input.dt, input.acceleration_ned, displacement, m});
        try {
            average(next, left, time);
            if (parameters_.learn_bias) learn(next, m);
            finish(next, m);
        } catch (...) {
            history_.pop_back();
            throw;
        }
        while (!history_.empty() && history_.front().end <= left) history_.pop_front();
        time_ = time;
        state_ = next;
        return state_;
    }

private:
    struct Interval {
        long double end;
        double dt;
        Vec2 acceleration, displacement;
        Mat2 projection;
    };
    Parameters parameters_{};
    Estimate state_{};
    std::deque<Interval> history_;
    long double time_ = 0;
    bool initialized_ = false;

    static Vec2 multiply(const Mat2& a, const Vec2& b) {
        return {a[0][0]*b[0]+a[0][1]*b[1], a[1][0]*b[0]+a[1][1]*b[1]};
    }
    static void finite(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite input or numerical overflow");
    }
    static void finite(const Vec2& v) { for (double x : v) finite(x); }
    static void validate(const Parameters& p) {
        for (double x : {p.fade_in, p.flat, p.fade_out, p.lead, p.bias_warmup, p.bias_memory}) {
            finite(x);
            if (x < 0) throw std::invalid_argument("Durations must be nonnegative");
        }
        finite(p.windowSeconds()); finite(p.coordinate_scale);
        if (p.windowSeconds() <= 0 || p.coordinate_scale <= 0)
            throw std::invalid_argument("Window and coordinate scale must be positive");
        finite(p.initial_velocity); finite(p.initial_bias_body);
        if (p.quaternion_direction != QuaternionDirection::BodyToNed &&
            p.quaternion_direction != QuaternionDirection::NedToBody)
            throw std::invalid_argument("Invalid quaternion direction");
    }
    double weight(double age, double span) const {
        double w = 1;
        if (parameters_.fade_in > 0) w = std::min(w, (span-age)/parameters_.fade_in);
        if (parameters_.fade_out > 0) w = std::min(w, age/parameters_.fade_out);
        return std::max(0.0, w);
    }

    void average(Estimate& s, long double left, long double time) const {
        Vec2 integral{}, visual_sum{}, inertial_sum{};
        Mat2 rotation_integral{}, rotation_sum{};
        s.weight_sum = s.weighted_duration_s = 0;
        for (const auto& interval : history_) {
            if (interval.end <= left) continue;
            const double h = static_cast<double>(std::min(static_cast<long double>(interval.dt), interval.end-left));
            const double age = static_cast<double>(time-interval.end) + h/2;
            const double w = weight(age, s.window_span_s);
            const double mass = w*h;
            const double fraction = h/interval.dt; // <=1; no tiny-dt visual derivative
            const auto bias_ned = multiply(interval.projection, s.bias_body_used);
            for (int j = 0; j < 2; ++j) {
                const double delta = (interval.acceleration[j]-bias_ned[j])*h;
                inertial_sum[j] += mass*(integral[j]+delta/2);
                integral[j] += delta;             // acceleration is NEVER faded
                visual_sum[j] += w*(interval.displacement[j]*fraction);
                for (int k = 0; k < 2; ++k) {
                    const double dr = interval.projection[j][k]*h;
                    rotation_sum[j][k] += mass*(rotation_integral[j][k]+dr/2);
                    rotation_integral[j][k] += dr;
                }
            }
            s.weight_sum += w;
            s.weighted_duration_s += mass;
        }
        if (!(s.weighted_duration_s > 0))
            throw std::invalid_argument("Window has no positive weighted duration");
        for (int j = 0; j < 2; ++j) {
            s.window_integral[j] = integral[j];
            s.mean_visual_velocity[j] = visual_sum[j]/s.weighted_duration_s;
            s.mean_integrated_velocity[j] = inertial_sum[j]/s.weighted_duration_s;
            s.normalization_offset[j] = s.mean_visual_velocity[j]-s.mean_integrated_velocity[j];
            s.velocity[j] = integral[j]+s.normalization_offset[j];
            for (int k = 0; k < 2; ++k)
                s.bias_response[j][k] = rotation_integral[j][k]-rotation_sum[j][k]/s.weighted_duration_s;
        }
    }

    void learn(Estimate& s, const Mat2& m) const {
        Mat2 h{};
        const auto response_now = multiply(s.bias_response, s.bias_body_used);
        const auto response_prev = multiply(state_.bias_response, state_.bias_body_used);
        Vec2 residual{};
        for (int j = 0; j < 2; ++j) {
            residual[j] = s.raw_acceleration[j]*s.dt_s-(s.velocity[j]-state_.velocity[j])
                          -response_now[j]+response_prev[j];
            for (int k = 0; k < 2; ++k)
                h[j][k] = m[j][k]-(s.bias_response[j][k]-state_.bias_response[j][k])/s.dt_s;
        }
        double information_dt = s.dt_s, residual_weight = 1;
        if (parameters_.bias_memory > 0) {
            const double ratio = s.dt_s/parameters_.bias_memory;
            const double decay = std::exp(-ratio);
            information_dt = ratio == 0 ? s.dt_s : -parameters_.bias_memory*std::expm1(-ratio);
            residual_weight = information_dt/s.dt_s;
            for (double& x : s.bias_normal) x *= decay;
            for (double& x : s.bias_rhs) x *= decay;
            s.bias_prior_information_s *= decay;
        }
        for (int j = 0; j < 2; ++j) {
            s.bias_normal[0] += information_dt*h[j][0]*h[j][0];
            s.bias_normal[1] += information_dt*h[j][0]*h[j][1];
            s.bias_normal[2] += information_dt*h[j][1]*h[j][1];
            for (int k = 0; k < 2; ++k) s.bias_rhs[k] += residual_weight*h[j][k]*residual[j];
        }
        for (double x : s.bias_normal) finite(x);
        finite(s.bias_rhs);
        const auto a = s.bias_normal[0], b = s.bias_normal[1], d = s.bias_normal[2];
        const double center = a/2+d/2, radius = std::hypot(a/2-d/2, b);
        const double high = std::max(0.0, center+radius), low = std::max(0.0, center-radius);
        const double angle = 0.5*std::atan2(2*b, a-d), c = std::cos(angle), sn = std::sin(angle);
        auto project = [&](Vec2 v) -> Vec2 { return {c*v[0]+sn*v[1], -sn*v[0]+c*v[1]}; };
        Vec2 components{};
        if (parameters_.bias_warmup > 0) {
            const auto b0 = parameters_.initial_bias_body;
            const auto prior = project(b0);
            const auto error = project({s.bias_rhs[0]-a*b0[0]-b*b0[1], s.bias_rhs[1]-b*b0[0]-d*b0[1]});
            const double floor = std::max(s.bias_prior_information_s, std::max(high*1e-8, 1e-12));
            components = {prior[0]+error[0]/std::max(high, floor),
                          prior[1]+error[1]/std::max(low, floor)};
        } else {
            const double threshold = std::max(high*1e-8, 1e-12);
            const auto rhs = project(s.bias_rhs);
            components = project(s.bias_body);
            if (high > threshold) components[0] = rhs[0]/high;
            if (low > threshold) components[1] = rhs[1]/low;
        }
        s.bias_body = {c*components[0]-sn*components[1], sn*components[0]+c*components[1]};
    }

    void finish(Estimate& s, const Mat2& m) const {
        s.bias_ned = multiply(m, s.bias_body);
        for (int j = 0; j < 2; ++j)
            s.forward_position[j] = s.measured_position[j]+parameters_.lead*s.velocity[j];
        for (double x : {s.time_s, s.dt_s, s.window_span_s, s.weight_sum, s.weighted_duration_s, s.bias_prior_information_s}) finite(x);
        for (const auto& v : {s.raw_position, s.measured_position, s.forward_position, s.velocity,
                             s.mean_visual_velocity, s.mean_integrated_velocity, s.window_integral,
                             s.normalization_offset, s.raw_acceleration, s.corrected_acceleration,
                             s.integral_raw, s.bias_body, s.bias_body_used, s.bias_ned, s.bias_ned_used, s.bias_rhs}) finite(v);
        for (const auto& row : s.bias_response) finite(row);
        for (double x : s.bias_normal) finite(x);
    }
};

} // namespace flight_velocity
