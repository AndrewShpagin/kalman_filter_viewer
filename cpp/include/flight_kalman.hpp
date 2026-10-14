#pragma once

// C++17, header-only, no external dependencies or allocations on a valid update.
// Matches kalman_flight_viewer.py: horizontal NED position/velocity, with a
// cumulative BODY-X/Y bias fit outside the Kalman state. Body-Z bias is zero.
// Acceleration is already in NED, in m/s^2 (not raw body specific force).
// dt is in seconds. Acceleration/attitude at row i describe interval (i-1, i].

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <stdexcept>

namespace flight_kalman {

using Vec2 = std::array<double, 2>;
using Rotation = std::array<std::array<double, 3>, 3>;

struct Quaternion {
    double x = 0, y = 0, z = 0, w = 1; // x,y,z,w, active right-handed rotation
};

enum class QuaternionDirection { BodyToNed, NedToBody };

inline Quaternion normalized(Quaternion q) {
    const double n = std::hypot(std::hypot(q.x, q.y), std::hypot(q.z, q.w));
    if (!std::isfinite(n) || n < 1e-8)
        throw std::invalid_argument("Quaternion has a zero/non-finite norm");
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}

inline Rotation rotationMatrix(Quaternion input,
                               QuaternionDirection direction = QuaternionDirection::BodyToNed) {
    const auto q = normalized(input);
    const auto x = q.x, y = q.y, z = q.z, w = q.w;
    Rotation r{{
        {{1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)}},
        {{2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)}},
        {{2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)}}
    }};
    if (direction == QuaternionDirection::NedToBody)
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j)
                std::swap(r[i][j], r[j][i]);
    return r;
}

struct NoiseParameters {
    double acceleration_sigma = 1.0;       // independent per-sample stddev, m/s^2
    double min_position_sigma = 0.001;     // m, strictly positive
    double height_error_fraction = 0.1;    // 0.1 = 10% of |height|
    double initial_velocity_sigma = 1.0;   // m/s; used only at initialization/reset
};

struct Parameters {
    NoiseParameters noise;
    double coordinate_scale = 1.0;
    QuaternionDirection quaternion_direction = QuaternionDirection::BodyToNed;
};

struct Measurement {
    Vec2 acceleration_ned{};
    Quaternion quaternion{};
    Vec2 visual_position{};
    double height = 0;                     // above ground, m
    double dt = 0;                         // >0 after initialization; first row may use 0
    std::optional<double> position_sigma;  // m; optional per-row override of height formula
};

struct AxisCovariance { double xx = 0, xv = 0, vv = 0; };
struct AxisGain { double position = 0, velocity = 0; };

struct Estimate {
    double time_s = 0, dt_s = 0, height = 0, sigma_position = 0, coordinate_scale = 1;
    Vec2 raw_position{}, measured_position{}, position{}, velocity{};
    Vec2 raw_acceleration{}, corrected_acceleration{}, integral_raw{};
    Vec2 bias_body{}, bias_body_used{}, bias_ned{}, bias_ned_used{}, bias_rhs{};
    std::array<double, 3> bias_normal{}; // xx,xy,yy of accumulated normal matrix
    std::array<AxisCovariance, 2> covariance{};
    std::array<AxisGain, 2> gain{};
    Quaternion quaternion{}; // normalized INPUT, even for NedToBody convention
};

class FlightKalmanFilter {
public:
    explicit FlightKalmanFilter(Parameters parameters = {}) : parameters_(parameters) {
        validateNoise(parameters.noise);
        positive(parameters.coordinate_scale, "Coordinate scale must be finite and >0");
    }

    const Parameters& parameters() const noexcept { return parameters_; }
    const Estimate& state() const noexcept { return state_; }
    bool initialized() const noexcept { return initialized_; }

    // New noise applies on the NEXT update. State, covariance and bias history
    // are retained. initial_velocity_sigma takes effect on the next reset/init.
    void setNoiseParameters(const NoiseParameters& noise) {
        validateNoise(noise);
        parameters_.noise = noise;
    }

    // Scales future measurements; does not retroactively rescale stored state.
    // To change the scale of an entire flight, reset and replay its measurements.
    void setCoordinateScale(double scale) {
        positive(scale, "Coordinate scale must be finite and >0");
        parameters_.coordinate_scale = scale;
    }

    void reset(Vec2 initial_velocity = {}) {
        finite(initial_velocity, "Initial velocity must be finite");
        initial_velocity_ = initial_velocity;
        state_ = Estimate{};
        initialized_ = false;
    }

    // First call initializes position/covariance and returns t=0, dt=0 without
    // integrating acceleration. Later calls require finite dt>0, with no fixed
    // sample period stored in this class. Invalid calls leave all state intact.
    const Estimate& update(const Measurement& m) {
        finite(m.acceleration_ned, "Acceleration must be finite");
        finite(m.visual_position, "Visual position must be finite");
        if (!std::isfinite(m.height) || !std::isfinite(m.dt) || m.dt < 0 ||
            (initialized_ && m.dt == 0))
            throw std::invalid_argument("Height must be finite; dt must be >0 after initialization");
        if (m.position_sigma)
            positive(*m.position_sigma, "Position sigma override must be finite and >0");
        const auto r = rotationMatrix(m.quaternion, parameters_.quaternion_direction);
        auto next = *this; // fixed-size copy gives strong exception safety
        next.advance(m, r);
        next.validateState();
        *this = next;
        return state_;
    }

    const Estimate& update(Vec2 acceleration_ned, Quaternion quaternion,
                           Vec2 visual_position, double height, double dt) {
        return update(Measurement{acceleration_ned, quaternion, visual_position, height, dt, {}});
    }

private:
    Parameters parameters_;
    Estimate state_;
    Vec2 initial_velocity_{};
    bool initialized_ = false;

    static void positive(double v, const char* message) {
        if (!std::isfinite(v) || v <= 0) throw std::invalid_argument(message);
    }
    static void finite(Vec2 v, const char* message) {
        if (!std::isfinite(v[0]) || !std::isfinite(v[1])) throw std::invalid_argument(message);
    }
    static void validateNoise(const NoiseParameters& n) {
        positive(n.min_position_sigma, "Minimum position sigma must be finite and >0");
        for (double v : {n.acceleration_sigma, n.height_error_fraction, n.initial_velocity_sigma})
            if (!std::isfinite(v) || v < 0)
                throw std::invalid_argument("Noise values must be finite and nonnegative");
        if (!std::isfinite(n.acceleration_sigma * n.acceleration_sigma) ||
            !std::isfinite(n.initial_velocity_sigma * n.initial_velocity_sigma))
            throw std::invalid_argument("Noise variance overflow");
    }
    static Vec2 project(const Rotation& r, Vec2 body) {
        return {r[0][0]*body[0] + r[0][1]*body[1],
                r[1][0]*body[0] + r[1][1]*body[1]};
    }

    void advance(const Measurement& m, const Rotation& r) {
        auto& s = state_;
        s.height = m.height;
        s.sigma_position = m.position_sigma.value_or(
            std::max(parameters_.noise.min_position_sigma,
                     parameters_.noise.height_error_fraction * std::abs(m.height)));
        const double variance = s.sigma_position * s.sigma_position;
        if (!std::isfinite(variance) || variance <= 0)
            throw std::invalid_argument("Position variance underflow/overflow");
        s.coordinate_scale = parameters_.coordinate_scale;
        s.raw_position = m.visual_position;
        for (int j = 0; j < 2; ++j)
            s.measured_position[j] = m.visual_position[j] * s.coordinate_scale;
        s.raw_acceleration = m.acceleration_ned;
        s.quaternion = normalized(m.quaternion);
        if (!initialized_) {
            s.position = s.measured_position;
            s.velocity = initial_velocity_;
            s.corrected_acceleration = m.acceleration_ned;
            const double vv = parameters_.noise.initial_velocity_sigma * parameters_.noise.initial_velocity_sigma;
            for (auto& p : s.covariance) p = {variance, 0, vv};
            initialized_ = true;
            return;
        }
        const double dt = m.dt, dt2 = dt*dt;
        const Vec2 previous_velocity = s.velocity;
        s.time_s += dt;
        s.dt_s = dt;
        s.bias_body_used = s.bias_body;
        s.bias_ned_used = project(r, s.bias_body_used);
        for (int j = 0; j < 2; ++j) {
            const double a = m.acceleration_ned[j] - s.bias_ned_used[j];
            s.corrected_acceleration[j] = a;
            s.integral_raw[j] += m.acceleration_ned[j] * dt;
            s.position[j] += s.velocity[j]*dt + 0.5*a*dt2;
            s.velocity[j] += a*dt;
            const auto old = s.covariance[j];
            const double q = parameters_.noise.acceleration_sigma * parameters_.noise.acceleration_sigma;
            const double xx = old.xx + 2*dt*old.xv + dt2*old.vv + 0.25*dt2*dt2*q;
            const double xv = old.xv + dt*old.vv + 0.5*dt2*dt*q;
            const double vv = old.vv + dt2*q;
            const double denominator = xx + variance;
            const double kx = xx / denominator, kv = xv / denominator;
            const double innovation = s.measured_position[j] - s.position[j];
            s.position[j] += kx * innovation;
            s.velocity[j] += kv * innovation;
            const double h = variance / denominator;
            s.covariance[j] = {h*h*xx + kx*kx*variance,
                               h*(xv-kv*xx) + kx*kv*variance,
                               vv - 2*kv*xv + kv*kv*(xx+variance)};
            s.gain[j] = {kx, kv};
        }
        const Vec2 residual{m.acceleration_ned[0]*dt - (s.velocity[0]-previous_velocity[0]),
                            m.acceleration_ned[1]*dt - (s.velocity[1]-previous_velocity[1])};
        const double a = r[0][0], b = r[0][1], c = r[1][0], d = r[1][1];
        s.bias_normal[0] += dt*(a*a+c*c);
        s.bias_normal[1] += dt*(a*b+c*d);
        s.bias_normal[2] += dt*(b*b+d*d);
        s.bias_rhs[0] += a*residual[0] + c*residual[1];
        s.bias_rhs[1] += b*residual[0] + d*residual[1];
        fitBodyBias();
        s.bias_ned = project(r, s.bias_body);
    }

    void fitBodyBias() {
        // Analytic symmetric 2x2 eigendecomposition; same observability cutoff
        // as numpy.linalg.eigh in the Python implementation. No unstable inverse
        // near a 90-degree tilt: keep the old bias in unobservable directions.
        auto& s = state_;
        const double a = s.bias_normal[0], b = s.bias_normal[1], d = s.bias_normal[2];
        const double center = 0.5*(a+d), radius = std::hypot(0.5*(a-d), b);
        const double hi = center + radius, lo = center - radius;
        const double angle = 0.5*std::atan2(2*b, a-d);
        const double c = std::cos(angle), sn = std::sin(angle);
        const double threshold = std::max(hi*1e-8, 1e-12);
        double high_component = c*s.bias_body[0] + sn*s.bias_body[1];
        double low_component = -sn*s.bias_body[0] + c*s.bias_body[1];
        if (hi > threshold) high_component = (c*s.bias_rhs[0]+sn*s.bias_rhs[1])/hi;
        if (lo > threshold) low_component = (-sn*s.bias_rhs[0]+c*s.bias_rhs[1])/lo;
        s.bias_body = {c*high_component-sn*low_component, sn*high_component+c*low_component};
    }

    void validateState() const {
        const auto& s = state_;
        auto check = [](double v) {
            if (!std::isfinite(v)) throw std::overflow_error("Non-finite filter state; check input/noise/dt");
        };
        for (double v : {s.time_s, s.dt_s, s.sigma_position}) check(v);
        for (const auto& v : {s.measured_position, s.position, s.velocity, s.corrected_acceleration,
                              s.integral_raw, s.bias_body, s.bias_body_used, s.bias_ned,
                              s.bias_ned_used, s.bias_rhs})
            for (double value : v) check(value);
        for (double v : s.bias_normal) check(v);
        for (const auto& p : s.covariance) { check(p.xx); check(p.xv); check(p.vv); }
        for (const auto& k : s.gain) { check(k.position); check(k.velocity); }
    }
};

} // namespace flight_kalman
