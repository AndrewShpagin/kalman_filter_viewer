#include "flight_velocity.hpp"
#include <iostream>
#include <limits>
#include <string>

namespace fv = flight_velocity;
double velocityHeaderOdr();

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance = 1e-9) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance)
        throw std::runtime_error("Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
template<class F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, "Invalid input accepted");
}
fv::Quaternion attitude(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll/2), sr = std::sin(roll/2);
    const double cp = std::cos(pitch/2), sp = std::sin(pitch/2);
    const double cy = std::cos(yaw/2), sy = std::sin(yaw/2);
    return {sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy};
}

void weightedAndClipped() {
    fv::Parameters p;
    p.fade_in = p.fade_out = 2; p.flat = 6; p.learn_bias = false;
    fv::FlightVelocityEstimator f(p);
    f.update({}, {}, {}, 0);
    f.update({2, 0}, {}, {1, 0}, 1);
    auto s = f.update({4, 0}, {}, {16, 0}, 3);
    // interval weights .25, .75; weighted durations .25, 2.25
    close(s.weighted_duration_s, 2.5);
    close(s.mean_visual_velocity[0], 4.6);
    close(s.mean_integrated_velocity[0], 7.3);
    close(s.window_integral[0], 14);
    close(s.velocity[0], 11.3);
    close(s.forward_position[0], 27.3);

    p.fade_in = p.fade_out = 0; p.flat = 2;
    f.reset(p);
    f.update({}, {}, {}, 0);
    f.update({2, 0}, {}, {1, 0}, 1);
    s = f.update({4, 0}, {}, {16, 0}, 3);
    close(s.window_span_s, 2);
    close(s.window_integral[0], 8);
    close(s.mean_visual_velocity[0], 5);  // 2/3 of last displacement, over 2s
    close(s.velocity[0], 9);
    require(s.window_complete && f.historySize() == 1, "Expired interval retained");
}

void analyticMotion() {
    fv::Parameters p;
    p.fade_in = 2; p.flat = 30; p.fade_out = 3;
    p.initial_velocity = {1.1, -.7}; p.learn_bias = false;
    p.initial_bias_body = {.2, -.1}; p.coordinate_scale = 1.2;
    fv::FlightVelocityEstimator f(p);
    fv::Vec2 pos{2, -3}, v = p.initial_velocity;
    const fv::Vec2 a{.3, -.4};
    auto s = f.update({}, {}, {pos[0]/1.2, pos[1]/1.2}, .5);
    close(s.time_s, 0); close(s.dt_s, 0); close(s.velocity[0], v[0]);
    const double intervals[]{.017, .053, .14, 1e-7};
    for (int i = 0; i < 400; ++i) {
        const double dt = intervals[i%4];
        for (int j = 0; j < 2; ++j) {
            pos[j] += v[j]*dt+.5*a[j]*dt*dt;
            v[j] += a[j]*dt;
        }
        s = f.update({a[0]+.2, a[1]-.1}, {}, {pos[0]/1.2, pos[1]/1.2}, dt);
        close(s.velocity[0], v[0]); close(s.velocity[1], v[1]);
        close(s.bias_body[0], .2);
    }
    f.reset();
    require(!f.initialized() && f.historySize() == 0, "Reset did not clear history");
    close(f.state().bias_body[0], .2);
}

void biasAndRotation() {
    fv::Parameters p;
    fv::FlightVelocityEstimator f(p);
    auto s = f.update({.3, -.2}, {}, {}, 0);
    close(s.bias_body[0], 0);
    s = f.update({.3, -.2}, {}, {}, .05);
    close(s.bias_body[0], .3*(-60*std::expm1(-.05/60))/4/(6*std::exp(-.05/60)));
    close(s.bias_body_used[0], 0); // newly fitted bias only affects the next update
    const double initial = s.bias_body[0];
    s = f.update({.3, -.2}, {}, {}, .05);
    close(s.bias_body_used[0], initial);

    p.initial_bias_body = {-.15, .08};
    f.reset(p);
    p.quaternion_direction = fv::QuaternionDirection::NedToBody;
    fv::FlightVelocityEstimator inverse(p);
    double time = 0;
    for (int i = 0; i <= 4000; ++i) {
        const double dt = i == 0 ? 0 : (i%2 ? .043 : .057);
        time += dt;
        const auto q = attitude(.25*std::sin(.4*time), .2*std::cos(.3*time), .37*time);
        const auto r = fv::rotationMatrix(q);
        const fv::Vec2 raw{r[0][0]*.3-r[0][1]*.2, r[1][0]*.3-r[1][1]*.2};
        s = f.update(raw, q, {}, dt);
        auto inv = inverse.update(raw, {-q.x*1.7, -q.y*1.7, -q.z*1.7, q.w*1.7}, {}, dt);
        for (int j = 0; j < 2; ++j) {
            close(s.velocity[j], inv.velocity[j], 1e-8);
            close(s.bias_body[j], inv.bias_body[j], 1e-8);
        }
    }
    close(s.bias_body[0], .3, 1e-8); close(s.bias_body[1], -.2, 1e-8);
    close(s.velocity[0], 0, 1e-8); close(s.velocity[1], 0, 1e-8);
    require(f.historySize() <= 362, "History grows beyond window");

    for (double warmup : {0.0, 6.0}) {
        p = fv::Parameters{}; p.bias_warmup = warmup; p.initial_bias_body = {.7, 0};
        f.reset(p);
        const auto q = attitude(0, std::acos(-1.0)/2, 0); // body X unobservable in horizontal plane
        for (int i = 0; i < 1000; ++i) s = f.update({0, -.2}, q, {}, i ? .05 : 0);
        close(s.bias_body[0], .7, 1e-8); close(s.bias_body[1], -.2, 1e-8);
    }
}

void invalidAndRollback() {
    fv::FlightVelocityEstimator f;
    f.update({}, {}, {}, 0);
    f.update({.1, .2}, {}, {.03, -.04}, .05);
    const auto snapshot = f.state();
    const auto size = f.historySize();
    for (double dt : {0., -1., std::numeric_limits<double>::quiet_NaN()})
        rejects([&] { f.update({}, {}, {}, dt); });
    rejects([&] { f.update({}, {0, 0, 0, 0}, {}, .1); });
    rejects([&] { f.update({1e308, 0}, {}, {}, 10); }); // overflow AFTER history insertion
    require(f.historySize() == size, "Rejected update changed history");
    close(f.state().time_s, snapshot.time_s);
    auto copy = f;
    const auto a = f.update({.2, .3}, {}, {.05, -.06}, .07);
    const auto b = copy.update({.2, .3}, {}, {.05, -.06}, .07);
    close(a.velocity[0], b.velocity[0]); close(a.bias_body[1], b.bias_body[1]);
    auto p = f.parameters(); p.fade_out = -1;
    rejects([&] { f.reset(p); });
    close(f.state().time_s, a.time_s);
}

void changingBias() {
    fv::Parameters p;
    fv::FlightVelocityEstimator recent(p);
    p.bias_memory = 0;
    fv::FlightVelocityEstimator cumulative(p);
    fv::Estimate a, b;
    for (int i = 0; i < 8000; ++i) {
        const double time = i*.05;
        const fv::Vec2 bias = time < 160 ? fv::Vec2{.05, -.05} : fv::Vec2{.35, .2};
        const auto q = attitude(.2*std::sin(.23*time), .3*std::cos(.31*time), .37*time);
        const auto r = fv::rotationMatrix(q);
        const double bx = bias[0]+.8*std::sin(9*time), by = bias[1]+.7*std::cos(11*time);
        const fv::Vec2 acc{r[0][0]*bx+r[0][1]*by, r[1][0]*bx+r[1][1]*by};
        a = recent.update(acc, q, {}, i ? .05 : 0);
        b = cumulative.update(acc, q, {}, i ? .05 : 0);
    }
    close(a.bias_body[0], .35, .012); close(a.bias_body[1], .2, .012);
    require(std::hypot(b.bias_body[0]-.35, b.bias_body[1]-.2) > .12, "Regression does not expose stale bias");
}
} // namespace

int main() {
    try {
        close(velocityHeaderOdr(), 4);
        weightedAndClipped(); analyticMotion(); biasAndRotation(); invalidAndRollback(); changingBias();
        std::cout << "PASS velocity: weighted variable dt, clipped windows, motion, startup, rotating bias, reset and rollback\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL velocity: " << e.what() << '\n';
        return 1;
    }
}
