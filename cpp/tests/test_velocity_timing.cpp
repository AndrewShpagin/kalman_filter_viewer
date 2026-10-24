#include "flight_velocity.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace fv = flight_velocity;

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double omega = 2*pi/4; // 4-second oscillation; 1 m position amplitude

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Fit y = offset + sine*sin(omega*t) + cosine*cos(omega*t), using elapsed
// time as weight. Centering removes the intercept without a 3x3 solver.
// This reference knows nothing about the estimator's window or bias equations.
struct HarmonicFit {
    double mass=0, s=0, c=0, ss=0, cc=0, sc=0, y=0, ys=0, yc=0;
    void add(double t, double dt, double value) {
        const double sn=std::sin(omega*t), cs=std::cos(omega*t);
        mass+=dt; s+=dt*sn; c+=dt*cs;
        ss+=dt*sn*sn; cc+=dt*cs*cs; sc+=dt*sn*cs;
        y+=dt*value; ys+=dt*value*sn; yc+=dt*value*cs;
    }
    fv::Vec2 coefficients() const {
        const double a=ss-s*s/mass, b=sc-s*c/mass, d=cc-c*c/mass;
        const double r=ys-y*s/mass, q=yc-y*c/mass, det=a*d-b*b;
        return {(d*r-b*q)/det, (a*q-b*r)/det};
    }
    double phaseDegrees() const {
        const auto v=coefficients();
        return std::atan2(v[1], v[0])*180/pi;
    }
    double amplitude() const {
        const auto v=coefficients();
        return std::hypot(v[0], v[1]);
    }
};

double interval(int mode, int i) {
    if (mode==0) return 1.0/60;
    if (mode==1) return 1.0/20;
    // Abrupt FPS changes plus irregular intervals, all within 20...60 FPS.
    const double jitter[]{1.0/60, 1.0/23, 1.0/37, 1.0/20, 1.0/53};
    switch ((i/120)%3) {
        case 0: return 1.0/60;
        case 1: return 1.0/20;
        default: return jitter[i%5];
    }
}

void oscillation(int mode, bool learn_bias, bool visual_only) {
    fv::Parameters p;
    p.learn_bias=learn_bias;
    p.initial_velocity={visual_only ? 0 : omega, 0};
    fv::FlightVelocityEstimator estimator(p);
    estimator.update({}, {}, {}, 0);
    double t=0, previous_v=omega, max_speed_error=0, max_bias=0;
    HarmonicFit speed, forward;
    for (int i=0; t<120; ++i) {
        const double dt=interval(mode, i);
        t+=dt;
        const double position=std::sin(omega*t), velocity=omega*std::cos(omega*t);
        // Exact interval-average acceleration: integration must recover the
        // endpoint velocity. Position is the continuous sine, not generated
        // by the estimator's piecewise-constant acceleration approximation.
        const double acceleration=visual_only ? 0 : (velocity-previous_v)/dt;
        previous_v=velocity;
        const double yaw=.31*t;
        const fv::Quaternion q{0, 0, std::sin(yaw/2), std::cos(yaw/2)};
        const auto r=estimator.update({acceleration, 0}, q, {position, 0}, dt);
        if (t<60) continue; // full window; assess steady response separately
        max_speed_error=std::max(max_speed_error, std::abs(r.velocity[0]-velocity));
        max_bias=std::max(max_bias, std::hypot(r.bias_body[0], r.bias_body[1]));
        speed.add(t, dt, r.velocity[0]);
        forward.add(t, dt, r.forward_position[0]);
    }
    std::cout << (mode==0 ? "60 FPS" : mode==1 ? "20 FPS" : "20..60 FPS")
              << ", " << (visual_only ? "vision-only artefact" : "physical sine")
              << ", learn=" << learn_bias
              << ": speed amplitude=" << speed.amplitude()
              << ", forward phase=" << forward.phaseDegrees()
              << " deg, forward gain=" << forward.amplitude()
              << ", max |body bias|=" << max_bias;
    if (!visual_only) std::cout << ", max speed error=" << max_speed_error;
    std::cout << '\n';
    if (visual_only) {
        // At this frequency the 6/6/6 s visual averaging strongly suppresses
        // the derivative of the artificial position, while raw position
        // still goes directly into forward_position. This is expected,
        // not evidence of a dt bug or a closed-loop stability result.
        require(speed.amplitude()<.01, "Visual artefact unexpectedly passed the long velocity window");
        require(std::abs(forward.phaseDegrees())<1, "Visual-only signal unexpectedly gained substantial lead");
        require(std::abs(forward.amplitude()-1)<.01, "Raw visual position did not pass through");
    } else {
        const double expected_phase=std::atan(omega*p.lead)*180/pi;
        const double expected_gain=std::hypot(1.0, omega*p.lead);
        require(max_speed_error<.005, "Variable FPS corrupted analytic speed");
        require(max_bias<.002, "Zero-bias analytic motion created a large bias");
        require(std::abs(speed.phaseDegrees()-90)<.1, "Analytic speed has unexpected phase delay");
        require(std::abs(forward.phaseDegrees()-expected_phase)<.1, "Forward phase differs from p + lead*v");
        require(std::abs(forward.amplitude()-expected_gain)<.003, "Forward gain differs from p + lead*v");
    }
}

void zeroBiasConstantAcceleration() {
    // Constant acceleration, including startup: changing dt must not invent
    // appreciable bias. A partially clipped oldest displacement still has
    // the documented linear-interpolation approximation.
    fv::Parameters p;
    p.initial_velocity={.7, -.4};
    fv::FlightVelocityEstimator f(p);
    fv::Vec2 position{}, velocity=p.initial_velocity;
    f.update({}, {}, position, 0);
    for (int i=0; i<1800; ++i) {
        const double dt=interval(2, i);
        const fv::Vec2 acceleration{.03, -.02};
        for (int j=0; j<2; ++j) {
            position[j]+=velocity[j]*dt+.5*acceleration[j]*dt*dt;
            velocity[j]+=acceleration[j]*dt;
        }
        const auto s=f.update(acceleration, {}, position, dt);
        for (int j=0; j<2; ++j) {
            require(std::abs(s.velocity[j]-velocity[j])<1e-8, "Variable dt changed exact motion");
            require(std::abs(s.bias_body[j])<1e-8, "Variable dt created a fictitious bias");
        }
    }
}
} // namespace

int main() {
    try {
        std::cout << std::setprecision(7);
        zeroBiasConstantAcceleration();
        for (int mode=0; mode<3; ++mode) oscillation(mode, true, false);
        oscillation(2, false, false);
        oscillation(2, false, true);
        std::cout << "PASS timing: elapsed-time integration and analytic phase; no flight-stability claim\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL timing: " << e.what() << '\n';
        return 1;
    }
}
