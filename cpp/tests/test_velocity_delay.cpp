#include "flight_velocity.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace fv = flight_velocity;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
constexpr double w = .8;
fv::Vec2 position(double t) { return {2*std::sin(w*t), std::cos(.6*t)}; }
fv::Vec2 velocity(double t) { return {2*w*std::cos(w*t), -.6*std::sin(.6*t)}; }
fv::Quaternion attitude(double t) {
    const double roll=.2*std::sin(.19*t), pitch=.25*std::cos(.23*t), yaw=.3*t;
    const double cr=std::cos(roll/2), sr=std::sin(roll/2), cp=std::cos(pitch/2), sp=std::sin(pitch/2);
    const double cy=std::cos(yaw/2), sy=std::sin(yaw/2);
    return {sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy};
}

void delayedMotion(bool learn, bool rotating, double delay) {
    fv::Parameters p;
    p.fade_in=.6; p.flat=.4; p.fade_out=.3; p.vision_delay=delay;
    p.initial_velocity=velocity(0); p.learn_bias=learn;
    p.bias_memory=4; p.bias_warmup=2;
    const fv::Vec2 truth_bias{.12, -.08};
    if (!learn) p.initial_bias_body=truth_bias;
    fv::FlightVelocityEstimator f(p);
    auto wrong_parameters=p; wrong_parameters.vision_delay=0;
    fv::FlightVelocityEstimator wrong(wrong_parameters);
    f.update({}, {}, position(-delay), 0);
    wrong.update({}, {}, position(-delay), 0);
    double t=0, squared_error=0, wrong_error=0, bias_error=0, mass=0;
    double delayed_error=0;
    std::size_t max_history=0;
    for (int i=0; t<90; ++i) {
        const double steps[]{1./60, 1./23, 1./37, 1./20, 1./53};
        const double dt=steps[i%5], previous_t=t;
        t+=dt;
        const auto q=rotating ? attitude(t) : fv::Quaternion{};
        const auto r=fv::rotationMatrix(q);
        const auto v=velocity(t), old_v=velocity(previous_t);
        fv::Vec2 a{};
        for (int j=0; j<2; ++j)
            a[j]=(v[j]-old_v[j])/dt+r[j][0]*truth_bias[0]+r[j][1]*truth_bias[1];
        const auto s=f.update(a, q, position(t-delay), dt);
        const auto unaligned=wrong.update(a, q, position(t-delay), dt);
        if (t<delay) {
            require(!s.alignment_ready && s.bias_fit_dt_s==0, "Learned bias before aligned history");
            require(s.bias_body==p.initial_bias_body, "Changed initial bias before alignment");
        }
        if (previous_t<delay) require(s.bias_fit_dt_s==0, "Used a pre-log velocity for bias fitting");
        max_history=std::max(max_history, f.historySize());
        if (t<30) continue;
        mass+=dt;
        for (int j=0; j<2; ++j) {
            squared_error+=dt*std::pow(s.velocity[j]-v[j], 2);
            wrong_error+=dt*std::pow(unaligned.velocity[j]-v[j], 2);
            bias_error=std::max(bias_error, std::abs(s.bias_body[j]-truth_bias[j]));
            delayed_error=std::max(delayed_error, std::abs(s.delayed_velocity[j]-velocity(t-delay)[j]));
            require(std::abs(s.velocity[j]-s.delayed_velocity[j]-s.prediction_delta_velocity[j])<1e-12,
                    "Prediction diagnostic is inconsistent");
            require(std::abs(s.forward_position[j]-position(t-delay)[j]-p.lead*s.velocity[j])<1e-12,
                    "Vision delay silently changed forward-position semantics");
        }
    }
    const double rms=std::sqrt(squared_error/mass), wrong_rms=std::sqrt(wrong_error/mass);
    std::cout << "delay=" << delay << ", learn=" << learn << ", rotating=" << rotating
              << ": velocity RMS=" << rms << ", without alignment=" << wrong_rms
              << ", max bias error=" << bias_error << ", delayed speed error=" << delayed_error
              << ", max history=" << max_history << '\n';
    require(rms<.003 && delayed_error<.005, "Delay compensation failed analytic motion");
    require(bias_error<.003, "Aligned rotating motion corrupted body bias");
    require(rms<wrong_rms*.05, "Compensation did not remove the known delay error");
    require(max_history<=static_cast<std::size_t>(std::ceil((p.windowSeconds()+delay)*60))+2,
            "History grew beyond window + delay");
}

void startupAndDynamicFades() {
    fv::Parameters p; p.vision_delay=.237; p.initial_velocity={2, -1};
    p.fade_in=.2; p.flat=.2; p.fade_out=.2;
    fv::FlightVelocityEstimator f(p);
    f.update({}, {}, {-2*p.vision_delay, p.vision_delay}, 0);
    double t=0;
    for (int i=0; i<3000; ++i) {
        const double dt=i%2 ? .02 : .05; t+=dt;
        if (i==1000) f.setFades(.7, .1, .2);
        if (i==2000) f.setFades(.1, .1, .1);
        auto s=f.update({}, {}, {2*(t-p.vision_delay), -(t-p.vision_delay)}, dt);
        require(std::abs(s.velocity[0]-2)<1e-10 && std::abs(s.velocity[1]+1)<1e-10,
                "Startup or fade changes altered constant speed");
        require(std::hypot(s.bias_body[0], s.bias_body[1])<1e-10, "Fade change manufactured bias");
        if (i>2100) require(f.historySize()<30, "Shrink failed to prune delayed history");
    }
    const auto before=f.state(); const auto count=f.historySize();
    bool threw=false;
    try { f.update({1e308, 0}, {}, {}, 10); } catch (const std::invalid_argument&) { threw=true; }
    require(threw && f.historySize()==count && f.state().velocity==before.velocity,
            "Failed delayed update modified state");
    f.reset();
    require(f.historySize()==0 && !f.state().alignment_ready, "Reset kept aligned history");
    p.vision_delay=-.1; threw=false;
    try { fv::FlightVelocityEstimator invalid(p); } catch (const std::invalid_argument&) { threw=true; }
    require(threw, "Accepted negative vision delay");
}
}
int main() {
    try {
        for (double delay : {.237, 2.137}) {
            delayedMotion(false, true, delay);
            delayedMotion(true, true, delay);
            delayedMotion(true, false, delay);
        }
        startupAndDynamicFades();
        std::cout << "PASS delay: fractional alignment, variable dt, body rotation, prediction, startup and bounded history\n";
    } catch (const std::exception& e) { std::cerr << "FAIL delay: " << e.what() << '\n'; return 1; }
}
