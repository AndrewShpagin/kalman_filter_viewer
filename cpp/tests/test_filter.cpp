#include "flight_kalman.hpp"
#include <iostream>
#include <limits>
#include <string>

namespace fk = flight_kalman;
double fromSecondTranslationUnit();

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void close(double a, double b, double tolerance = 1e-10) {
    if (!(std::abs(a-b) <= tolerance))
        throw std::runtime_error("Values differ: " + std::to_string(a) + " / " + std::to_string(b));
}

void exactMotionWithVariableDt() {
    fk::FlightKalmanFilter filter;
    filter.reset({2, -1});
    filter.update({0.8, -0.2}, {}, {3, 4}, 10, 0);
    double time = 0;
    for (int i = 0; i < 500; ++i) {
        const double dt = (i % 3 == 0) ? 0.017 : (i % 3 == 1) ? 0.053 : 0.14;
        time += dt;
        const fk::Vec2 position{3 + 2*time + 0.4*time*time, 4 - time - 0.1*time*time};
        const auto& s = filter.update({0.8, -0.2}, {}, position, 10, dt);
        close(s.position[0], position[0], 1e-9);
        close(s.velocity[0], 2+0.8*time, 1e-9);
        close(s.velocity[1], -1-0.2*time, 1e-9);
        close(s.bias_body[0], 0, 1e-10);
        close(s.time_s, time);
        for (const auto& p : s.covariance)
            require(p.xx >= 0 && p.vv >= 0 && p.xx*p.vv-p.xv*p.xv > -1e-12, "Covariance lost PSD");
    }
}

void dynamicNoiseAndReset() {
    fk::FlightKalmanFilter f;
    f.update({0,0}, {}, {1,2}, 10, 0);
    auto noise = f.parameters().noise;
    noise.acceleration_sigma = 2;
    noise.min_position_sigma = 0.2;
    noise.height_error_fraction = 0;
    noise.initial_velocity_sigma = 4;
    f.setNoiseParameters(noise);
    close(f.state().covariance[0].vv, 1); // do not reinitialize existing covariance
    const auto& s = f.update({0,0}, {}, {1,2}, 10, 0.2);
    close(s.sigma_position, 0.2);
    const double predicted_xx = 1 + 0.2*0.2 + 0.25*std::pow(0.2,4)*4;
    close(s.gain[0].position, predicted_xx/(predicted_xx+0.04));
    fk::Measurement m{{0,0},{},{1,2},10,0.1,0.5};
    close(f.update(m).sigma_position, 0.5);
    f.setCoordinateScale(2);
    close(f.update({0,0}, {}, {1,2}, 10, 0.1).measured_position[0], 2);
    f.reset({3,4});
    require(!f.initialized(), "Reset must clear initialization");
    const auto& reset = f.update({100,100}, {}, {1,2}, 10, 0.2);
    close(reset.time_s, 0);
    close(reset.position[0], 2);
    close(reset.velocity[0], 3);
    close(reset.covariance[0].vv, 16);
    close(reset.bias_body[0], 0);
}

void rotatingBodyBias() {
    fk::Parameters p;
    p.noise.height_error_fraction = 0;
    p.noise.min_position_sigma = 1e-5;
    fk::FlightKalmanFilter f(p);
    const fk::Vec2 velocity{1.2,-0.5};
    f.reset(velocity);
    f.update({0,0}, {}, {0,0}, 10, 0);
    double t = 0;
    for (int i = 1; i <= 2000; ++i) {
        const double dt = i%2 ? 0.03 : 0.07;
        t += dt;
        const double yaw = 0.2*t;
        const fk::Quaternion q{0,0,std::sin(yaw/2),std::cos(yaw/2)};
        const auto r = fk::rotationMatrix(q);
        const fk::Vec2 acceleration{r[0][0]*0.2-r[0][1]*0.13, r[1][0]*0.2-r[1][1]*0.13};
        f.update(acceleration, q, {velocity[0]*t,velocity[1]*t}, 10, dt);
    }
    close(f.state().bias_body[0], 0.2, 0.003);
    close(f.state().bias_body[1], -0.13, 0.003);
    fk::Parameters inverse = p;
    inverse.quaternion_direction = fk::QuaternionDirection::NedToBody;
    const auto a = fk::rotationMatrix({0.1,0.2,0.3,0.9});
    const auto b = fk::rotationMatrix({-0.1,-0.2,-0.3,0.9}, inverse.quaternion_direction);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) close(a[i][j],b[i][j]);
}

void unobservableAndInvalidInput() {
    fk::Parameters p;
    p.noise.height_error_fraction = 0;
    p.noise.min_position_sigma = 1e-5;
    fk::FlightKalmanFilter f(p);
    const double q = std::sqrt(0.5);
    for (int i = 0; i < 300; ++i) f.update({0,0.13}, {0,q,0,q}, {0,0}, 0, i ? 0.05 : 0);
    close(f.state().bias_body[0], 0, 1e-8); // body X cannot be seen in NED XY
    close(f.state().bias_body[1], 0.13, 0.002);
    const auto before = f.state();
    for (const auto& bad : {
        fk::Measurement{{0,0},{},{0,0},0,0,{}},
        fk::Measurement{{0,0},{},{0,0},0,-0.1,{}},
        fk::Measurement{{0,0},{0,0,0,0},{0,0},0,0.1,{}},
        fk::Measurement{{std::numeric_limits<double>::quiet_NaN(),0},{},{0,0},0,0.1,{}},
        fk::Measurement{{0,0},{},{0,0},0,0.1,0.0},
        fk::Measurement{{0,0},{},{0,0},0,1e200,{}}
    }) {
        bool rejected = false;
        try { f.update(bad); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Bad measurement was accepted");
        close(f.state().time_s, before.time_s);
        close(f.state().position[0], before.position[0]);
        close(f.state().bias_body[1], before.bias_body[1]);
    }
    auto noise = f.parameters().noise;
    noise.acceleration_sigma = -1;
    bool rejected = false;
    try { f.setNoiseParameters(noise); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Bad noise was accepted");
    close(f.parameters().noise.acceleration_sigma, p.noise.acceleration_sigma);
}

int main() {
    try {
        close(fromSecondTranslationUnit(), 3);
        exactMotionWithVariableDt();
        dynamicNoiseAndReset();
        rotatingBodyBias();
        unobservableAndInvalidInput();
        std::cout << "All C++ filter checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
