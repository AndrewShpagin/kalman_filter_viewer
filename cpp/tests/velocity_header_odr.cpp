#include "flight_velocity.hpp"
#include "flight_kalman.hpp"

double velocityHeaderOdr() {
    flight_velocity::FlightVelocityEstimator estimator;
    flight_kalman::FlightKalmanFilter kalman;
    return estimator.update({0, 0}, {}, {4, 5}, 0).forward_position[0]
           + kalman.update({0, 0}, {}, {0, 0}, 1, 0).position[0];
}
