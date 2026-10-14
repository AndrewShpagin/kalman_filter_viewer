#include "flight_kalman.hpp"

double fromSecondTranslationUnit() {
    flight_kalman::FlightKalmanFilter filter;
    return filter.update({0, 0}, {}, {3, 4}, 10, 0).position[0];
}
