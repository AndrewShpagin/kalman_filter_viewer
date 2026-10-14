# C++ Kalman filter

`include/flight_kalman.hpp` is a standalone, header-only C++17 implementation of
the Python viewer's horizontal filter and external body-bias estimator. It uses
only the standard library; Eigen and Python are not needed to build or run it.

`kalman_filter_csv.cpp` reads the same semicolon-separated flight CSV and writes
`result_cpp.csv` next to the input. The class accepts an individual `dt` on every
update. The CSV runner can use timestamps, a column of elapsed intervals, or a
fixed interval for comparison with the Python viewer.

## Build

From the repository root, with CMake 3.16+ and a C++17 compiler:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Linux/macOS the executable is `build/bin/kalman_filter_csv` with a
single-configuration generator. With Visual Studio 2022 on Windows:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\bin\Release\kalman_filter_csv.exe flight.csv
```

Windows uses a Unicode entry point and filesystem paths, including paths with
spaces and Cyrillic characters. GCC/Linux is covered by the initial validation;
the Visual Studio configuration still needs a native Windows build check.

A direct build is also possible:

```sh
g++ -std=c++17 -O2 -Icpp/include cpp/kalman_filter_csv.cpp -o kalman_filter_csv
```

To embed the class, copy the single header into your application, or link the
CMake interface target `flight_kalman` to inherit its include path and C++17
requirement. No compiled library is required.

## Class API

```cpp
#include "flight_kalman.hpp"

flight_kalman::Parameters parameters;
parameters.noise.acceleration_sigma = 1.0;       // m/s^2
parameters.noise.min_position_sigma = 0.001;     // m
parameters.noise.height_error_fraction = 0.1;    // 10% of height
parameters.noise.initial_velocity_sigma = 1.0;   // m/s
parameters.coordinate_scale = 1.0;
parameters.quaternion_direction = flight_kalman::QuaternionDirection::BodyToNed;

flight_kalman::FlightKalmanFilter filter(parameters);
filter.reset({0.0, 0.0}); // initial NED velocity; optional, zero is the default

// First observation initializes position. No preceding interval is integrated.
filter.update({0.1, -0.2}, {0, 0, 0, 1}, {10, 20}, 5.0, 0.0);

// Noise can change between samples without clearing state, covariance or bias.
auto noise = filter.parameters().noise;
noise.acceleration_sigma = 0.7;
noise.height_error_fraction = 0.08;
filter.setNoiseParameters(noise);

// Order: acceleration NED, quaternion x/y/z/w, visual position, height, dt.
auto estimate = filter.update({0.12, -0.18}, {0, 0, 0, 1}, {10.03, 19.99}, 5.1, 0.063);
double x  = estimate.position[0];
double vx = estimate.velocity[0];
double bx = estimate.bias_body[0];

// Optional direct measurement sigma, instead of the height-based formula.
flight_kalman::Measurement sample;
sample.acceleration_ned = {0.11, -0.17};
sample.quaternion = {0, 0, 0.1, 0.99498743710662};
sample.visual_position = {10.05, 19.98};
sample.height = 5.2;
sample.dt = 0.047;
sample.position_sigma = 0.25; // metres; omit to use height and minimum sigma
estimate = filter.update(sample);
```

All uncertainty parameters are **standard deviations**, not variances.
`sigma_position = max(min_position_sigma, height_error_fraction * abs(height))`;
the measurement variance is its square. The coordinate multiplier applies to
`vis_x/y`, while position sigma is specified separately in metres after scaling.

| Method / field | Behaviour |
| --- | --- |
| `update(...)` | Initializes on the first call; predicts and corrects on every later call |
| `Measurement::dt` | Seconds since the previous sample; finite and strictly positive after initialization |
| `setNoiseParameters(...)` | Applies to subsequent updates, retaining state, covariance and accumulated bias |
| `initial_velocity_sigma` | Used only at initialization, including after `reset()`; changing it does not replace existing covariance |
| `setCoordinateScale(...)` | Scales subsequent visual measurements; reset and replay to rescale an entire flight |
| `reset({vx0, vy0})` | Clears state and bias history, preserves parameters, sets the velocity for the next initialization |
| `state()` / `initialized()` | Read the current estimate / check whether a sample has initialized it |
| `Measurement::position_sigma` | Optional positive per-sample position sigma, overriding the height formula |

`update()` returns a reference to internal state, valid until the next update or
reset. Use `auto estimate = ...` for a snapshot. The estimate exposes position,
velocity, raw/scaled visual position, raw/corrected acceleration, old/new body
and NED bias, covariance, Kalman gains, and bias-fit accumulators. Each covariance
has `xx`, `xv`, `vv`; each gain has `position` and `velocity`.

The first call returns `time_s = dt_s = 0`; its acceleration does not change
velocity or bias. Later calls use the supplied interval directly, with no fixed
period or clamping. Acceleration and attitude for sample `i` represent the
interval ending at that sample. This API takes both acceleration and a visual
measurement on each update; it does not implement separate asynchronous sensor
queues or prediction-only updates.

Zero-length quaternions, nonfinite inputs, invalid sigma or time values, and
numeric overflow throw exceptions. A failed update leaves all filter state and
bias history intact. The valid update path uses fixed-size storage and performs
no dynamic allocation.

## CSV runner

Required input columns and aliases match the Python viewer: `vis_x`, `vis_y`,
`mheight`, `acc_x/y` (or `acc.x/y`), `quat.x/y/z/w` (or underscore names), and
`yaw`. Optional raw acceleration columns support frame diagnostics. The parser
accepts UTF-8 BOM, quoted fields, decimal commas and CRLF. Bad required values
are rejected instead of dropping rows.

The runner checks quaternion direction against yaw and optional body raw
acceleration. `--quat-direction body-to-ned|ned-to-body` and
`--yaw-unit deg|rad` override automatic selection. The header-only class takes
an explicit quaternion convention and requires no yaw measurement.

```sh
# Exactly the Python viewer's default time base: fixed 50 ms.
build/bin/kalman_filter_csv flight.csv

# Actual intervals from timestamps in milliseconds.
build/bin/kalman_filter_csv flight.csv --time-column timestamp --time-scale 0.001

# Explicit elapsed-time column already in seconds.
build/bin/kalman_filter_csv flight.csv --dt-column elapsed_s

# Settings corresponding to the Python viewer's fields.
build/bin/kalman_filter_csv flight.csv --dt 0.05 --sigma-acc 0.7 \
  --min-pos-sigma 0.02 --k-height 0.1 --initial-vel-sigma 1 \
  --coord-scale 1.1 --vx0 0 --vy0 0 --export result_cpp.csv

# Optional columns supplying noise independently at each sample.
build/bin/kalman_filter_csv flight.csv --time-column timestamp --time-scale 0.001 \
  --sigma-acc-column accel_sigma --position-sigma-column vision_sigma

build/bin/kalman_filter_csv --help
```

Use only one of `--dt`, `--time-column` or `--dt-column`. `--time-scale` converts
the selected column's units to seconds (1 by default). Timestamps must strictly
increase; output time starts at zero. In interval-column mode, the first cell
may be empty or nonnegative and is ignored; all later intervals must be positive.
Noise columns, when selected, must contain valid values on every row.

By default the output is **`result_cpp.csv` beside the input**, with `;` as the
separator. `--export PATH` / `--output PATH` changes the destination. The runner
refuses to overwrite its input. It validates and calculates before atomically
replacing the output, so a failed calculation leaves an existing result intact.

The export contains all **48 Python columns in the same order**, including raw
`vis_x/y`, modified `vis_scaled_x/y`, estimated position/speed, biases, covariance
and gains. Two columns are appended: `dt_s` and `sigma_acc`. Use `time_s` as the
time axis when loading either export in PlotJuggler.

## Model and frames

The two Kalman states are `[x, vx]` and `[y, vy]` in horizontal NED coordinates.
Acceleration input is already in NED, in m/s², before bias correction. The
quaternion is normalized and its full roll/pitch/yaw rotation maps the external
body X/Y bias into NED; body-Z bias is assumed zero. Height only determines
visual measurement uncertainty; vertical position and velocity are not fused.

The estimator retains the Python model: bias is a cumulative least-squares fit
outside the Kalman state, using NED velocity increments transformed with the
current attitude. Its uncertainty is not included in the Kalman covariance.
Unobservable body directions retain their prior bias estimate. Acceleration
sigma describes an independent per-sample error held constant over the supplied
interval, not a continuous noise spectral density. Both covariance prediction
and bias accumulation use the current `dt`.

## Comparison tests

`ctest` runs the C++ core tests: variable intervals, analytic motion, dynamic
noise, rotating body bias, inverse quaternions, singular orientations, invalid
input preservation and inclusion from multiple translation units.

For an independent numerical comparison, install NumPy and run:

```sh
python cpp/tests/compare_with_python.py --exe build/bin/kalman_filter_csv

# Also compare a real flight at fixed dt and with its actual timestamps.
python cpp/tests/compare_with_python.py --exe build/bin/kalman_filter_csv \
  --csv flight.csv --time-column timestamp --time-scale 0.001
```

On Windows, pass `build/bin/Release/kalman_filter_csv.exe` as `--exe`. The script
checks every shared output column. Fixed-step comparisons call the original
Python `estimate()` directly. Variable-step comparisons reuse the original
Python `Kalman1D` and NumPy `BodyBiasEstimator`, feeding per-row intervals and
noise values. Synthetic tests cover tilt, changing yaw, inverse quaternions,
dynamic noise, coordinate scaling, CSV quoting and Unicode paths. Temporary
test CSVs are removed automatically; your input is never modified.

The Python viewer still uses a fixed interval. For a direct `result.csv` versus
`result_cpp.csv` comparison, give both programs the same fixed `--dt` and noise
settings. Results calculated using actual timestamps can legitimately differ.
