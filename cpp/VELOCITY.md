# C++ windowed velocity estimator

`include/flight_velocity.hpp` is a **standalone, header-only C++17 class** for
the algorithm in `velocity_flight_viewer.py`, including the corrected startup
body-bias learning. Copy this one header into your application. It uses only
the standard library. The older `FlightKalmanFilter` remains a separate class.

## Integration

```cpp
#include "flight_velocity.hpp"

flight_velocity::Parameters params;
params.fade_in = 6.0;
params.flat = 6.0;
params.fade_out = 6.0;
params.bias_warmup = 6.0;
params.bias_memory = 60.0;
params.coordinate_scale = 1.0;
params.lead = 1.0;
flight_velocity::FlightVelocityEstimator estimator(params);

// Call once per synchronized measurement, in time order.
// acceleration_ned: m/s^2, gravity already removed, our bias NOT removed.
// quaternion: x,y,z,w, body -> NED by default.
// visual_position: horizontal NED metres; dt_seconds: actual elapsed seconds.
auto result = estimator.update(
    {ned_acc.x, ned_acc.y},
    {q.x, q.y, q.z, q.w},
    {pos.x, pos.y},
    dt_seconds);

double vx = result.velocity[0];
double vy = result.velocity[1];
double forward_x = result.forward_position[0]; // scale*vis_x + lead*vx
double forward_y = result.forward_position[1];
```

Keep the same estimator instance across updates. The **first call initializes**
position and initial velocity/bias: `time_s=dt_s=0`, with no integration.
Its input `dt` may be zero or positive and is ignored. Subsequent calls require
finite `dt > 0`; zero/negative values are rejected, never clamped. Inputs must
be finite; a zero quaternion is rejected. Quaternions are normalized internally.
For inverse input, set `quaternion_direction = QuaternionDirection::NedToBody`.
The class does not guess conventions.

`update(const Measurement&)` is an equivalent overload. `state()`, `parameters()`
and `initialized()` inspect current state; `historySize()` reports the retained
interval count. `update` returns an independent `Estimate` value. Invalid input
or arithmetic overflow throws `std::invalid_argument`, leaving state and
history intact.

Acceleration and attitude at update `i` represent the interval `(i-1, i]`;
vision is its endpoint position. Align measurements to the same time. Input
acceleration is translational NED, not body accelerometer specific force.
The class does not remove gravity or compensate sensor latency. Body X/Y bias
is learned internally and rotated using full roll/pitch/yaw. Body-Z bias is
assumed zero. Height is not used by this algorithm.

## Configuration and state

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `fade_in`, `flat`, `fade_out` | 6, 6, 6 s | Oldest to newest; sum must be positive |
| `bias_warmup` | 6 s | Minimum information time for full trust per body direction; 0 disables ramp |
| `bias_memory` | 60 s | Exponential memory of bias observations; 0 restores the whole-flight fit |
| `initial_bias_body` | `{0, 0}` m/s² | Starting body X/Y bias |
| `initial_velocity` | `{0, 0}` m/s | Starting NED velocity; set if already moving |
| `learn_bias` | `true` | `false` holds initial body bias fixed |
| `coordinate_scale` | 1 | Positive multiplier on visual positions |
| `lead` | 1 s | Forward-position horizon; does not affect velocity or bias fitting |
| `quaternion_direction` | `BodyToNed` | Explicit quaternion convention |

`reset()` clears history and restores configured initial bias/velocity.
`reset(new_parameters)` changes configuration and starts a new history.
Replay all measurements to reproduce the Python UI's Apply behavior.
There are no Kalman measurement-noise parameters in this algorithm.

Only intervals overlapping the trailing window are retained. Bias-fit
information has its own exponential memory (60 s by default). Work and memory per update are
O(samples in window), about 360 intervals at 20 Hz with defaults. The class
uses `std::deque` and can allocate; it is not an allocation-free hard real-time
interface. There is no fixed sample-count cap. Synchronize access externally
if called from multiple threads.

| Output | Meaning |
| --- | --- |
| `velocity` | Estimated NED X/Y speed, m/s |
| `raw_position`, `measured_position` | Original and scaled visual positions, m |
| `forward_position` | Scaled visual position + `lead * velocity`, m |
| `bias_body_used`, `bias_ned_used` | Bias applied to this velocity update, m/s² |
| `bias_body`, `bias_ned` | Newly fitted bias, first applied on the next update |
| `corrected_acceleration` | Current raw NED acceleration minus `bias_ned_used` |
| `mean_visual_velocity`, `mean_integrated_velocity` | Two duration-weighted means being matched |
| `window_integral`, `normalization_offset` | Their sum is the output velocity |
| `bias_normal`, `bias_rhs` | Accumulated bias-fit information and right-hand side; not covariance |
| `bias_response` | 2x2 velocity response `C` to body bias |
| `window_span_s`, `window_complete` | Available window duration and whether the full window exists |

## Time integration and bias

For retained interval length `h_j` and midpoint fade weight `w_j`:

```
D = sum(w_j * h_j)
mean_visual   = sum(w_j * displacement_j) / D
mean_inertial = sum(w_j * h_j * I_mid_j) / D
velocity     = I_end - mean_inertial + mean_visual
```

`I` integrates `acceleration_ned - R_xy * previous_body_bias`, **without fade**.
The same previous bias is re-applied to each historical sample with its own
quaternion. The core accumulates visual displacements directly, avoiding
per-sample visual division by tiny `dt`. Both means use the same duration
weights and interval midpoints, matching the Python discrete model.

The oldest interval is clipped to `now - window_seconds`. Its visual displacement
is interpolated linearly; acceleration and attitude are held constant within
each input interval. Fades are evaluated at each retained interval's midpoint,
not integrated continuously. These sampling approximations are less accurate
for long gaps. Startup uses available intervals only; edge fades can overlap.

After calculating velocity, bias fitting accounts for the window response `C`
and the changing bias used between updates:

```
H = R_xy - (C_now - C_previous) / dt
r = raw_acc*dt - (v_now-v_previous)
    - C_now*b_used_now + C_previous*b_used_previous
decay = exp(-dt / bias_memory)
information_dt = -bias_memory * expm1(-dt / bias_memory)
normal = decay*normal + information_dt * transpose(H) * H
rhs    = decay*rhs + (information_dt/dt) * transpose(H) * r
```

For `bias_memory=0`, use decay=1 and information_dt=dt (the previous whole-flight
fit). Positive memory follows a changing effective offset instead of averaging
all earlier flight conditions forever. Its interval weights account for actual
elapsed time, including long intervals. It does not fade acceleration integration.

An analytic symmetric 2x2 eigensolve floors each information eigenvalue at
`bias_warmup * exp(-elapsed_learning_time / bias_memory)`, plus a small numerical
observability floor, blending from `initial_bias_body`. With memory=0, the startup
floor remains `bias_warmup`. Decaying the startup prior avoids permanently
attenuating bias when a rotating body provides little information per second.
Initial bias is zero by default
and trust accumulates gradually. This is not a hard bias or slew-rate limit.
The newly fitted bias is used on the **next** update.

## Build and CSV comparison

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

# Same time model as Python (20 Hz).
build/bin/velocity_filter_csv flight.csv --dt 0.05

# Actual timestamps in milliseconds.
build/bin/velocity_filter_csv flight.csv --time-column timestamp --time-scale 0.001

# Or elapsed seconds in a dt column.
build/bin/velocity_filter_csv flight.csv --dt-column dt
build/bin/velocity_filter_csv --help
```

With Visual Studio, executables are under `build/bin/Release/` with `.exe`.
Without CMake:

```sh
c++ -std=c++17 -O2 -Icpp/include cpp/velocity_filter_csv.cpp -o velocity_filter_csv
```

The runner shares the existing CSV parser and offline quaternion/yaw checks.
Input columns match Python, including `mheight` and `yaw` for export/diagnostics.
BOM, quoted fields, decimal commas and Unicode paths are supported. Output
defaults to **`result_velocity_cpp.csv`** beside the input; use `--export PATH`
to change it. Input overwriting is refused. Calculation and validation complete
before atomic output replacement.

The first 64 output columns match the Python velocity export, including
`bias_memory`. Six are appended:
`dt_s` (integrated interval, zero at initialization), `weighted_duration_s`
(`D` in seconds), and `bias_response_xx/xy/yx/yy`. `weight_sum` remains the
dimensionless sum of fade weights; it is not the normalizer for unequal
intervals. With fixed steps, `dt` retains the configured value on every row for
Python compatibility; in timestamp/dt-column mode it equals `dt_s`. Use
`time_s` as the PlotJuggler time axis.

```sh
python cpp/tests/compare_velocity_with_python.py --exe build/bin/velocity_filter_csv
python cpp/tests/compare_velocity_with_python.py --exe build/bin/velocity_filter_csv \
  --csv flight.csv --time-column timestamp --time-scale 0.001
```

With fixed steps dividing the window duration (including defaults), comparisons
call the Python estimator and check all 64 shared columns. For a
non-integer window/step ratio, Python currently retains whole oldest intervals;
C++ clips the partial interval, so differences are expected. Variable-dt
comparisons use a separate vectorized NumPy reference with the same clipping.
The Python viewer itself still uses fixed `--dt`, ignoring CSV timestamps.

Core tests cover analytic motion, unequal interval weights, partial windows,
rotating body bias, inverse quaternions, startup, unobservable directions,
reset, invalid-update rollback and multiple-translation-unit inclusion.
Numerical agreement verifies the port, not closed-loop flight stability.
