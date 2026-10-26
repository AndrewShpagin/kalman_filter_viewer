# C++ windowed velocity estimator

`include/flight_velocity.hpp` is a **standalone, header-only C++17 class** for
the algorithm in `velocity_flight_viewer.py`, including the corrected startup
body-bias learning. Copy this one header into your application. It uses only
the standard library. The older `FlightKalmanFilter` remains a separate class.

Detailed Ukrainian walkthrough: [VELOCITY_ALGORITHM_UK.md](VELOCITY_ALGORITHM_UK.md).
It derives the bias equations, explains every processing stage and variable-dt
weight, and includes an analytic 20–60 FPS phase-response check. The header
also contains Ukrainian comments beside the implementation.

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
params.vision_delay = 0.0; // seconds, relative to IMU; set the measured/assumed delay
flight_velocity::FlightVelocityEstimator estimator(params);

// Call once per received measurement bundle, in time order.
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
vision is the position at `time_s - vision_delay`. Acceleration is translational
NED, not body accelerometer specific force. The class removes the configured
constant vision/IMU time offset; it does not remove gravity or estimate latency. Body X/Y bias
is learned internally and rotated using full roll/pitch/yaw. Body-Z bias is
assumed zero. Height is not used by this algorithm.

## Configuration and state

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `fade_in`, `flat`, `fade_out` | 6, 6, 6 s | Oldest to newest; sum must be positive |
| `bias_warmup` | 6 s | Initial information floor per body direction; decays with finite memory; 0 disables ramp |
| `bias_memory` | 60 s | Exponential memory of bias observations; 0 restores the whole-flight fit |
| `initial_bias_body` | `{0, 0}` m/s² | Starting body X/Y bias |
| `initial_velocity` | `{0, 0}` m/s | Starting NED velocity; set if already moving |
| `learn_bias` | `true` | `false` holds initial body bias fixed |
| `coordinate_scale` | 1 | Positive multiplier on visual positions |
| `vision_delay` | 0 s | Vision age relative to IMU; finite, nonnegative; configure at construction/reset |
| `lead` | 1 s | Forward-position horizon; does not affect velocity or bias fitting |
| `quaternion_direction` | `BodyToNed` | Explicit quaternion convention |

Window fades and forward time can be changed during streaming, without reset:

```cpp
estimator.setFades(3.0, 6.0, 3.0); // fade_in, flat, fade_out, seconds
estimator.setForwardTime(0.7);     // forward_time / Parameters::lead, seconds

auto result = estimator.update(acceleration_ned, quaternion, visual_position, dt);
```

`setFades()` takes effect on the next successful `update()`. It preserves
elapsed time, learned body bias, its fit accumulators, and available history.
The velocity can change when weights change. To keep that change from being
mistaken for acceleration, the bias fit re-evaluates the previous timestamp
with the new window before differencing velocity and its bias response.
The public `state()` retains the previous estimate until the next update.
Repeated setters with unchanged fade values are a no-op and can be called
each measurement.

If the window grows beyond retained history, its actual `window_span_s` grows
as new data arrives; `window_complete` stays false until the requested duration
is available. Previously discarded samples are not recreated or zero-padded.
Shrinking the window applies on the next update and discards expired intervals.

`setForwardTime()` immediately recomputes `state().forward_position` using the
current scaled position and velocity. It changes no velocity, bias, time, or
history. Previously returned `Estimate` copies remain unchanged. Both setters
validate input before committing; invalid values or arithmetic overflow leave
the current settings/state intact. Fade values and forward time must be finite
and nonnegative; the sum of fade/flat durations must be positive. Synchronize
setters and `update()` with the same mutex if used from different threads.

`reset()` clears history and restores configured initial bias/velocity.
`reset(new_parameters)` changes configuration and starts a new history.
Replay all measurements to reproduce the Python UI's Apply behavior.
There are no Kalman measurement-noise parameters in this algorithm.

Only intervals overlapping `window_seconds + vision_delay` are retained. Bias-fit
information has its own exponential memory (60 s by default). Work and memory per update are
O(samples in window + delay), about 360 intervals at 20 Hz with zero-delay defaults.
For a fixed window/delay and bounded sample rate, memory does not grow with flight duration. The class
uses `std::deque` and can allocate; it is not an allocation-free hard real-time
interface. There is no fixed sample-count cap. Synchronize access externally
if called from multiple threads.

| Output | Meaning |
| --- | --- |
| `velocity` | Estimated NED X/Y speed **now**, m/s |
| `delayed_velocity` | Speed at the aligned vision epoch; valid when `alignment_ready` |
| `prediction_delta_velocity` | Corrected IMU integral from that epoch to now; before alignment, integral from time zero |
| `vision_time_s`, `alignment_ready` | `time_s - vision_delay`; whether this epoch has reached the available IMU time origin |
| `bias_fit_dt_s` | Duration admitted to bias learning this update; zero during alignment startup or fixed-bias mode |
| `raw_position`, `measured_position` | Original and scaled visual positions, m |
| `forward_position` | Scaled visual position + `lead * velocity`, m |
| `bias_body_used`, `bias_ned_used` | Bias applied to this velocity update, m/s² |
| `bias_body`, `bias_ned` | Newly fitted bias, first applied on the next update |
| `corrected_acceleration` | Current raw NED acceleration minus `bias_ned_used` |
| `mean_visual_velocity`, `mean_integrated_velocity` | Two duration-weighted means being matched |
| `window_integral`, `normalization_offset` | Their sum is `delayed_velocity` when aligned |
| `bias_normal`, `bias_rhs` | Accumulated bias-fit information and right-hand side; not covariance |
| `bias_response` | 2x2 **delayed** velocity response `C` to body bias |
| `window_span_s`, `window_complete` | Available window duration and whether the full window exists |

## Time integration and bias

For retained interval length `h_j` and midpoint fade weight `w_j`:

```
D = sum(w_j * h_j)
mean_visual   = sum(w_j * displacement_j) / D
mean_inertial = sum(w_j * h_j * I_mid_j) / D
delayed_velocity = I_end - mean_inertial + mean_visual
velocity = delayed_velocity + integral(corrected_acc, vision_epoch, now)
```

`I` integrates `acceleration_ned - R_xy * previous_body_bias`, **without fade**.
The same previous bias is re-applied to each historical sample with its own
quaternion. The core accumulates visual displacements directly, avoiding
per-sample visual division by tiny `dt`. Both means use the same duration
weights and interval midpoints, matching the Python discrete model.

The window ends at `vision_epoch = now - vision_delay`; its left edge is
`vision_epoch - window_seconds`, clipped to available history. Two cursors merge
IMU boundaries with shifted vision boundaries, splitting fractional intervals.
Vision position is interpolated linearly; acceleration and attitude are held
constant within each IMU interval. Fades are evaluated at each shared piece's midpoint,
not integrated continuously. These sampling approximations are less accurate
for long gaps. Startup uses available intervals only; edge fades can overlap.

After calculating velocity, bias fitting accounts for the window response `C`
and the changing bias used between updates:

```
delta_raw = integral(raw_acc, previous_vision_epoch, vision_epoch)
delta_M   = integral(R_xy, previous_vision_epoch, vision_epoch)
H = (delta_M - C_now + C_previous) / dt
r = delta_raw - (delayed_v_now-delayed_v_previous)
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

Python and C++ export the same **78 columns**. The old 70 C++ columns retain
names/order; eight delay diagnostics are appended: `vision_delay`,
`vision_time_s`, `alignment_ready`, `bias_fit_dt_s`, `delayed_speed_x/y`,
`prediction_delta_v_x/y`. `dt_s` is zero at initialization;
`weighted_duration_s` is the mean normalizer. `weight_sum` sums weights over
shared subintervals and is not comparable across different delays/sample rates.
With fixed steps `dt` keeps the configured value on every row; in timestamp/dt
mode it equals `dt_s`. Use `time_s` as the PlotJuggler axis.

```sh
python cpp/tests/compare_velocity_with_python.py --exe build/bin/velocity_filter_csv
python cpp/tests/compare_velocity_with_python.py --exe build/bin/velocity_filter_csv \
  --csv flight.csv --time-column timestamp --time-scale 0.001
```

Both implementations support fixed or variable time, fractional windows and
fractional delay. The comparison checks all exported fields. Python now clips
partial oldest intervals too; older Python releases retained the full interval,
so fractional-window results can differ slightly even with zero delay.

Core tests cover analytic motion, unequal interval weights, partial windows,
rotating body bias, inverse quaternions, startup, unobservable directions,
reset, invalid-update rollback and multiple-translation-unit inclusion.
Numerical agreement verifies the port, not closed-loop flight stability.

## Constant vision delay

Set `params.vision_delay` before constructing the estimator (or use
`reset(new_parameters)` and replay). The GUI's **Vision delay (s)** field and
both runners' `--vision-delay` option use the same convention. Zero is the
default; there is no automatic delay search. Dynamic `setFades()` and
`setForwardTime()` still preserve the streaming state; changing delay requires
reset/replay so differently aligned bias observations are not mixed.

Before `time_s >= vision_delay`, no pre-log IMU samples exist. Keep initial
body bias, integrate current speed from `initial_velocity` at IMU time zero,
and export `alignment_ready=0`. Establish the first delayed-speed reference
once overlap exists. Learn bias only between two available aligned epochs.
`window_complete` becomes true after `vision_delay + window_seconds` (or later
when a previously shortened window grows). The new bias first affects the
next update; prediction to now uses the same old bias as the window.

`forward_position = scaled_vision + lead * velocity_now` is unchanged.
Vision position itself still belongs to `vision_time_s`; the implementation
does **not** additionally integrate position to now or silently add the delay
to `lead`. This is the requested experimental output, not an exact prediction
of physical position at `now + lead`.

```sh
velocity_filter_csv flight.csv --time-column timestamp --time-scale .001 \
  --vision-delay .2 --fade-in 32 --flat 1 --fade-out 1 --lead 1
```

Here `.2` is an example input, not an inferred delay for a particular flight.
`test_velocity_delay.cpp` checks a known 237 ms and 2137 ms delay, irregular
20–60 FPS, rotating BODY bias, startup, dynamic fades, history bounds and
invalid-update rollback. The analytic test includes an uncompensated control;
it does not infer the real flight's delay or flight stability.
