# Kalman Filter Viewer

Desktop viewer for horizontal drone flight data from semicolon-separated CSV files. It combines vision positions and NED accelerations in a Kalman filter, estimates accelerometer bias in body axes, and replays the results in a moving time window.

The repository also includes a [C++17 header-only filter and CSV runner](cpp/README.md), with per-sample time intervals and dynamically adjustable noise. The runner exports `result_cpp.csv` with the same columns as the Python viewer, plus `dt_s` and `sigma_acc`.

An independent [experimental velocity viewer](VELOCITY_VIEWER.md) estimates speed by integrating bias-corrected acceleration and matching its window mean to visual speed. It uses no Kalman filter and has no C++ implementation. Run `python velocity_flight_viewer.py` to open its CSV replay window, or pass a CSV filename. It exports `result_velocity.csv` and plots `vis + lead * estimated_speed` with a default lead of one second.

## Install

Use Python 3.10 or newer with a desktop GUI environment.

```bash
git clone https://github.com/AndrewShpagin/kalman_filter_viewer.git
cd kalman_filter_viewer
python -m pip install -r requirements.txt
```

The default Tk GUI needs Tkinter. On Debian/Ubuntu, install it with `sudo apt install python3-tk` if it is missing. Drag and drop supports Matplotlib's TkAgg backend via `tkinterdnd2`, or QtAgg when a Qt binding is installed. If drag and drop is unavailable, the window shows a diagnostic and **Open CSV** remains available.

## Run

Start without an input file, then drop one CSV onto the window or click **Open CSV**:

```bash
python kalman_flight_viewer.py
```

Or open a flight directly:

```bash
python kalman_flight_viewer.py flight.csv
```

The viewer shows X/Y position, velocity, body/NED bias, and raw/corrected acceleration. Playback defaults to real time with a 10-second window and a fixed 50 ms sample interval.

UI parameters:

- Acceleration standard deviation, in m/s².
- Initial velocity standard deviation, in m/s.
- Minimum position standard deviation, in metres.
- Position error as a percentage of height: `10` means `K_height = 0.1`.
- Coordinate multiplier: scales the original `vis_x/y` before filter initialization and every measurement update.

Click **Apply + export** to recalculate the entire flight and save the result. Opening or dropping another CSV uses the current fields and starts paused at time zero. The multiplier is always applied to the original data, so repeated recalculation does not compound it.

| Control | Action |
| --- | --- |
| Space / Play / Pause | Toggle playback |
| Left / Right | Seek backward / forward one second |
| Home / End | Jump to the beginning / end |
| `+` / `-` | Increase / decrease playback speed |
| Time slider | Seek and pause |

### Open results in PlotJuggler

Click **Open in PlotJuggler** to apply the current UI parameters, refresh the exported CSV, and launch PlotJuggler with that file. The button becomes available after loading a flight.

The viewer checks an explicit `--plotjuggler` path, `PLOTJUGGLER_PATH`, the remembered executable, `PATH`, and common installation directories. On Windows it also checks the installation registry, Program Files, and per-user installations. If no executable is found, a file dialog asks you to locate the installed `plotjuggler.exe` (or the application executable/AppImage on other systems). A successful launch remembers the path for subsequent sessions. PlotJuggler must already be installed.

```bash
python kalman_flight_viewer.py --plotjuggler "C:\Program Files\PlotJuggler\plotjuggler.exe"
```

The launch uses PlotJuggler's [`-d / --datafile` option](https://github.com/PlotJuggler/PlotJuggler/blob/main/plotjuggler_app/main.cpp) with an absolute CSV path. If PlotJuggler shows its CSV import dialog, select `;` as the delimiter and `time_s` as the time column. Subsequent changes in the viewer require reopening or reloading the data in PlotJuggler.

The remembered path is stored in `KalmanFlightViewer/settings.json` under `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, or `$XDG_CONFIG_HOME` / `~/.config` on Linux. This is independent of the input CSV directory and repository.

## CSV input

Use `;` as the separator and at least two data rows. Decimal points and decimal commas are accepted. Input positions are horizontal NED coordinates; horizontal accelerations must already be in NED.

| Columns | Meaning |
| --- | --- |
| `vis_x`, `vis_y` | Measured horizontal position, metres |
| `mheight` | Height above ground, metres |
| `acc_x`, `acc_y` or `acc.x`, `acc.y` | Horizontal NED acceleration, m/s² |
| `quat.x`, `quat.y`, `quat.z`, `quat.w` | Attitude quaternion; underscore aliases are also accepted |
| `yaw` | Yaw, with degrees/radians checked automatically |

Optional diagnostics: `rawacc.x/y/z` (or underscore aliases), `acc.z` (or `acc_z`), and `G`. Quaternion direction is checked against yaw and optional raw acceleration. Use `--quat-direction body-to-ned` / `ned-to-body` or `--yaw-unit deg` / `rad` to override automatic interpretation.

The time step comes from `--dt` (default `0.05` seconds), not CSV timestamps. Malformed required values are rejected instead of silently dropping samples.

## Export and command-line options

Loading a flight or applying parameters automatically writes **`result.csv` next to the current input file**, replacing an existing output. An explicit `--export` path stays fixed when opening other flights. An empty startup window does not write a CSV.

The semicolon-separated output includes:

- Original `vis_x/y`, scaled `vis_scaled_x/y`, and `coord_scale`.
- `estimated_position_x/y` and `estimated_speed_x/y`.
- Raw/corrected acceleration, body bias, and NED bias projections.
- Position uncertainty, covariance entries, Kalman gains, quaternion/yaw diagnostics, and bias-fit accumulators.

```bash
# Set parameters and start playback at 200 seconds.
python kalman_flight_viewer.py flight.csv --start 200 --coord-scale 1.1 --k-height 0.1

# Export without a desktop window (requires an input CSV).
python kalman_flight_viewer.py flight.csv --no-gui --export estimates.csv

# Save a plot without a desktop window.
python kalman_flight_viewer.py flight.csv --no-gui --start 200 --snapshot view.png

# All available options and the model description.
python kalman_flight_viewer.py --help
```

## Estimation model

Each horizontal axis uses a position/velocity Kalman filter. Position measurement standard deviation is `max(K_height * abs(mheight), min_position_sigma)`; its square is the measurement variance. Acceleration noise is treated as an independent per-sample error held constant within each time step.

Bias is a cumulative least-squares estimate in body X/Y, outside the Kalman state. The full quaternion rotates this estimate into NED before acceleration correction. Bias fitting uses NED velocity increments and the corresponding attitude, avoiding direct subtraction of rotating-frame velocity from an integral of body acceleration.

Body-Z bias is assumed zero, vertical motion is not fused, and bias uncertainty is not included in the Kalman covariance. Initial X/Y velocities default to zero and can be set with `--vx0` / `--vy0`. Once frame conventions are determined, filtering is a forward pass; the moving display window does not reset the estimate. Further equations and assumptions are in the script's module docstring.

## Tests

```bash
python -m unittest discover -s tests -v
```

The PlotJuggler integration checks executable discovery, remembered paths, UI parameter application, export/launch errors, and argument handling. The subprocess test uses a small recording executable; it does not require PlotJuggler to be installed.
