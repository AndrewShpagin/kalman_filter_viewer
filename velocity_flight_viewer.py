#!/usr/bin/env python3
"""Experimental window-normalized inertial velocity with a body-frame bias.

Run: python velocity_flight_viewer.py [flight.csv]
     python velocity_flight_viewer.py flight.csv --vision-delay .2 \
         --time-column timestamp --time-scale .001
Export: add --no-gui. Default result_velocity.csv; separator ';'.

Raw acceleration is gravity-compensated NED, before our bias subtraction.
Quaternion xyzw rotates BODY to NED (or explicitly select the inverse).
Vision received at t represents t - vision_delay. Align its window with
historical IMU intervals and quaternion projections. Integrate acceleration
without fade, then match its duration-weighted mean velocity to visual motion.
Learn the next BODY bias from differences at aligned vision epochs, including
the window's bias response C. Predict velocity from the vision epoch to now
using the same previous BODY bias, rotated by each historical quaternion.

Before IMU/vision histories overlap, integrate from the configured initial
velocity at time zero and do not learn bias. No pre-log IMU is invented.
forward_position = scaled_visual_position + lead * velocity_now; this does not
also advance the visual position by vision_delay. Lead and delay are distinct.

The default time base is fixed --dt .05. --time-column or --dt-column enables
real elapsed times; --time-scale converts their units to seconds. Non-increasing
or invalid times are rejected. See VELOCITY_VIEWER.md and cpp/VELOCITY_ALGORITHM_UK.md.

The GUI can start empty, open/drop CSV, apply edited parameters, export and open
in PlotJuggler. Editing fields pauses replay; only Apply recomputes the flight.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

from kalman_flight_viewer import (
    Attitude, BodyBiasEstimator, FlightViewer, check_attitude, read_flight,
    wrap_degrees,
)


@dataclass
class VelocityConfig:
    dt: float = 0.05
    fade_in: float = 6.0
    flat: float = 6.0
    fade_out: float = 6.0
    lead: float = 1.0
    vision_delay: float = 0.0
    time_column: str = ''
    dt_column: str = ''
    time_scale: float = 1.0
    coord_scale: float = 1.0
    view_window: float = 10.0
    bias_x0: float = 0.0
    bias_y0: float = 0.0
    bias_warmup: float = 6.0
    bias_memory: float = 60.0
    vx0: float = 0.0
    vy0: float = 0.0
    learn_bias: bool = True
    quaternion_direction: str = 'auto'
    yaw_unit: str = 'auto'

    @property
    def fit_window(self):
        return self.fade_in + self.flat + self.fade_out

    def validate(self):
        for name in ('dt', 'coord_scale', 'view_window', 'time_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name}: must be finite and > 0.')
        for name in ('fade_in', 'flat', 'fade_out', 'lead', 'vision_delay', 'bias_warmup', 'bias_memory'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: must be finite and >= 0.')
        for name in ('bias_x0', 'bias_y0', 'vx0', 'vy0'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'{name}: must be finite.')
        if not math.isfinite(self.fit_window) or self.fit_window <= 0 or not math.isfinite(self.fit_window + self.vision_delay):
            raise ValueError('Window must be positive; window + delay must be finite.')
        if self.time_column and self.dt_column:
            raise ValueError('Choose time_column OR dt_column.')
        if self.quaternion_direction not in ('auto', 'body-to-ned', 'ned-to-body'):
            raise ValueError('Invalid quaternion direction.')
        if self.yaw_unit not in ('auto', 'deg', 'rad'):
            raise ValueError('Invalid yaw unit.')


def interval_weights(age: np.ndarray, span: float, cfg: VelocityConfig) -> np.ndarray:
    """Trapezoid from oldest -> newest: fade-in, flat, fade-out.

    During startup span is shortened, retaining the edge slopes; if the two
    fades overlap, the weights form a triangle. Normalize by actual sum later.
    """
    weights = np.ones_like(age)
    if cfg.fade_in > 0:
        weights = np.minimum(weights, (span - age) / cfg.fade_in)
    if cfg.fade_out > 0:
        weights = np.minimum(weights, age / cfg.fade_out)
    return np.where((age >= 0) & (age <= span), np.maximum(weights, 0), 0)


def read_velocity_flight(path: Path, cfg: VelocityConfig):
    """Keep the shared sensor reader; optionally append actual elapsed times."""
    data, mapping = read_flight(path)
    column = cfg.time_column or cfg.dt_column
    if not column:
        return data, mapping
    values = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter=';')
        header = [s.strip() for s in next(reader)]
        if header.count(column) != 1:
            raise ValueError(f'Time column must occur exactly once: {column}')
        index = header.index(column)
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                value = np.longdouble(row[index].strip().replace(',', '.'))
            except (ValueError, IndexError):
                raise ValueError(f'CSV line {reader.line_num}: invalid {column}') from None
            if not np.isfinite(value):
                raise ValueError(f'CSV line {reader.line_num}: non-finite {column}')
            values.append(value)
    if len(values) != len(data):
        raise ValueError('Sensor/time row count mismatch.')
    values = np.asarray(values, dtype=np.longdouble)
    intervals = (np.r_[0, np.diff(values)] if cfg.time_column else values) * cfg.time_scale
    intervals[0] = 0  # first row initializes, never integrates
    if np.any(intervals[1:] <= 0):
        raise ValueError('Time must strictly increase; dt must be positive after row zero.')
    return np.column_stack((data, np.asarray(intervals, dtype=float))), mapping


def estimate_velocity(data: np.ndarray, cfg: VelocityConfig,
                      attitude: Attitude | None = None,
                      intervals: np.ndarray | None = None) -> dict[str, np.ndarray]:
    cfg.validate()
    if data.ndim != 2 or len(data) < 2 or data.shape[1] < 10:
        raise ValueError('At least two rows of position, acceleration and attitude are required.')
    if not np.isfinite(data[:, :10]).all():
        raise ValueError('Required input values must be finite.')
    if attitude is None:
        attitude = check_attitude(data, cfg)
    n = len(data)
    if attitude.rotation.shape != (n, 3, 3) or not np.isfinite(attitude.rotation).all():
        raise ValueError('Attitude rotations do not match the flight.')
    variable_time = intervals is not None or bool(cfg.time_column or cfg.dt_column)
    if intervals is None:
        if cfg.time_column or cfg.dt_column:
            if data.shape[1] < 16:
                raise ValueError('Load this CSV with read_velocity_flight to use a time column.')
            intervals = data[:, 15]
        else:
            intervals = np.r_[0., np.full(n-1, cfg.dt)]
    dt = np.asarray(intervals, dtype=float)
    if dt.shape != (n,) or not np.isfinite(dt).all() or dt[0] != 0 or np.any(dt[1:] <= 0):
        raise ValueError('Intervals must be finite: row zero = 0, remaining rows > 0.')
    t = np.cumsum(dt.astype(np.longdouble))
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or not np.isfinite(t.astype(float)).all():
        raise ValueError('Elapsed time is outside representable range.')
    epoch = t - cfg.vision_delay
    measured = data[:, :2] * cfg.coord_scale
    raw_acc = data[:, 3:5]
    matrices = attitude.rotation[:, :2, :2]
    raw_integral = np.cumsum(raw_acc * dt[:, None], axis=0)
    rotation_integral = np.cumsum(matrices * dt[:, None, None], axis=0)
    visual_velocity = np.empty((n, 2))
    visual_velocity[0] = cfg.vx0, cfg.vy0
    visual_velocity[1:] = np.diff(measured, axis=0) / dt[1:, None]
    span = np.asarray(np.clip(epoch, 0, cfg.fit_window), dtype=float)
    out = {
        'time': np.asarray(t, dtype=float), 'height': data[:, 2],
        'dt_s': dt.copy(), 'dt': dt.copy() if variable_time else np.full(n, cfg.dt),
        'vision_time_s': np.asarray(epoch, dtype=float),
        'alignment_ready': (epoch >= 0).astype(float), 'bias_fit_dt_s': np.zeros(n),
        'raw_position': data[:, :2], 'measured_position': measured,
        'raw_acceleration': raw_acc, 'visual_velocity': visual_velocity,
        'integral_raw': raw_integral, 'quaternion': attitude.quaternion,
        'yaw_csv_deg': attitude.yaw_csv_deg, 'yaw_quat_deg': attitude.yaw_quat_deg,
        'yaw_error_deg': wrap_degrees(attitude.yaw_quat_deg - attitude.yaw_csv_deg),
        'window_span_s': span,
        'window_complete': (span + 1e-12*cfg.fit_window >= cfg.fit_window).astype(float),
        'weight_sum': np.zeros(n), 'weighted_duration_s': np.zeros(n),
        'bias_normal': np.zeros((n, 3)), 'bias_response': np.zeros((n, 2, 2)),
    }
    for key in ('velocity', 'delayed_velocity', 'prediction_delta_velocity', 'forward_position',
                'mean_visual_velocity', 'mean_integrated_velocity', 'window_integral',
                'normalization_offset', 'corrected_acceleration', 'bias_body',
                'bias_body_used', 'bias', 'bias_used', 'bias_rhs'):
        out[key] = np.zeros((n, 2))
    for name in ('fade_in', 'flat', 'fade_out', 'lead', 'vision_delay', 'coord_scale',
                 'bias_x0', 'bias_y0', 'bias_warmup', 'bias_memory', 'vx0', 'vy0', 'learn_bias'):
        out[name] = np.full(n, float(getattr(cfg, name)))
    initial_v = np.array([cfg.vx0, cfg.vy0])
    for key in ('velocity', 'delayed_velocity', 'mean_visual_velocity', 'normalization_offset'):
        out[key][0] = initial_v
    bias_fit = BodyBiasEstimator((cfg.bias_x0, cfg.bias_y0), cfg.bias_warmup, cfg.bias_memory)
    out['bias_body'][0] = out['bias_body_used'][0] = bias_fit.bias
    out['bias'][0] = out['bias_used'][0] = matrices[0] @ bias_fit.bias
    out['corrected_acceleration'][0] = raw_acc[0] - out['bias_used'][0]

    def prefixes_at(times, upto):
        # Right-end sample represents (previous_time, current_time]. Clip the
        # lookup to rows already received: processing is causal, even offline.
        indices = np.searchsorted(t[:upto+1], times, side='left').clip(1, upto)
        h = np.asarray(times - t[indices-1], dtype=float)
        return (raw_integral[indices-1] + raw_acc[indices]*h[..., None],
                rotation_integral[indices-1] + matrices[indices]*h[..., None, None])

    for i in range(1, n):
        b = bias_fit.bias.copy()
        m = matrices[i]
        out['bias_body_used'][i] = b
        out['bias_used'][i] = m @ b
        out['corrected_acceleration'][i] = raw_acc[i] - m @ b
        end = epoch[i]
        left = max(np.longdouble(0), end-cfg.fit_window)
        if end > left:
            # Union of IMU and shifted vision boundaries. Within each piece
            # acceleration/attitude and interpolated visual speed are constant.
            ia, ib = np.searchsorted(t[:i+1], [left, end], side='right')
            va, vb = np.searchsorted(epoch[:i+1], [left, end], side='right')
            edges = np.unique(np.r_[left, t[ia:ib], epoch[va:vb], end])
            h = np.asarray(np.diff(edges), dtype=float)
            mid = (edges[1:] + edges[:-1])/2
            w = interval_weights(np.asarray(end-mid, dtype=float), float(end-left), cfg)
            mass = w*h
            total = float(mass.sum())
            if not total > 0:
                raise ValueError('Window has no positively weighted sample intervals.')
            a, r = prefixes_at(edges, i)
            a = a-a[0]; r = r-r[0]
            integral = a-r@b
            vi = np.searchsorted(epoch[:i+1], mid, side='right')
            dp = measured[vi]-measured[vi-1]
            mean_visual = np.einsum('i,ij->j', w*(h/dt[vi]), dp)/total
            mean_inertial = np.einsum('i,ij->j', mass, (integral[:-1]+integral[1:])/2)/total
            response = r[-1]-np.einsum('i,ijk->jk', mass, (r[:-1]+r[1:])/2)/total
            out['weight_sum'][i] = w.sum()
            out['weighted_duration_s'][i] = total
            out['window_integral'][i] = integral[-1]
            out['mean_visual_velocity'][i] = mean_visual
            out['mean_integrated_velocity'][i] = mean_inertial
            out['normalization_offset'][i] = mean_visual-mean_inertial
            out['delayed_velocity'][i] = integral[-1]+mean_visual-mean_inertial
            out['bias_response'][i] = response
            if cfg.learn_bias and out['alignment_ready'][i-1]:
                a_step, r_step = prefixes_at(np.array([epoch[i-1], end]), i)
                da, dr = a_step[1]-a_step[0], r_step[1]-r_step[0]
                previous_response = out['bias_response'][i-1]
                delta_v = out['delayed_velocity'][i]-out['delayed_velocity'][i-1]
                effective_projection = (dr-response+previous_response)/dt[i]
                residual = (da-delta_v-response@b
                            + previous_response@out['bias_body_used'][i-1])
                bias_fit.update_residual(effective_projection, residual, dt[i])
                out['bias_fit_dt_s'][i] = dt[i]
        else:
            for key in ('delayed_velocity', 'mean_visual_velocity', 'normalization_offset'):
                out[key][i] = initial_v
        # Prediction uses the SAME previous bias as the aligned window. The
        # newly learned bias becomes available to the next update only.
        a_pred, r_pred = prefixes_at(np.array([max(np.longdouble(0), end), t[i]]), i)
        prediction = a_pred[1]-a_pred[0]-(r_pred[1]-r_pred[0])@b
        out['prediction_delta_velocity'][i] = prediction
        out['velocity'][i] = out['delayed_velocity'][i]+prediction
        out['bias_body'][i] = bias_fit.bias
        out['bias'][i] = m@bias_fit.bias
        normal = bias_fit.normal
        out['bias_normal'][i] = normal[0, 0], normal[0, 1], normal[1, 1]
        out['bias_rhs'][i] = bias_fit.rhs
    out['forward_position'] = measured + cfg.lead*out['velocity']
    for key, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Estimate became non-finite: {key}; check input and parameters.')
    return out


def export_velocity(path: Path, result: dict[str, np.ndarray]) -> None:
    """Atomic semicolon export, including enough intermediates to audit the fit."""
    path = Path(path)
    names = ['time_s', 'mheight']
    columns = [result['time'], result['height']]
    for key in ('dt', 'fade_in', 'flat', 'fade_out', 'lead', 'coord_scale',
                'window_span_s', 'window_complete', 'weight_sum', 'learn_bias',
                'bias_x0', 'bias_y0', 'bias_warmup', 'bias_memory', 'vx0', 'vy0'):
        names.append(key)
        columns.append(result[key])
    for key, prefix in (
        ('raw_position', 'vis'), ('measured_position', 'vis_scaled'),
        ('velocity', 'estimated_speed'), ('forward_position', 'forward_position'),
        ('visual_velocity', 'visual_interval_speed'),
        ('mean_visual_velocity', 'mean_visual_speed'),
        ('mean_integrated_velocity', 'mean_inertial_speed'),
        ('window_integral', 'window_integral_acc'),
        ('normalization_offset', 'velocity_offset'),
        ('raw_acceleration', 'acc_raw'), ('corrected_acceleration', 'acc_used'),
        ('integral_raw', 'integral_acc_raw'),
        ('bias_body_used', 'bias_body_used'), ('bias_body', 'bias_body_estimated'),
        ('bias_used', 'bias_ned_used'), ('bias', 'bias_ned_estimated'),
        ('bias_rhs', 'bias_fit_rhs'),
    ):
        for axis, suffix in enumerate(('x', 'y')):
            names.append(prefix + '_' + suffix)
            columns.append(result[key][:, axis])
    for axis, suffix in enumerate(('x', 'y')):
        names.append('niacc.' + suffix)
        columns.append(result['velocity'][:, axis])
    for j, axis in enumerate(('x', 'y', 'z', 'w')):
        names.append('quat_' + axis + '_normalized_input')
        columns.append(result['quaternion'][:, j])
    for field in ('yaw_csv_deg', 'yaw_quat_deg', 'yaw_error_deg'):
        names.append(field)
        columns.append(result[field])
    for j, suffix in enumerate(('xx', 'xy', 'yy')):
        names.append('bias_fit_normal_' + suffix)
        columns.append(result['bias_normal'][:, j])
    for name in ('dt_s', 'weighted_duration_s'):
        names.append(name); columns.append(result[name])
    for j, k, suffix in ((0, 0, 'xx'), (0, 1, 'xy'), (1, 0, 'yx'), (1, 1, 'yy')):
        names.append('bias_response_' + suffix); columns.append(result['bias_response'][:, j, k])
    for name in ('vision_delay', 'vision_time_s', 'alignment_ready', 'bias_fit_dt_s'):
        names.append(name); columns.append(result[name])
    for key, prefix in (('delayed_velocity', 'delayed_speed'), ('prediction_delta_velocity', 'prediction_delta_v')):
        for j, axis in enumerate(('x', 'y')):
            names.append(prefix + '_' + axis); columns.append(result[key][:, j])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(names)
            writer.writerows(zip(*columns))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class VelocityViewer(FlightViewer):
    viewer_name = 'Inertial velocity viewer'
    replay_name = 'Windowed velocity replay'
    default_export_name = 'result_velocity.csv'
    panel_titles = ('Vision / forward position', 'Velocity', 'Bias', 'Acceleration')

    def __init__(self, *args, **kwargs):
        from matplotlib.widgets import CheckButtons
        super().__init__(*args, **kwargs)
        self.bias_toggle = CheckButtons(self.fig.add_axes([0.815, 0.735, 0.16, 0.035]),
                                       ['Learn body bias'], [self.cfg.learn_bias])
        self.bias_toggle.on_clicked(self.parameters_edited)

    def parameter_specs(self, cfg):
        return (
            ('fade_in', 'Old-edge fade (s)', cfg.fade_in),
            ('flat', 'Flat part (s)', cfg.flat),
            ('fade_out', 'Recent-edge fade (s)', cfg.fade_out),
            ('lead', 'Forward time (s)', cfg.lead),
            ('vision_delay', 'Vision delay (s)', cfg.vision_delay),
            ('coord_scale', 'Coordinate multiplier', cfg.coord_scale),
            ('dt', 'Fixed step (s)', cfg.dt),
            ('view_window', 'Plot window (s)', cfg.view_window),
            ('bias_x0', 'Initial body bias X', cfg.bias_x0),
            ('bias_y0', 'Initial body bias Y', cfg.bias_y0),
            ('bias_warmup', 'Bias warmup (s)', cfg.bias_warmup),
            ('bias_memory', 'Bias memory (s; 0=all)', cfg.bias_memory),
        )

    def plot_groups(self, axis):
        groups = list(super().plot_groups(axis))
        groups[0] = [('measured_position', 'Vision x multiplier', '#7e8793', '--'),
                     ('forward_position', 'Vision + lead * speed', '#0072B2', '-')]
        groups[1] = [('velocity', 'Speed now', '#009E73', '-'),
                     ('delayed_velocity', 'Speed at vision time', '#E69F00', ':'),
                     ('mean_visual_velocity', 'Window mean visual speed', '#CC79A7', '--')]
        return groups

    def parameter_text(self):
        c = self.cfg
        return (f'Active: window {c.fade_in:g} + {c.flat:g} + {c.fade_out:g} = {c.fit_window:g} s; '
                f'lead={c.lead:g} s; delay={c.vision_delay:g} s; scale={c.coord_scale:g}; '
                f'time={c.time_column or c.dt_column or str(c.dt)+" s fixed"}'
                f'{" x "+str(c.time_scale)+" s" if c.time_column or c.dt_column else ""}; '
                f'view={self.window:g} s; body bias: {"learning" if c.learn_bias else "fixed"}; '
                f'bias warmup={c.bias_warmup:g} s; bias memory={c.bias_memory:g} s (0=all).')

    def status_details(self, index):
        r = self.result
        state = ('waiting for aligned history' if not r['alignment_ready'][index] else
                 'full window' if r['window_complete'][index] else 'warmup')
        vx, vy = r['velocity'][index]
        return f'vx={vx:.3f}, vy={vy:.3f} m/s; {state} ({r["window_span_s"][index]:.2f} s)'

    def config_from_fields(self):
        values = {}
        for key, widget in self.fields.items():
            try:
                values[key] = float(widget.text.strip().replace(',', '.'))
            except ValueError:
                raise ValueError(f'{key}: enter a number.') from None
        values['learn_bias'] = (self.bias_toggle.get_status()[0] if hasattr(self, 'bias_toggle')
                                else self.cfg.learn_bias)
        cfg = replace(self.cfg, **values)
        cfg.validate()
        return cfg

    def read_input(self, path, cfg):
        return read_velocity_flight(path, cfg)

    def run_estimate(self, data, cfg, attitude):
        return estimate_velocity(data, cfg, attitude)

    def save_estimates(self, path, result):
        export_velocity(path, result)

    def after_apply(self):
        self.window = self.cfg.view_window


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, nargs='?', help='CSV; optional in GUI mode')
    for name, default, help_text in (
        ('dt', None, 'Fixed step in seconds (default .05); mutually exclusive with time/dt column'),
        ('fade-in', 6., 'Fade from zero at the oldest edge, seconds'),
        ('flat', 6., 'Full-weight part of the window, seconds'),
        ('fade-out', 6., 'Fade to zero at the current edge, seconds'),
        ('lead', 1., 'Forward position = scaled vision + this many seconds * current speed'),
        ('vision-delay', 0., 'Vision capture age relative to IMU, seconds'),
        ('time-scale', 1., 'Convert time/dt column to seconds (milliseconds: .001)'),
        ('coord-scale', 1., 'Multiplier on original vis_x/y'),
        ('bias-x0', 0., 'Initial body X accelerometer bias, m/s^2'),
        ('bias-y0', 0., 'Initial body Y accelerometer bias, m/s^2'),
        ('bias-warmup', 6., 'Minimum accumulated information time for body bias, seconds; 0 = immediate fit'),
        ('bias-memory', 60., 'Exponential memory of body bias, seconds; 0 = whole flight'),
        ('vx0', 0., 'Initial NED X velocity, m/s'),
        ('vy0', 0., 'Initial NED Y velocity, m/s'),
    ):
        parser.add_argument('--' + name, type=float, default=default, help=help_text)
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--time-column', default='', help='Strictly increasing CSV timestamp column')
    time_group.add_argument('--dt-column', default='', help='CSV elapsed-interval column')
    parser.add_argument('--window', dest='view_window', type=float, default=10.,
                        help='Visible plot duration in seconds; independent of the estimation window')
    parser.add_argument('--fixed-bias', dest='learn_bias', action='store_false',
                        help='Keep --bias-x0/y0 fixed instead of learning from estimated speed')
    parser.add_argument('--quat-direction', dest='quaternion_direction',
                        choices=('auto', 'body-to-ned', 'ned-to-body'), default='auto')
    parser.add_argument('--yaw-unit', choices=('auto', 'deg', 'rad'), default='auto')
    parser.add_argument('--playback-speed', type=float, default=1.)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--export', type=Path,
                        help='Fixed output path; otherwise result_velocity.csv next to the input')
    parser.add_argument('--plotjuggler', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--no-gui', action='store_true')
    args = parser.parse_args(argv)
    if args.dt is not None and (args.time_column or args.dt_column):
        parser.error('Choose --dt OR --time-column OR --dt-column')
    if args.time_scale != 1 and not (args.time_column or args.dt_column):
        parser.error('--time-scale requires --time-column or --dt-column')
    if args.dt is None:
        args.dt = .05
    cfg = VelocityConfig(**{key: getattr(args, key) for key in VelocityConfig.__dataclass_fields__})
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.no_gui and args.csv is None:
        parser.error('--no-gui requires an input CSV')
    if not math.isfinite(args.start) or args.start < 0:
        parser.error('--start must be finite and >= 0')
    if not math.isfinite(args.playback_speed) or not 0.0625 <= args.playback_speed <= 64:
        parser.error('--playback-speed must be between 0.0625 and 64')
    args.export_follows_input = args.export is None
    if args.export is None:
        args.export = (args.csv.parent if args.csv else Path.cwd()) / VelocityViewer.default_export_name
    paths = [p.resolve() for p in (args.csv, args.export, args.snapshot) if p is not None]
    if len(set(paths)) != len(paths):
        parser.error('Input, export and snapshot paths must be different')
    return args, cfg


def main(argv=None):
    args, cfg = parse_args(argv)
    try:
        data = attitude = result = None
        if args.csv is not None:
            data, _ = read_velocity_flight(args.csv, cfg)
            attitude = check_attitude(data, cfg)
            report = attitude.report
            print(f'Quaternion: {report["direction"]}; yaw: {report["yaw_unit"]}; '
                  f'yaw RMS: {report["yaw_rms_deg"]:.6f} deg')
            for warning in report['warnings']:
                print('Frame check: ' + warning, file=sys.stderr)
            result = estimate_velocity(data, cfg, attitude)
            export_velocity(args.export, result)
            print(f'{len(data)} samples; {result["time"][-1]:.2f} s; '
                  f'time={cfg.time_column or cfg.dt_column or str(cfg.dt)+" s fixed"}; '
                  f'window={cfg.fade_in:g}/{cfg.flat:g}/{cfg.fade_out:g} s; lead={cfg.lead:g} s; delay={cfg.vision_delay:g} s')
            print(f'Final body bias: {result["bias_body"][-1]} m/s^2. Exported: {args.export}')
        if not args.no_gui or args.snapshot is not None:
            import matplotlib
            if args.no_gui:
                matplotlib.use('Agg')
            elif matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template'):
                raise RuntimeError('No GUI backend. Use a desktop with Tk/Qt, or --no-gui.')
            viewer = VelocityViewer(result, cfg, window=cfg.view_window,
                                    playback_speed=args.playback_speed, start=args.start,
                                    paused=args.paused or args.no_gui,
                                    title=str(args.csv) if args.csv else 'Flight', data=data,
                                    attitude=attitude, input_path=args.csv, export_path=args.export,
                                    export_follows_input=args.export_follows_input,
                                    plotjuggler_path=args.plotjuggler)
            if args.snapshot is not None:
                viewer.fig.savefig(args.snapshot, dpi=130)
                print(f'Snapshot: {args.snapshot}')
            if not args.no_gui:
                viewer.show()
            else:
                viewer.plt.close(viewer.fig)
    except (ValueError, OSError, RuntimeError, OverflowError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
