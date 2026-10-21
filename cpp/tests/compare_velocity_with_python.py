#!/usr/bin/env python3
"""Compare C++ velocity exports against Python (NumPy required).

Fixed dt calls the unmodified viewer. Variable dt uses its NumPy bias solver
and independent vectorized window integration with duration-weighted means.
No input files are modified; exports are made in a temporary directory.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import velocity_flight_viewer as py
from compare_with_python import synthetic


def variable_reference(data, cfg, dt):
    attitude = py.check_attitude(data, cfg)
    out = py.estimate_velocity(data, cfg, attitude)
    t = np.cumsum(dt.astype(np.longdouble))
    out['time'] = np.asarray(t, dtype=float)
    out['dt'] = dt.copy()
    out['window_span_s'] = np.minimum(out['time'], cfg.fit_window)
    out['window_complete'] = (t + 1e-12*cfg.fit_window >= cfg.fit_window).astype(float)
    out['integral_raw'] = np.cumsum(data[:, 3:5]*dt[:, None], axis=0)
    out['visual_velocity'][1:] = np.diff(out['measured_position'], axis=0)/dt[1:, None]
    bias = py.BodyBiasEstimator((cfg.bias_x0, cfg.bias_y0), cfg.bias_warmup, cfg.bias_memory)
    responses = np.zeros((len(data), 2, 2))
    durations = np.zeros(len(data))
    for i in range(1, len(data)):
        left = max(np.longdouble(0), t[i]-cfg.fit_window)
        start = max(1, int(np.searchsorted(t, left, side='right')))
        indices = np.arange(start, i+1)
        h = np.asarray(np.minimum(dt[indices], t[indices]-left), dtype=float)
        ages = np.asarray(t[i]-t[indices], dtype=float)+h/2
        w = py.interval_weights(ages, float(t[i]-left), cfg)
        mass = w*h
        durations[i] = mass.sum()
        out['weight_sum'][i] = w.sum()
        b = bias.bias.copy()
        m = attitude.rotation[i, :2, :2]
        projections = attitude.rotation[indices, :2, :2]
        rotation = np.concatenate((np.zeros((1, 2, 2)), np.cumsum(projections*h[:, None, None], axis=0)))
        integral = np.vstack((np.zeros(2), np.cumsum((data[indices, 3:5]-projections@b)*h[:, None], axis=0)))
        delta_pos = out['measured_position'][indices]-out['measured_position'][indices-1]
        mean_visual = np.einsum('i,ij->j', w*(h/dt[indices]), delta_pos)/mass.sum()
        mean_inertial = np.einsum('i,ij->j', mass, (integral[:-1]+integral[1:])/2)/mass.sum()
        response = rotation[-1]-np.einsum('i,ijk->jk', mass, (rotation[:-1]+rotation[1:])/2)/mass.sum()
        out['bias_body_used'][i] = b
        out['bias_used'][i] = m@b
        out['corrected_acceleration'][i] = data[i, 3:5]-m@b
        out['window_integral'][i] = integral[-1]
        out['mean_visual_velocity'][i] = mean_visual
        out['mean_integrated_velocity'][i] = mean_inertial
        out['normalization_offset'][i] = mean_visual-mean_inertial
        out['velocity'][i] = integral[-1]+mean_visual-mean_inertial
        if cfg.learn_bias:
            h_fit = m-(response-responses[i-1])/dt[i]
            residual = (data[i, 3:5]*dt[i]-(out['velocity'][i]-out['velocity'][i-1])
                        -response@b+responses[i-1]@out['bias_body_used'][i-1])
            bias.update_residual(h_fit, residual, dt[i])
        out['bias_body'][i] = bias.bias
        out['bias'][i] = m@bias.bias
        out['bias_rhs'][i] = bias.rhs
        out['bias_normal'][i] = bias.normal[0, 0], bias.normal[0, 1], bias.normal[1, 1]
        responses[i] = response
    out['forward_position'] = out['measured_position']+cfg.lead*out['velocity']
    return out, durations, responses


def compare(exe, source, work, cfg, flags=(), intervals=None):
    output = work/'result_velocity_cpp.csv'
    args = [str(exe), str(source), '--export', str(output)]
    for option, field in (('fade-in', 'fade_in'), ('flat', 'flat'), ('fade-out', 'fade_out'),
                          ('lead', 'lead'), ('coord-scale', 'coord_scale'), ('bias-warmup', 'bias_warmup'),
                          ('bias-memory', 'bias_memory'),
                          ('bias-x0', 'bias_x0'), ('bias-y0', 'bias_y0'), ('vx0', 'vx0'), ('vy0', 'vy0'),
                          ('quat-direction', 'quaternion_direction'), ('yaw-unit', 'yaw_unit')):
        args += ['--'+option, str(getattr(cfg, field))]
    if not cfg.learn_bias:
        args += ['--no-learn-bias']
    if intervals is None:
        args += ['--dt', str(cfg.dt)]
    completed = subprocess.run(args+list(flags), capture_output=True, text=True)
    if completed.returncode:
        raise AssertionError(completed.stdout+completed.stderr)
    data, _ = py.read_flight(source)
    responses = None
    if intervals is None:
        out = py.estimate_velocity(data, cfg)
        durations = cfg.dt*out['weight_sum']
        dt = np.r_[0., np.full(len(data)-1, cfg.dt)]
    else:
        out, durations, responses = variable_reference(data, cfg, intervals)
        dt = intervals
    expected_path = work/'reference.csv'
    py.export_velocity(expected_path, out)
    expected = np.genfromtxt(expected_path, delimiter=';', names=True)
    actual = np.genfromtxt(output, delimiter=';', names=True)
    extras = ('dt_s', 'weighted_duration_s', 'bias_response_xx', 'bias_response_xy', 'bias_response_yx', 'bias_response_yy')
    assert actual.dtype.names == expected.dtype.names+extras
    for name in expected.dtype.names:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-9, atol=1e-9, err_msg=name)
    np.testing.assert_allclose(actual['dt_s'], dt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual['weighted_duration_s'], durations, rtol=1e-10, atol=1e-10)
    if responses is not None:
        for field, (j, k) in zip(extras[2:], ((0, 0), (0, 1), (1, 0), (1, 1))):
            np.testing.assert_allclose(actual[field], responses[:, j, k], rtol=1e-9, atol=1e-9)
    error = max(float(np.max(np.abs(actual[k]-expected[k]))) for k in expected.dtype.names)
    print(f'PASS {source.name}: {len(data)} rows, {len(expected.dtype.names)} shared columns, '
          f'{"fixed" if intervals is None else "variable"} dt; max error {error:.3g}')


def rejections(exe, source, work):
    output = work/'protected.csv'
    output.write_text('previous valid output', encoding='utf-8')
    for flags in (['--dt', '0'], ['--fade-in', '-1'], ['--bias-warmup', '-1'], ['--bias-memory', '-1'],
                  ['--fade-in', '0', '--flat', '0', '--fade-out', '0'],
                  ['--time-column', 'timestamp', '--dt', '.05'],
                  ['--time-column', 'pos_sigma'], ['--dt-column', 'acc_y']):
        p = subprocess.run([str(exe), str(source), '--export', str(output), *flags], capture_output=True)
        assert p.returncode != 0, flags
        assert output.read_text(encoding='utf-8') == 'previous valid output'
    original = source.read_bytes()
    p = subprocess.run([str(exe), str(source), '--export', str(source)], capture_output=True)
    assert p.returncode != 0 and source.read_bytes() == original
    print('PASS invalid parameters/time and input-output collision preserve existing files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--time-column')
    parser.add_argument('--time-scale', type=float, default=1)
    args = parser.parse_args()
    if args.time_column and not args.csv:
        parser.error('--time-column requires --csv')
    exe = args.exe.resolve()
    with tempfile.TemporaryDirectory(prefix='velocity_cpp_') as directory:
        work = Path(directory)
        source = work/'політ with spaces.csv'
        dt, _, _ = synthetic(source)
        compare(exe, source, work, py.VelocityConfig())
        cfg = py.VelocityConfig(fade_in=1.2, flat=2.4, fade_out=.8, coord_scale=1.12,
                               bias_x0=-.1, bias_y0=.07, bias_warmup=.6, vx0=.7, vy0=-.4, lead=.7)
        compare(exe, source, work, cfg)
        compare(exe, source, work, replace(cfg, bias_warmup=0))
        compare(exe, source, work, replace(cfg, bias_memory=0))
        compare(exe, source, work, replace(cfg, learn_bias=False))
        compare(exe, source, work, cfg, ['--dt-column', 'elapsed'], dt)
        compare(exe, source, work, cfg, ['--time-column', 'timestamp', '--time-scale', '.001'], dt)
        compare(exe, source, work, replace(cfg, fade_in=0, flat=.13, fade_out=0), ['--dt-column', 'elapsed'], dt)
        synthetic(source, inverse=True)
        compare(exe, source, work, cfg)
        compare(exe, source, work, replace(cfg, quaternion_direction='ned-to-body', yaw_unit='rad'))
        rejections(exe, source, work)
        if args.csv:
            compare(exe, args.csv.resolve(), work, py.VelocityConfig())
            if args.time_column:
                with args.csv.open(encoding='utf-8-sig', newline='') as f:
                    t = np.array([np.longdouble(row[args.time_column].replace(',', '.'))
                                  for row in csv.DictReader(f, delimiter=';')])
                dt = np.r_[0., np.asarray(np.diff(t)*args.time_scale, dtype=float)]
                compare(exe, args.csv.resolve(), work, py.VelocityConfig(),
                        ['--time-column', args.time_column, '--time-scale', str(args.time_scale)], dt)


if __name__ == '__main__':
    main()
