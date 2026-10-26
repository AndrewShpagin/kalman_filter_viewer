#!/usr/bin/env python3
"""Compare C++ velocity exports against Python (NumPy required).

Fixed and variable dt call the Python viewer, whose vectorized prefix-integral
implementation is independent of the C++ streaming interval walk. Check delay
alignment, prediction and all exported diagnostics as well as the old columns.
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
    out = py.estimate_velocity(data, cfg, intervals=dt)
    return out, out['weighted_duration_s'], out['bias_response']


def compare(exe, source, work, cfg, flags=(), intervals=None):
    output = work/'result_velocity_cpp.csv'
    args = [str(exe), str(source), '--export', str(output)]
    for option, field in (('fade-in', 'fade_in'), ('flat', 'flat'), ('fade-out', 'fade_out'),
                          ('lead', 'lead'), ('vision-delay', 'vision_delay'), ('coord-scale', 'coord_scale'), ('bias-warmup', 'bias_warmup'),
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
        durations = out['weighted_duration_s']
        dt = np.r_[0., np.full(len(data)-1, cfg.dt)]
    else:
        out, durations, responses = variable_reference(data, cfg, intervals)
        dt = intervals
    expected_path = work/'reference.csv'
    py.export_velocity(expected_path, out)
    expected = np.genfromtxt(expected_path, delimiter=';', names=True)
    actual = np.genfromtxt(output, delimiter=';', names=True)
    extras = ('dt_s', 'weighted_duration_s', 'bias_response_xx', 'bias_response_xy', 'bias_response_yx', 'bias_response_yy')
    assert actual.dtype.names == expected.dtype.names
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
        for delay in (.137, 8.123):
            compare(exe, source, work, replace(cfg, vision_delay=delay))
            compare(exe, source, work, replace(cfg, vision_delay=delay), ['--dt-column', 'elapsed'], dt)
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
