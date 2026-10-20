import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

import kalman_flight_viewer as shared
import velocity_flight_viewer as viewer


def flight(acceleration, dt=0.05, initial_velocity=(0., 0.), body_bias=(0., 0.), rotating=False):
    """Exact motion with acceleration held constant over each sample interval."""
    n = len(acceleration)
    t = np.arange(n) * dt
    data = np.zeros((n, 15))
    data[:, 2] = 100.
    data[:, 8] = 1.
    data[:, 10:] = np.nan
    if rotating:
        yaw = 0.37 * t
        roll = 0.35 * np.sin(t * 0.23)
        pitch = 0.28 * np.cos(t * 0.31)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        data[:, 5:9] = np.column_stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                                       cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy))
        data[:, 9] = np.rad2deg(yaw)
    rotation, _ = shared.quaternion_matrices(data[:, 5:9])
    data[:, 3:5] = acceleration + rotation[:, :2, :2] @ np.asarray(body_bias)
    velocity = np.zeros((n, 2))
    velocity[0] = initial_velocity
    for i in range(1, n):
        velocity[i] = velocity[i - 1] + acceleration[i] * dt
        data[i, :2] = data[i - 1, :2] + (velocity[i - 1] + velocity[i]) * (dt / 2)
    return data, velocity


def config(**kwargs):
    return viewer.VelocityConfig(quaternion_direction='body-to-ned', yaw_unit='deg', **kwargs)


class VelocityModelTests(unittest.TestCase):
    def test_rotating_body_does_not_create_bias_for_constant_ned_velocity(self):
        data, truth = flight(np.zeros((1000, 2)), initial_velocity=(3., -2.), rotating=True)
        out = viewer.estimate_velocity(data, config(vx0=3., vy0=-2.))
        np.testing.assert_allclose(out['velocity'], truth, atol=2e-11)
        np.testing.assert_allclose(out['bias_body'], 0., atol=2e-11)

    def test_preserves_acceleration_with_known_body_bias_and_full_attitude(self):
        t = np.arange(1400) * 0.05
        acceleration = np.column_stack((0.4*np.sin(t/3) + 0.7*(t > 14), 0.6*np.cos(t/5)))
        data, truth = flight(acceleration, initial_velocity=(1., -0.5),
                             body_bias=(0.3, -0.2), rotating=True)
        cfg = config(learn_bias=False, bias_x0=0.3, bias_y0=-0.2, vx0=1., vy0=-0.5)
        out = viewer.estimate_velocity(data, cfg)
        # Includes a real acceleration step: fading the acceleration itself or
        # subtracting a linear speed trend would fail this physics check.
        np.testing.assert_allclose(out['velocity'], truth, atol=2e-10)
        np.testing.assert_allclose(out['corrected_acceleration'], acceleration, atol=1e-12)
        np.testing.assert_allclose(out['forward_position'], data[:, :2] + truth, atol=2e-10)

    def test_learns_stationary_body_bias_while_rotating(self):
        bias = np.array([0.3, -0.2])
        data, _ = flight(np.zeros((4000, 2)), body_bias=bias, rotating=True)
        out = viewer.estimate_velocity(data, config())
        np.testing.assert_allclose(out['bias_body'][-1], bias, atol=0.005)
        self.assertLess(np.max(np.abs(out['velocity'][-400:])), 0.02)

    def test_feedback_is_next_sample_and_identity_matches_integral_formula(self):
        t = np.arange(900) * 0.05
        acceleration = np.column_stack((np.sin(t/4), np.cos(t/7)))
        data, _ = flight(acceleration, initial_velocity=(1., -2.), body_bias=(0.15, -0.1))
        out = viewer.estimate_velocity(data, config(vx0=1., vy0=-2.))
        np.testing.assert_array_equal(out['bias_body_used'][1:], out['bias_body'][:-1])
        expected = (out['integral_raw'][1:] - out['velocity'][1:] + [1., -2.]) / out['time'][1:, None]
        np.testing.assert_allclose(out['bias_body'][1:], expected, atol=2e-12)

    def test_future_rows_cannot_change_past_estimates(self):
        t = np.arange(600) * 0.05
        data, _ = flight(np.column_stack((np.sin(t), np.cos(t))), body_bias=(0.1, 0.2), rotating=True)
        full = viewer.estimate_velocity(data, config())
        short = viewer.estimate_velocity(data[:245], config())
        for key in ('velocity', 'bias_body', 'forward_position', 'mean_visual_velocity'):
            np.testing.assert_allclose(full[key][:245], short[key], rtol=0., atol=1e-12)

    def test_matching_means_and_short_log_warmup(self):
        data, truth = flight(np.zeros((3, 2)), initial_velocity=(2., -1.))
        out = viewer.estimate_velocity(data, config(vx0=2., vy0=-1.))
        np.testing.assert_allclose(out['velocity'], truth, atol=1e-12)
        np.testing.assert_array_equal(out['window_complete'], 0.)
        np.testing.assert_allclose(out['mean_integrated_velocity'] + out['normalization_offset'],
                                   out['mean_visual_velocity'], atol=1e-12)
        data, _ = flight(np.zeros((362, 2)))
        out = viewer.estimate_velocity(data, config())
        self.assertEqual(out['window_complete'][359], 0)
        self.assertEqual(out['window_complete'][360], 1)

    def test_rectangular_and_asymmetric_windows(self):
        acceleration = np.full((240, 2), [0.2, -0.1])
        data, truth = flight(acceleration)
        for fade_in, flat, fade_out in ((0, 2, 0), (1, 2, 0), (0, 1, 2), (1, 0, 1), (0.13, 0.11, 0.17)):
            out = viewer.estimate_velocity(data, config(learn_bias=False, fade_in=fade_in,
                                                        flat=flat, fade_out=fade_out))
            np.testing.assert_allclose(out['velocity'], truth, atol=1e-11)

    def test_rejects_invalid_parameters(self):
        for changes in ({'dt': 0}, {'fade_in': -1}, {'coord_scale': 0},
                        {'lead': float('nan')}, {'fade_in': 0, 'flat': 0, 'fade_out': 0}):
            with self.assertRaises(ValueError):
                config(**changes).validate()


class VelocityUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'Політ with spaces'
        self.folder.mkdir()
        self.source = self.folder / 'flight.csv'
        self.source.write_text(
            'vis_x;vis_y;mheight;acc.x;acc.y;quat.x;quat.y;quat.z;quat.w;yaw\n'
            + ''.join(f'{i*0.05};{-i*0.1};10;0;0;0;0;0;1;0\n' for i in range(30)),
            encoding='utf-8')
        self.viewer = viewer.VelocityViewer(None, config(vx0=1., vy0=-2.), paused=True,
                                            export_path=self.folder / 'result_velocity.csv',
                                            export_follows_input=True)
        self.addCleanup(self.viewer.plt.close, self.viewer.fig)

    def test_open_export_apply_time_scale_and_mean_columns(self):
        v = self.viewer
        self.assertFalse(v.apply_button.active)
        self.assertTrue(v.load_csv(self.source))
        self.assertTrue(v.apply_button.active)
        self.assertEqual(v.export_path.name, 'result_velocity.csv')
        v.fields['coord_scale'].set_val('2')
        v.fields['dt'].set_val('0.1')
        v.fields['lead'].set_val('0.5')
        v.fields['view_window'].set_val('20')
        self.assertTrue(v.apply_parameters())
        self.assertAlmostEqual(v.duration, 2.9)
        self.assertAlmostEqual(v.slider.valmax, 2.9)
        self.assertEqual(v.window, 20)
        np.testing.assert_allclose(v.result['velocity'], [[1., -2.]]*30, atol=1e-11)
        with v.export_path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter=';'))
        self.assertEqual(len(rows), 30)
        self.assertAlmostEqual(float(rows[-1]['forward_position_x']), 2.9 + 0.5)
        self.assertAlmostEqual(float(rows[-1]['estimated_speed_x']), float(rows[-1]['niacc.x']))
        self.assertIn('mean_visual_speed_x', rows[-1])
        self.assertIn('bias_body_used_x', rows[-1])
        original = v.result
        previous_csv = v.export_path.read_bytes()
        v.fields['dt'].set_val('0')
        self.assertFalse(v.apply_parameters())
        self.assertIs(v.result, original)
        self.assertEqual(v.export_path.read_bytes(), previous_csv)

    def test_drop_then_fixed_bias_and_plotjuggler_export(self):
        v = self.viewer
        pending = []
        self.assertTrue(v.queue_dropped_files([str(self.source)], pending.append))
        self.assertIsNone(v.result)
        pending.pop()()
        self.assertIsNotNone(v.result)
        v.fields['bias_x0'].set_val('-0.1')
        v.bias_toggle.set_active(0)
        self.assertTrue(v.apply_parameters())
        np.testing.assert_allclose(v.result['bias_body'][:, 0], -0.1)
        self.assertFalse(v.cfg.learn_bias)
        binary = self.folder / 'plotjuggler'
        with patch.object(shared, 'find_plotjuggler', return_value=binary), \
                patch.object(shared, 'save_plotjuggler_path'), \
                patch.object(shared.subprocess, 'Popen') as launch:
            self.assertTrue(v.open_in_plotjuggler())
            self.assertEqual(launch.call_args.args[0], [str(binary), '-d', str(v.export_path.resolve())])
        self.assertIn('estimated_speed_x', v.export_path.read_text().splitlines()[0])

    def test_protects_input_and_failed_load_keeps_flight(self):
        v = self.viewer
        self.assertTrue(v.load_csv(self.source))
        previous = v.result
        self.assertFalse(v.load_csv(self.folder / 'missing.csv'))
        self.assertIs(v.result, previous)
        v.export_follows_input = False
        v.export_path = self.source
        before = self.source.read_bytes()
        self.assertFalse(v.load_csv(self.source))
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
