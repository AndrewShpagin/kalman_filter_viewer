"""Exercise real Matplotlib input events, not just programmatic set_val()."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from matplotlib.backend_bases import KeyEvent, MouseEvent, ResizeEvent
import numpy as np

import kalman_flight_viewer as kalman
import velocity_flight_viewer as velocity


class ParameterEntryTests(unittest.TestCase):
    def make_viewer(self, module, loaded=True):
        cfg_class = velocity.VelocityConfig if module is velocity else kalman.Config
        viewer_class = velocity.VelocityViewer if module is velocity else kalman.FlightViewer
        estimate = velocity.estimate_velocity if module is velocity else kalman.estimate
        cfg = cfg_class(quaternion_direction='body-to-ned', yaw_unit='deg')
        data = np.zeros((80, 15)) if loaded else None
        if loaded:
            data[:, 0] = np.arange(len(data)) * .05
            data[:, 2] = 100
            data[:, 8] = 1
            data[:, 10:] = np.nan
        result = estimate(data, cfg) if loaded else None
        v = viewer_class(result, cfg, data=data, paused=True, start=2)
        self.addCleanup(v.plt.close, v.fig)
        v.fig.canvas.draw()
        return v

    def click(self, canvas, ax):
        x, y = ax.bbox.get_points().mean(axis=0)
        for kind in ('motion_notify_event', 'button_press_event', 'button_release_event'):
            canvas.callbacks.process(kind, MouseEvent(kind, canvas, x, y, button=1))

    def keys(self, canvas, keys):
        for key in keys:
            canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', canvas, key))

    def test_typing_and_focus_do_not_redraw_plots_or_recompute(self):
        for module in (velocity, kalman):
            with self.subTest(viewer=module.__name__):
                v = self.make_viewer(module)
                canvas = v.fig.canvas
                field = v.fields['coord_scale']
                original = v.result
                clock = v.current_time
                with patch.object(canvas, 'draw', wraps=canvas.draw) as draw, \
                        patch.object(v, 'draw', wraps=v.draw) as plots, \
                        patch.object(v, 'run_estimate') as estimate, \
                        patch.object(v, 'save_estimates') as export:
                    self.click(canvas, field.ax)
                    self.keys(canvas, ['end', 'backspace', '0', ',', '8', 'left', 'delete', '9'])
                    self.assertEqual(field.text, '0,9')
                    self.assertEqual(v.config_from_fields().coord_scale, .9)
                    self.keys(canvas, ['enter'])
                    self.click(canvas, v.axes[0, 0])
                    self.assertFalse(field.capturekeystrokes)
                    self.assertFalse(field.cursor.get_visible())
                    draw.assert_not_called()
                    plots.assert_not_called()
                    estimate.assert_not_called()
                    export.assert_not_called()
                self.assertIs(v.result, original)
                self.assertEqual(v.current_time, clock)
                self.assertTrue(v.paused)
                self.assertIn('Parameters edited', v.message.get_text())

    def test_partial_repaint_matches_full_draw_after_clear_and_resize(self):
        v = self.make_viewer(velocity)
        canvas = v.fig.canvas
        field = v.fields['coord_scale']
        for resize in (False, True):
            if resize:
                v.fig.set_size_inches(14, 9)
                canvas.callbacks.process('resize_event', ResizeEvent('resize_event', canvas))
                canvas.draw()
            self.click(canvas, field.ax)
            self.keys(canvas, ['end'] + ['backspace'] * len(field.text) + list('-123.456'))
            partial = np.asarray(canvas.buffer_rgba()).copy()
            canvas.draw()
            full = np.asarray(canvas.buffer_rgba()).copy()
            # Text/caret area must match a complete render, with no old glyphs.
            x0, y0, x1, y1 = map(int, field.ax.bbox.extents)
            height = full.shape[0]
            crop = np.s_[height-y1+3:height-y0-3, x0+3:x1-3]
            np.testing.assert_array_equal(partial[crop], full[crop])
            field.stop_typing()

    def test_no_blit_fallback_and_empty_window_entry(self):
        v = self.make_viewer(velocity, loaded=False)
        canvas = v.fig.canvas
        field = v.fields['lead']
        with patch.object(canvas, 'supports_blit', False), \
                patch.object(canvas, 'draw', wraps=canvas.draw) as draw:
            self.click(canvas, field.ax)
            self.keys(canvas, ['end', 'backspace', '2'])
            field.stop_typing()
            self.assertEqual(field.text, '2')
            self.assertGreater(draw.call_count, 0)
        self.assertIsNone(v.result)
        self.assertIn('open or drop', v.message.get_text())


if __name__ == '__main__':
    unittest.main()
