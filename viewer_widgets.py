"""Small desktop controls that do not redraw the flight plots while editing."""

from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, IdentityTransform
from matplotlib.widgets import TextBox


def redraw_controls(figure, artists, previous_bounds=()):
    """Repaint opaque widget axes or figure text on the plain figure background.

    No background image is cached: full draws, resize, seeking and file changes
    cannot leave a stale plot behind a control. Non-blitting backends fall back
    to a normal draw. Callers supply the old bounds when figure text changes.
    """
    canvas = figure.canvas
    if not canvas.supports_blit or not hasattr(canvas, 'renderer'):
        canvas.draw_idle()
        return
    renderer = canvas.get_renderer()
    bounds = list(previous_bounds)
    bounds.extend(artist.get_window_extent(renderer) for artist in artists)
    region = Bbox.intersection(Bbox.union(bounds).padded(2), figure.bbox)
    if region is None:
        return
    # Axes paint their own background. Figure text needs its old pixels erased.
    for bbox in previous_bounds:
        box = bbox.padded(2)
        Rectangle((box.x0, box.y0), box.width, box.height,
                  transform=IdentityTransform(), facecolor=figure.get_facecolor(),
                  edgecolor='none').draw(renderer)
    for artist in artists:
        figure.draw_artist(artist)
    canvas.blit(region)


class ResponsiveTextBox(TextBox):
    """TextBox editing semantics with local redraws on TkAgg/QtAgg/Agg.

    Matplotlib's TextBox redraws the entire figure for cursor movement, hover,
    and even stop_typing() on inactive fields. Override these rendering hooks;
    keep its keyboard handling, callbacks and shortcut lock unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Long input must not paint outside the region we can locally erase.
        self.text_disp.set_clip_on(True)

    def _redraw(self):
        redraw_controls(self.ax.figure, [self.ax])

    def _rendercursor(self):
        canvas = self.ax.figure.canvas
        if not hasattr(canvas, 'get_renderer'):
            return super()._rendercursor()
        renderer = canvas.get_renderer()
        text = self.text
        prefix = text[:self.cursor_index]
        full_box = self.text_disp.get_window_extent(renderer)
        self.text_disp.set_text(prefix or ',')
        prefix_box = self.text_disp.get_window_extent(renderer)
        self.text_disp.set_text(text)
        x = full_box.x0 + (prefix_box.width if prefix else 0)
        y0, y1 = full_box.y0, full_box.y1
        if y0 == y1:
            y0 -= prefix_box.height / 2
            y1 += prefix_box.height / 2
        self.cursor.set(segments=[[(x, y0), (x, y1)]],
                        visible=self.capturekeystrokes)
        self._redraw()

    def stop_typing(self):
        if not self.capturekeystrokes:
            return
        self._on_stop_typing()
        self._on_stop_typing = None
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self._redraw()
        if self.eventson:
            self._observers.process('submit', self.text)

    def _motion(self, event):
        if self.ignore(event):
            return
        color = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not colors.same_color(color, self.ax.get_facecolor()):
            self.ax.set_facecolor(color)
            if self.drawon:
                self._redraw()
