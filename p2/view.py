import typing
from dataclasses import dataclass
import enum
import matplotlib.pyplot as plt  # type:ignore
import cv2  # type:ignore
import numpy as np  # type:ignore

from p2 import model
from p2.model import Params, Model


class View(enum.Enum):
    # identity; return the original image
    ORIGINAL = enum.auto()
    # after camera distortion removed
    UNDISTORT = enum.auto()
    # after sobel/saturation thresholding; same data, different visualizations
    THRESHOLD_RAW = enum.auto()
    THRESHOLD_COLOR = enum.auto()
    # before/after perspective transform
    PERSPECTIVE_PRE = enum.auto()
    PERSPECTIVE_THRESHOLD_PRE = enum.auto()
    PERSPECTIVE_POST = enum.auto()
    PERSPECTIVE_THRESHOLD_POST = enum.auto()
    # lane-detection histogram
    HISTOGRAM_PLOT = enum.auto()
    HISTOGRAM_WINDOWS = enum.auto()
    # after all operations; don't quit early
    FULL = enum.auto()

    _renderer: typing.Optional[typing.Callable[[Model], np.ndarray]]

    def output_suffix(self) -> str:
        return f"-{self.value:02d}-{self.name.lower()}"

    def render(self, m: Model) -> np.ndarray:
        if self._renderer:
            return self._renderer(m)
        raise Exception("no renderer set for this view", self)


def _set_renderer(v: View):
    def _curried(renderer: typing.Callable[[Model], np.ndarray]):
        v._renderer = renderer
        return renderer

    return _curried


@_set_renderer(View.ORIGINAL)
def _render_original(m: Model) -> np.ndarray:
    return m.original


@_set_renderer(View.UNDISTORT)
def _render_undistort(m: Model) -> np.ndarray:
    return m.undistort


@_set_renderer(View.THRESHOLD_RAW)
def _render_threshold_raw(m: Model) -> np.ndarray:
    return m.thresholds


@_set_renderer(View.THRESHOLD_COLOR)
def _render_threshold_color(m: Model) -> np.ndarray:
    # Visualize the sobelx (blue) and saturation (green) thresholds for debugging
    return np.dstack((np.zeros_like(m.sobelx), m.sobelx, m.saturation)) * 255


@_set_renderer(View.PERSPECTIVE_PRE)
def _render_perspective_pre(m: Model, view: np.ndarray = None) -> np.ndarray:
    view = view if view is not None else m.undistort
    cv2.polylines(view, [np.array(m.perspective.srcs, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)
    return view


@_set_renderer(View.PERSPECTIVE_THRESHOLD_PRE)
def _render_perspective_threshold_pre(m: Model) -> np.ndarray:
    return _render_perspective_pre(m, _render_threshold_color(m))


@_set_renderer(View.PERSPECTIVE_POST)
def _render_perspective_post(m: Model, view: np.ndarray = None) -> np.ndarray:
    view = view if view is not None else m.undistort
    view = cv2.warpPerspective(view, m.perspective.matrix, m.warp_size, flags=cv2.INTER_LINEAR)
    cv2.polylines(view, [np.array(m.perspective.dests, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)
    return view


@_set_renderer(View.PERSPECTIVE_THRESHOLD_POST)
def _render_perspective_threshold_post(m: Model) -> np.ndarray:
    return _render_perspective_post(m, _render_threshold_color(m))


@_set_renderer(View.HISTOGRAM_PLOT)
def _render_histogram_plot(m: Model) -> np.ndarray:
    # histogram to image. thanks, https://stackoverflow.com/a/7821917
    # TODO: it'd be cool to overlay the perspective image with this
    fig = plt.figure()
    plt.plot(m.sw.histogram, figure=fig)
    fig.canvas.draw()
    view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    view = view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return view


@_set_renderer(View.HISTOGRAM_WINDOWS)
def _render_histogram_windows(m: Model) -> np.ndarray:
    # red/blue lane pixels
    view = np.dstack((m.warped, m.warped, m.warped)) * 255
    view[m.sw.lnonzeros] = (255, 0, 0)
    view[m.sw.rnonzeros] = (0, 0, 255)

    # green window boxes
    leftpts = model.sliding_window_pts(m.sw.lefts, view.shape[0], m.params)
    rightpts = model.sliding_window_pts(m.sw.rights, view.shape[0], m.params)
    for pts in leftpts + rightpts:
        # overlay = np.copy(img)
        # cv2.fillPoly(overlay, leftpts, color=(255, 255, 0))
        # cv2.fillPoly(overlay, rightpts, color=(255, 255, 0))
        # img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.polylines(view, pts, isClosed=True, color=(0, 255, 0, 0), thickness=3)

    # yellow polynomial lines
    # draw over the existing image. What a pain. Thanks, https://stackoverflow.com/a/34459284 and https://stackoverflow.com/a/9295367
    fig = plt.figure()
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(view)
    for fitx in m.sw.fitx:
        plt.plot(fitx, m.sw.ploty, figure=fig, color='yellow')
    fig.canvas.draw()
    view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    view = view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return view


@_set_renderer(View.FULL)
def _render_full(m: Model) -> np.ndarray:
    view = m.undistort
    # Red/blue lane lines
    lanes_overlay = np.copy(view)
    lanes_overlay.fill(0)
    lanes_overlay[m.sw.lnonzeros] = (255, 0, 0)
    lanes_overlay[m.sw.rnonzeros] = (0, 0, 255)
    # Green rectangle between the two lane lines
    # There's probably a fancier loop-free way to do this, but hell if I can figure it out
    center_overlay = np.copy(view)
    center_overlay.fill(0)
    position_overlay = np.copy(view)
    position_overlay.fill(0)
    for y in m.sw.ploty:
        y = int(y)
        left_fitx, right_fitx = m.sw.fitx
        center_overlay[y, int(left_fitx[y]):int(right_fitx[y])] = (0, 255, 0)
        xcenter = int((left_fitx[y] + right_fitx[y]) / 2)
        if int((y / 10) % 2) == 0:
            position_overlay[y, xcenter - 2:xcenter + 2] = (0, 255, 0)
    # Combine the overlays with the original
    overlay = lanes_overlay
    overlay = cv2.addWeighted(center_overlay, 0.2, overlay, 1, 0)
    overlay = cv2.addWeighted(position_overlay, 0.7, overlay, 1, 0)
    overlay = cv2.warpPerspective(overlay, m.perspective.inverse_matrix, m.warp_size, flags=cv2.INTER_LINEAR)
    view = cv2.addWeighted(overlay, 1, view, 1, 0)
    # green dot in center-bottom of lane
    # view[view.shape[0] - 6:view.shape[0] - 1, int(view.shape[1] // 2) - 2:int(view.shape[1] // 2) + 2] = (0, 255, 0)

    radius_text = f"radius of curvature: {m.radius: .2f}m"
    offset_text = f"""vehicle offset: {abs(m.offset): .2f}m {
    "(centered)" if m.offset == 0 else
    "left of lane center" if m.offset < 0 else
    "right of lane center"
    }{" - DANGER" if abs(m.offset) > 0.5 else ""}"""
    width_text = f"lane width: {m.lane_width: .2f}m"
    view = cv2.putText(view, radius_text, (50, 50 + 14 * 0),
                       fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255))
    view = cv2.putText(view, offset_text, (50, 50 + 14 * 1),
                       fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255))
    view = cv2.putText(view, width_text, (50, 50 + 14 * 2),
                       fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255))
    return view
