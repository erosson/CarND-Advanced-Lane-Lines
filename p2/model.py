import typing
from dataclasses import dataclass
import numpy as np  # type:ignore
import cv2  # type:ignore

from p2 import calibrate

Threshold = typing.Tuple[int, int]
Pt = typing.Tuple[int, int]


@dataclass
class Params:
    """All input necessary for lane line calculations (other than the current frame's pixels)."""
    calibration: calibrate.Calibration
    saturation_threshold: Threshold
    sobelx_threshold: Threshold
    # percent of image dimensions: (y, (closest-x, farthest-x))
    perspective_src_pcts: typing.Tuple[float, typing.Tuple[float, float, float, float]]
    perspective_dest_pct: float
    num_sliding_windows: int
    sliding_windows_margin: int
    sliding_windows_minpix: int
    xm_per_pix: float
    ym_per_pix: float

    output_dir: str = './output_images'
    subclip: typing.Optional[typing.Tuple[int, int]] = None
    out_path_timestamp: bool = False


# Color/gradient threshold. Based on section 7-12:
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/144d538f-335d-454d-beb2-b1736ec204cb/concepts/a1b70df9-638b-46bb-8af0-12c43dcfd0b4
def apply_threshold(data: np.ndarray, thresh: Threshold) -> np.ndarray:
    tmin, tmax = thresh
    binary = np.zeros_like(data)
    binary[(data >= tmin) & (data <= tmax)] = 1
    return binary


def saturation_threshold(img: np.ndarray, thresh: Threshold) -> np.ndarray:
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    return apply_threshold(s_channel, thresh)


def sobelx_threshold(img: np.ndarray, thresh: Threshold) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # x derivative
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return apply_threshold(sobel, thresh)


@dataclass
class PerspectiveTransform:
    """Perspective transformation output."""
    srcs: typing.List[typing.Tuple[float, float]]
    dests: typing.List[typing.Tuple[float, float]]
    matrix: np.ndarray
    inverse_matrix: np.ndarray


def perspective_transform(shape: typing.Sequence[int], params: Params) -> PerspectiveTransform:
    # Section 8-2: "you can assume the road is a flat plane. This isn't strictly true, but it can serve as an approximation for this project."
    # Points are constants, from params!
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/e6e02d4d-7c80-4bed-a79f-869ef496831b
    (src_y_pct, src_x_pcts) = params.perspective_src_pcts
    y_shape, x_shape = shape[:2]
    srcs_ = [
        (x_shape * src_x_pcts[0], y_shape - 1),
        (x_shape * src_x_pcts[1], y_shape * src_y_pct),
        (x_shape * src_x_pcts[2], y_shape * src_y_pct),
        (x_shape * src_x_pcts[3], y_shape - 1),
    ]
    srcs = np.array([(x, y) for x, y in srcs_], np.float32)
    dests_ = [
        (x_shape * params.perspective_dest_pct, y_shape - 1),
        (x_shape * params.perspective_dest_pct, 0),
        (x_shape * (1 - params.perspective_dest_pct), 0),
        (x_shape * (1 - params.perspective_dest_pct), y_shape - 1),
    ]
    dests = np.array([(x, y) for x, y in dests_], np.float32)
    matrix = cv2.getPerspectiveTransform(srcs, dests)
    inverse_matrix = cv2.getPerspectiveTransform(dests, srcs)
    return PerspectiveTransform(srcs=srcs, dests=dests, matrix=matrix, inverse_matrix=inverse_matrix)


@dataclass
class SlidingWindows:
    histogram: typing.List[int]
    lefts: typing.List[int]
    rights: typing.List[int]
    lnonzeros: typing.Tuple[np.ndarray, np.ndarray]
    rnonzeros: typing.Tuple[np.ndarray, np.ndarray]
    polys: typing.Tuple[np.array, np.array]
    ploty: typing.List[int]
    fitx: typing.Tuple[typing.List[float], typing.List[float]]


def sliding_windows(warped: np.ndarray, params: Params) -> SlidingWindows:
    # first lane line detection by histogram of the bottom half of the screen: section 8-3
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/011b8b18-331f-4f43-8a04-bf55787b347f
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    midpoint = len(histogram) // 2
    left = max(enumerate(histogram[:midpoint]), key=lambda v: v[1])[0]
    right = max(enumerate(histogram[midpoint:]), key=lambda v: v[1])[0] + midpoint

    # iterate for all windows after the first: section 8-4
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/4dd9f2c2-1722-412f-9a02-eec3de0c2207
    lefts = []
    rights = []
    lnonzeros = []
    rnonzeros = []
    nonzero_y, nonzero_x = [np.array(n) for n in warped.nonzero()]
    for window_num in range(params.num_sliding_windows):
        lefts.append(left)
        rights.append(right)
        left_pts = _sliding_window_pts(left, warped.shape[0], window_num, params)
        right_pts = _sliding_window_pts(right, warped.shape[0], window_num, params)
        left_xs = [x for x, y in left_pts]
        right_xs = [x for x, y in right_pts]
        ys = [y for x, y in left_pts]
        ymin, ymax = min(ys), max(ys)
        lxmin, lxmax = min(left_xs), max(left_xs)
        rxmin, rxmax = min(right_xs), max(right_xs)
        lnonzero = ((nonzero_y >= ymin) & (nonzero_y < ymax) & (nonzero_x >= lxmin) & (nonzero_x < lxmax)).nonzero()[0]
        rnonzero = ((nonzero_y >= ymin) & (nonzero_y < ymax) & (nonzero_x >= rxmin) & (nonzero_x < rxmax)).nonzero()[0]

        lnonzeros.append(lnonzero)
        rnonzeros.append(rnonzero)

        if len(lnonzero) > params.sliding_windows_minpix:
            left = np.int(np.mean(nonzero_x[lnonzero]))
        if len(rnonzero) > params.sliding_windows_minpix:
            right = np.int(np.mean(nonzero_x[rnonzero]))
    lnonzeros = np.concatenate(lnonzeros)
    rnonzeros = np.concatenate(rnonzeros)
    ly, lx = (nonzero_y[lnonzeros], nonzero_x[lnonzeros])
    ry, rx = (nonzero_y[rnonzeros], nonzero_x[rnonzeros])
    polys = lpoly, rpoly = (np.polyfit(ly, lx, 2), np.polyfit(ry, rx, 2))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    try:
        left_fitx = lpoly[0] * ploty ** 2 + lpoly[1] * ploty + lpoly[2]
        right_fitx = rpoly[0] * ploty ** 2 + rpoly[1] * ploty + rpoly[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        # print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return SlidingWindows(histogram=histogram, lefts=lefts, rights=rights, lnonzeros=(ly, lx), rnonzeros=(ry, rx),
                          polys=polys, fitx=(left_fitx, right_fitx), ploty=ploty)


def _sliding_window_pts(line_x: int, img_y: int, window_num: int, params: Params) -> typing.Tuple[Pt, Pt, Pt, Pt]:
    window_height = img_y // params.num_sliding_windows
    return (
        (line_x - params.sliding_windows_margin, img_y - (window_num) * window_height),
        (line_x + params.sliding_windows_margin, img_y - (window_num) * window_height),
        (line_x + params.sliding_windows_margin, img_y - (window_num + 1) * window_height),
        (line_x - params.sliding_windows_margin, img_y - (window_num + 1) * window_height),
    )


def sliding_window_pts(line_xs: typing.List[int], img_y: int, params: Params) -> typing.List[typing.List[np.array]]:
    return [[np.array(_sliding_window_pts(line_x, img_y, i, params), np.int32)] for i, line_x in enumerate(line_xs)]


def lane_radius_meters(ploty: np.array, polys: typing.Tuple[np.array, np.array], ym_per_pix: float) -> float:
    """Calculate radius of the lane, given polynomial curves for both lane lines."""
    # Equations based on section 8-7:
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/1a352727-390e-469d-87ea-c91cd78869d6
    # Numbers checked against highway curvature specs, and they seem reasonable:
    # http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm
    y_eval = np.max(ploty)
    curverads = [
        ((1 + (2 * poly[0] * y_eval * ym_per_pix + poly[1]) ** 2) ** 1.5) / np.absolute(2 * poly[0])
        for poly in polys]
    return sum(curverads) / len(curverads)


def lane_offset_width_meters(shape: typing.Sequence[int],
                             poly_evals: typing.Tuple[typing.List[float], typing.List[float]],
                             xm_per_pix: float) -> typing.Tuple[float, float]:
    """Calculate center-offset of the lane, given polynomial curves for both lane lines.

    The vehicle is left of lane-center for values less than 0, and right of lane-center for values greater than 0.
    """
    y_shape, x_shape = shape[:2]
    y_eval = y_shape - 1
    left_eval, right_eval = poly_evals

    vehicle_center = x_shape // 2
    lane_center = sum(poly[y_eval] for poly in poly_evals) / len(poly_evals)
    offset_pix = vehicle_center - lane_center
    offset_m = offset_pix * xm_per_pix

    lane_width_m = abs(left_eval[y_eval] - right_eval[y_eval]) * xm_per_pix
    return (offset_m, lane_width_m)


@dataclass
class Model:
    """All calculations derived from the original image + parameters."""
    params: Params
    original: np.ndarray
    undistort: np.ndarray
    sobelx: np.ndarray
    saturation: np.ndarray
    thresholds: np.ndarray
    perspective: PerspectiveTransform
    warp_size: typing.Tuple[int, int]
    warped: np.ndarray
    sw: SlidingWindows
    offset: float
    lane_width: float
    radius: float


def model(original: np.ndarray, params: Params) -> Model:
    undistort = calibrate.undistort(original, params.calibration)

    # saturation/sobel thresholding
    sobelx = sobelx_threshold(undistort, params.sobelx_threshold)
    saturation = saturation_threshold(undistort, params.saturation_threshold)
    # Combined thresholds
    thresholds = np.zeros_like(sobelx)
    thresholds[(saturation == 1) | (sobelx == 1)] = 1

    # perspective transform
    perspective = perspective_transform(undistort.shape, params)
    warp_size = (undistort.shape[1], undistort.shape[0])
    warped = cv2.warpPerspective(thresholds, perspective.matrix, warp_size, flags=cv2.INTER_LINEAR)

    # sliding windows; fit the lane to a polynomial
    sw = sliding_windows(warped, params)

    # lane position and radius
    offset, lane_width = lane_offset_width_meters(shape=warped.shape, poly_evals=sw.fitx, xm_per_pix=params.xm_per_pix)
    radius = lane_radius_meters(ploty=sw.ploty, polys=sw.polys, ym_per_pix=params.ym_per_pix)

    return Model(params=params, original=original, undistort=undistort, sobelx=sobelx, saturation=saturation,
                 thresholds=thresholds, perspective=perspective, warp_size=warp_size, warped=warped, sw=sw,
                 offset=offset, lane_width=lane_width, radius=radius)
