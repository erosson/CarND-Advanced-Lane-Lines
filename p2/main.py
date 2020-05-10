import matplotlib.pyplot as plt  # type:ignore
import matplotlib.image as mpimg  # type:ignore
import numpy as np  # type:ignore
import cv2  # type:ignore
import os
import sys
import moviepy.editor  # type:ignore
import typing
from dataclasses import dataclass
from p2 import calibrate
import enum
import glob

# Based on lesson 8 section 1, our overarching steps:
# x Camera calibration
# x Distortion correction
# x Color/gradient threshold
# x Perspective transform
# * Detect lane lines: histogram
# * Determine the lane curvature

# Color/gradient threshold. Based on section 7-12:
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/144d538f-335d-454d-beb2-b1736ec204cb/concepts/a1b70df9-638b-46bb-8af0-12c43dcfd0b4

Threshold = typing.Tuple[int, int]


class Step(enum.Enum):
    # identity; return the original image
    ORIGINAL = enum.auto()
    # after camera distortion removed
    UNDISTORT = enum.auto()
    # after sobel/saturation thresholding; same data, different visualizations
    THRESHOLD_RAW = enum.auto()
    THRESHOLD_COLOR = enum.auto()
    # before/after perspective transform
    PERSPECTIVE_PRE = enum.auto()
    PERSPECTIVE_POST = enum.auto()
    PERSPECTIVE_THRESHOLD_PRE = enum.auto()
    PERSPECTIVE_THRESHOLD_POST = enum.auto()
    # lane-detection histogram
    HISTOGRAM_PLOT = enum.auto()
    HISTOGRAM_WINDOWS = enum.auto()
    # after all operations; don't quit early
    FULL = enum.auto()


Pt = typing.Tuple[int, int]


@dataclass
class Params:
    calibration: calibrate.Calibration
    saturation_threshold: Threshold
    sobelx_threshold: Threshold
    # percent of image dimensions: (y, (closest-x, farthest-x))
    perspective_src_pcts: typing.Tuple[float, typing.Tuple[float, float, float, float]]
    perspective_dest_pct: float
    num_sliding_windows: int
    sliding_windows_margin: int

    step: Step = Step.FULL
    output_dir: str = './output_images'
    output_suffix: str = ''


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
    srcs: typing.List[typing.Tuple[float, float]]
    dests: typing.List[typing.Tuple[float, float]]
    matrix: np.ndarray


def perspective_transform(img: np.ndarray, params: Params) -> PerspectiveTransform:
    # Section 8-2: "you can assume the road is a flat plane. This isn't strictly true, but it can serve as an approximation for this project."
    # Points are constants, from params!
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/e6e02d4d-7c80-4bed-a79f-869ef496831b
    (src_y_pct, src_x_pcts) = params.perspective_src_pcts
    (y_shape, x_shape, a_shape) = img.shape
    srcs = [
        (x_shape * src_x_pcts[0], y_shape - 1),
        (x_shape * src_x_pcts[1], y_shape * src_y_pct),
        (x_shape * src_x_pcts[2], y_shape * src_y_pct),
        (x_shape * src_x_pcts[3], y_shape - 1),
    ]
    srcs = np.array([(x, y) for x, y in srcs], np.float32)
    dests = [
        (x_shape * params.perspective_dest_pct, y_shape - 1),
        (x_shape * params.perspective_dest_pct, 0),
        (x_shape * (1 - params.perspective_dest_pct), 0),
        (x_shape * (1 - params.perspective_dest_pct), y_shape - 1),
    ]
    dests = np.array([(x, y) for x, y in dests], np.float32)
    matrix = cv2.getPerspectiveTransform(srcs, dests)
    return PerspectiveTransform(srcs=srcs, dests=dests, matrix=matrix)


def threshold_color(sobelx: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    # Visualize the sobelx (blue) and saturation (green) thresholds for debugging
    return np.dstack((np.zeros_like(sobelx), sobelx, saturation)) * 255


@dataclass
class SlidingWindows:
    histogram: typing.List[int]
    lefts: typing.List[int]
    rights: typing.List[int]
    lnonzeros: typing.Tuple[np.ndarray, np.ndarray]
    rnonzeros: typing.Tuple[np.ndarray, np.ndarray]


def sliding_windows(warped: np.ndarray, params: Params) -> SlidingWindows:
    # TODO parameterize
    minpix = 50
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

        if len(lnonzero) > minpix:
            left = np.int(np.mean(nonzero_x[lnonzero]))
        if len(rnonzero) > minpix:
            right = np.int(np.mean(nonzero_x[rnonzero]))
    lnonzeros = np.concatenate(lnonzeros)
    rnonzeros = np.concatenate(rnonzeros)
    return SlidingWindows(
        histogram=histogram, lefts=lefts, rights=rights,
        lnonzeros=(nonzero_y[lnonzeros], nonzero_x[lnonzeros]),
        rnonzeros=(nonzero_y[rnonzeros], nonzero_x[rnonzeros]),
    )


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


def run_image(img: np.ndarray, params: Params) -> np.ndarray:
    if params.step is Step.ORIGINAL: return img
    img = calibrate.undistort(img, params.calibration)
    if params.step is Step.UNDISTORT: return img

    # saturation/sobel thresholding
    sobelx = sobelx_threshold(img, params.sobelx_threshold)
    saturation = saturation_threshold(img, params.saturation_threshold)
    # Combined thresholds
    thresholds = np.zeros_like(sobelx)
    thresholds[(saturation == 1) | (sobelx == 1)] = 1
    if params.step is Step.THRESHOLD_RAW: return thresholds
    if params.step is Step.THRESHOLD_COLOR:
        return threshold_color(sobelx=sobelx, saturation=saturation)

    # perspective transform
    transform = perspective_transform(img, params)
    if params.step is Step.PERSPECTIVE_PRE or params.step is Step.PERSPECTIVE_THRESHOLD_PRE:
        if params.step is Step.PERSPECTIVE_THRESHOLD_PRE:
            img = threshold_color(sobelx=sobelx, saturation=saturation)
        cv2.polylines(img, [np.array(transform.srcs, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)
        return img
    warp_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(thresholds, transform.matrix, warp_size, flags=cv2.INTER_LINEAR)
    if params.step is Step.PERSPECTIVE_POST or params.step is Step.PERSPECTIVE_THRESHOLD_POST:
        if params.step is Step.PERSPECTIVE_THRESHOLD_POST:
            img = threshold_color(sobelx=sobelx, saturation=saturation)
        img = cv2.warpPerspective(img, transform.matrix, warp_size, flags=cv2.INTER_LINEAR)
        cv2.polylines(img, [np.array(transform.dests, np.int32)], isClosed=True, color=(255, 0, 0), thickness=3)
        return img

    sw = sliding_windows(warped, params)
    if params.step is Step.HISTOGRAM_PLOT:
        # histogram to image. thanks, https://stackoverflow.com/a/7821917
        # TODO: it'd be cool to overlay the perspective image with this
        fig = plt.figure()
        plt.plot(sw.histogram, figure=fig)
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img
    if params.step is Step.HISTOGRAM_WINDOWS:
        # img = threshold_color(sobelx=sobelx, saturation=saturation)
        # img = cv2.warpPerspective(img, transform.matrix, warp_size, flags=cv2.INTER_LINEAR)
        img = np.dstack((warped, warped, warped)) * 255
        img[sw.lnonzeros] = (255, 0, 0)
        img[sw.rnonzeros] = (0, 0, 255)
        leftpts = sliding_window_pts(sw.lefts, img.shape[0], params)
        rightpts = sliding_window_pts(sw.rights, img.shape[0], params)
        for pts in leftpts + rightpts:
            # overlay = np.copy(img)
            # cv2.fillPoly(overlay, leftpts, color=(255, 255, 0))
            # cv2.fillPoly(overlay, rightpts, color=(255, 255, 0))
            # img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            cv2.polylines(img, pts, isClosed=True, color=(0, 255, 0, 0), thickness=3)
        return img
    if params.step is Step.FULL:
        return img
    raise Exception('no such step', params.step)


def run_video(clip: moviepy.editor.VideoClip, params: Params) -> moviepy.Clip:
    return clip.fl_image(lambda img: run_image(img, params))


def run_path(in_path: str, params: Params) -> None:
    basename, ext = os.path.splitext(os.path.basename(in_path))
    out_path: str = os.path.join(params.output_dir, basename + params.output_suffix + ext)
    print(in_path, out_path)
    # Handle all file i/o, so the rest of our code can be side-effect-free.
    # TODO: debug file outputs?
    if in_path.endswith('.jpg'):
        in_img = mpimg.imread(in_path)
        out_img = run_image(in_img, params=params)
        mpimg.imsave(out_path, out_img)
    elif in_path.endswith('.mp4'):
        in_clip = moviepy.editor.VideoFileClip(in_path)
        # in_clip = moviepy.editor.VideoFileClip(in_path).subclip(3, 6)
        out_clip = run_video(in_clip, params=params)
        # out_clip.write_videofile(out_path, audio=False, logger=None)
        out_clip.write_videofile(out_path, audio=False)
    else:
        raise Exception('unknown file extension', in_path)


TEST_IMAGES_DIR = './assets/test_images'
PROJECT_VIDEO_PATH = './assets/project_video.mp4'
CHALLENGE_VIDEO_PATH = './assets/challenge_video.mp4'
CHALLENGE2_VIDEO_PATH = './assets/harder_challenge_video.mp4'


def test_images_paths() -> typing.List[str]:
    return [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR)]


def main() -> None:
    # accept input files as command-line args
    args = sys.argv[1:]
    if not len(args):
        args = test_images_paths()
        # args = [PROJECT_VIDEO_PATH]
        # args = [CHALLENGE_VIDEO_PATH]
        # args = [CHALLENGE2_VIDEO_PATH]

    # One set of Params for every processing Step we're interested in.
    #
    # Processing one Step at a time is wasteful - we reprocess all earlier steps every time, when in theory we could
    # output all steps after running once. This is easier to implement, is reusable with images/video, and we're usually
    # only worried about 1-2 steps at a time.
    step_params = [Params(
        step=step,
        output_suffix='-' + step.name.lower(),
        calibration=calibrate.load_or_calibrate_default(debug=True),
        sobelx_threshold=(20, 100),
        saturation_threshold=(170, 255),
        # estimates from this image:
        # https://video.udacity-data.com/topher/2016/December/58448557_warped-straight-lines/warped-straight-lines.jpg
        perspective_src_pcts=(445 / 720, (200 / 1280, 600 / 1280, 680 / 1280, 1120 / 1280)),
        perspective_dest_pct=300 / 1280,
        num_sliding_windows=9,
        sliding_windows_margin=100,
    ) for step in
        # list(Step)
        [
            # Step.ORIGINAL,
            # Step.UNDISTORT,
            # Step.THRESHOLD_RAW,
            # Step.THRESHOLD_COLOR,
            # Step.PERSPECTIVE_PRE,
            # Step.PERSPECTIVE_POST,
            # Step.PERSPECTIVE_THRESHOLD_PRE,
            # Step.PERSPECTIVE_THRESHOLD_POST,
            # Step.HISTOGRAM_PLOT,
            Step.HISTOGRAM_WINDOWS,
            # Step.FULL,
        ]
    ]

    # clean up old output
    for ext in ['jpg', 'mp4']:
        for path in glob.glob(os.path.join(step_params[0].output_dir, "*." + ext)):
            os.remove(path)

    # finally, run each file
    for arg in args:
        for params in step_params:
            run_path(arg, params)


if __name__ == '__main__':
    main()
