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
# * Perspective transform
# * Detect lane lines
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
    # after all operations; don't quit early
    FULL = enum.auto()


@dataclass
class Params:
    calibration: calibrate.Calibration
    saturation_threshold: Threshold
    sobelx_threshold: Threshold
    step: Step = Step.FULL
    output_dir: str = './output_images'
    output_suffix: str = ''


def apply_threshold(data: np.array, thresh: Threshold) -> np.array:
    tmin, tmax = thresh
    binary = np.zeros_like(data)
    binary[(data >= tmin) & (data <= tmax)] = 1
    return binary


def saturation_threshold(img: np.array, thresh: Threshold) -> np.array:
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    return apply_threshold(s_channel, thresh)


def sobelx_threshold(img: np.array, thresh: Threshold) -> np.array:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # x derivative
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    return apply_threshold(sobel, thresh)


def run_image(img: np.array, params: Params) -> np.array:
    if params.step is Step.ORIGINAL: return img
    img = calibrate.undistort(img, params.calibration)
    if params.step is Step.UNDISTORT: return img

    sobelx = sobelx_threshold(img, params.sobelx_threshold)
    saturation = saturation_threshold(img, params.saturation_threshold)
    # Combined thresholds
    thresholds = np.zeros_like(sobelx)
    thresholds[(saturation == 1) | (sobelx == 1)] = 1
    # Visualize the sobelx (blue) and saturation (green) thresholds for debugging
    if params.step is Step.THRESHOLD_RAW: return thresholds
    if params.step is Step.THRESHOLD_COLOR:
        return np.dstack((np.zeros_like(sobelx), sobelx, saturation)) * 255

    # TODO
    if params.step is Step.FULL: return img
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
    ) for step in
        [
            # Step.THRESHOLD_RAW,
            Step.THRESHOLD_COLOR,
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
