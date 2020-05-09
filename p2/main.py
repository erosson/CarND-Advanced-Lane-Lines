import matplotlib.pyplot as plt  # type:ignore
import matplotlib.image as mpimg  # type:ignore
import numpy as np  # type:ignore
import cv2  # type:ignore
import os
import sys
import moviepy.editor  # type:ignore
import typing
from p2 import calibrate


# Based on lesson 8 section 1, our overarching steps:
# * Camera calibration
# * Distortion correction
#   * based on camera calibration; applies to each image/frame
# * Color/gradient threshold
# * Perspective transform
# * Detect lane lines
# * Determine the lane curvature


def run_image(img: np.array, calibration: calibrate.Calibration) -> np.array:
    # TODO
    return img


def run_video(clip: moviepy.editor.VideoClip, calibration: calibrate.Calibration) -> moviepy.Clip:
    # TODO
    return clip


def run_path(in_path: str, calibration: calibrate.Calibration) -> None:
    out_path: str = os.path.join('output_images', os.path.basename(in_path))
    print(in_path, out_path)
    # Handle all file i/o, so the rest of our code can be side-effect-free.
    # TODO: debug file outputs?
    if in_path.endswith('.jpg'):
        in_img = mpimg.imread(in_path)
        out_img = run_image(in_img, calibration=calibration)
        mpimg.imsave(out_path, out_img)
    elif in_path.endswith('.mp4'):
        # in_clip = moviepy.editor.VideoFileClip(in_path)
        in_clip = moviepy.editor.VideoFileClip(in_path).subclip(3, 6)
        out_clip = run_video(in_clip, calibration=calibration)
        out_clip.write_videofile(out_path)
    else:
        raise Exception('unknown file extension', in_path)


TEST_IMAGES_DIR = './assets/test_images'
PROJECT_VIDEO_PATH = './assets/project_video.mp4'


def test_images_paths() -> typing.List[str]:
    return [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR)]


def main() -> None:
    args = sys.argv[1:]
    if not len(args):
        args = test_images_paths()
        # args = [PROJECT_VIDEO_PATH]

    calibration = calibrate.load_or_calibrate_default(debug=True)
    print(args, len(calibration))
    for arg in args:
        run_path(arg, calibration=calibration)


if __name__ == '__main__':
    main()
