# type: ignore
# type:ignore
import matplotlib.pyplot as plt
# type:ignore
import matplotlib.image as mpimg
# type:ignore
import numpy as np
# type:ignore
import cv2
import math
import os
import shutil
import sys
import moviepy.editor
import typing
import pickle
import collections
import calibrate


# Based on lesson 8 section 1, our overarching steps:
# * Camera calibration
#   * `assets/camera_cal`, for the same camera used in all other images/videos)
#   * hardcode the folder, do this once at the beginning, and use the calibration data everywhere else
# * Distortion correction
#   * based on camera calibration; applies to each image/frame
# * Color/gradient threshold
# * Perspective transform
# * Detect lane lines
# * Determine the lane curvature


def run_image(img: np.array) -> typing.Dict[str, np.array]:
    # TODO
    return img


def run_video(clip: any) -> any:
    # TODO
    return clip


def run_path(in_path: str) -> None:
    out_path: str = os.path.join('output_images', os.path.basename(in_path))
    print(in_path, out_path)
    # Handle all file i/o, so the rest of our code can be side-effect-free.
    # TODO: debug file outputs?
    if in_path.endswith('.jpg'):
        in_img = mpimg.imread(in_path)
        out_img = run_image(in_img)
        mpimg.imsave(out_path, out_img)
    elif in_path.endswith('.mp4'):
        # in_clip = moviepy.editor.VideoFileClip(in_path)
        in_clip = moviepy.editor.VideoFileClip(in_path).subclip(3, 6)
        out_clip = run_video(in_clip)
        out_clip.write_videofile(out_path)
    else:
        raise Exception('unknown file extension', in_path)


def main() -> None:
    cal_dir = 'assets/camera_cal'
    cal = calibrate.calibrate((os.path.join(cal_dir, f) for f in os.listdir(cal_dir)), 9, 6, 'output_images/cal')
    print(cal)
    return

    args = sys.argv[1:]
    if not len(args):
        # default_dir = './assets/test_images'
        # args = [os.path.join(default_dir, f) for f in os.listdir(default_dir)]
        args = ['./assets/project_video.mp4']
    for arg in args:
        run_path(arg)


if __name__ == '__main__':
    main()
