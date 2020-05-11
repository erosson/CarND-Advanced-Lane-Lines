import typing
import os
import sys
import glob
import matplotlib.image as mpimg  # type:ignore
import numpy as np  # type:ignore
import moviepy.editor  # type:ignore

from p2 import calibrate, model, view
from p2.model import Model, Params
from p2.view import View


# x Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# x Apply a distortion correction to raw images.
# x Use color transforms, gradients, etc., to create a thresholded binary image.
# x Apply a perspective transform to rectify binary image ("birds-eye view").
# x Detect lane pixels and fit to find the lane boundary.
#   * section 8-5: search from prior state to skip sliding window, when possible, for faster processing. https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/474a329a-78d0-4a33-833a-34d02a35fc13
# x Determine the curvature of the lane and vehicle position with respect to center.
#   x vehicle position: lane midpoint. https://knowledge.udacity.com/questions/30469
# x Warp the detected lane boundaries back onto the original image.
# x Output visual display of the lane boundaries
#   x ...and numerical estimation of lane curvature and vehicle position.


def run_image(original: np.ndarray, params: Params, view_: View) -> np.ndarray:
    model_ = model.model(original, params)
    return view_.render(model_)


def run_video(clip: moviepy.editor.VideoClip, params: Params, view_: View) -> moviepy.Clip:
    return clip.fl_image(lambda img: run_image(img, params, view_))


def run_path(in_path: str, params: Params, view_: View) -> None:
    basename, ext = os.path.splitext(os.path.basename(in_path))
    out_path: str = os.path.join(params.output_dir, basename + view_.output_suffix() + ext)
    print(in_path, out_path)
    # Handle all file i/o, so the rest of our code can be side-effect-free.
    if in_path.endswith('.jpg'):
        in_img = mpimg.imread(in_path)
        out_img = run_image(in_img, params=params, view_=view_)
        mpimg.imsave(out_path, out_img)
    elif in_path.endswith('.mp4'):
        in_clip = moviepy.editor.VideoFileClip(in_path)
        if params.subclip:
            in_clip = in_clip.subclip(*params.subclip)
        out_clip = run_video(in_clip, params=params, view_=view_)
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
        args = []
        args += test_images_paths()
        # args += [PROJECT_VIDEO_PATH]
        # args += [CHALLENGE_VIDEO_PATH]
        # args += [CHALLENGE2_VIDEO_PATH]

    # One set of Params for every view we're interested in.
    #
    # Processing one View at a time is wasteful - we reprocess all earlier views every time, when in theory we could
    # output all views after running once. This is easier to implement, is reusable with images/video, and we're usually
    # only worried about 1-2 views at a time.
    params = Params(
        calibration=calibrate.load_or_calibrate_default(debug=True),
        sobelx_threshold=(20, 100),
        saturation_threshold=(170, 255),
        perspective_src_pcts=(445 / 720, (200 / 1280, 600 / 1280, 680 / 1280, 1120 / 1280)),
        perspective_dest_pct=300 / 1280,
        num_sliding_windows=9,
        sliding_windows_margin=100,
        sliding_windows_minpix=50,
        # meters per pixel, from section 8-7:
        # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/626f183c-593e-41d7-a828-eda3c6122573/concepts/1a352727-390e-469d-87ea-c91cd78869d6
        ym_per_pix=30 / 720,
        xm_per_pix=3.7 / 700,
        # subclip=(3, 6),
    )
    # views = list(View)
    views = [
        View.ORIGINAL,
        View.UNDISTORT,
        View.THRESHOLD_RAW,
        View.THRESHOLD_COLOR,
        View.PERSPECTIVE_PRE,
        View.PERSPECTIVE_POST,
        View.PERSPECTIVE_THRESHOLD_PRE,
        View.PERSPECTIVE_THRESHOLD_POST,
        View.HISTOGRAM_PLOT,
        View.HISTOGRAM_WINDOWS,
        View.FULL,
    ]

    # clean up old output
    for ext in ['jpg', 'mp4']:
        for path in glob.glob(os.path.join(params.output_dir, "*." + ext)):
            os.remove(path)

    # finally, run each file
    for arg in args:
        for view_ in views:
            run_path(arg, params, view_)


if __name__ == '__main__':
    main()
