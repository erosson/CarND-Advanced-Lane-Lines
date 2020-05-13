import typing
import os
import shutil
import pickle
import cv2  # type:ignore
import numpy as np  # type:ignore

Calibration = typing.NewType('Calibration', typing.Tuple[object, object, object, object, object])


def _calibrate(in_path: str, nx: int, ny: int) -> typing.Tuple[bool, np.array, object]:
    # Borrowed lots of code here from udacity section 6-11, "calibrating your camera":
    # https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/5d1efbaa-27d0-4ad5-a67a-48729ccebd9c/lessons/78afdfc4-f0fa-4505-b890-5d8e6319e15c/concepts/a30f45cb-c1c0-482c-8e78-a26604841ec0
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (ny, nx), None)
    return ret, corners, gray.shape


def calibrate(in_paths: typing.List[str], nx: int, ny: int, debug_out: typing.Optional[str] = None) -> Calibration:
    cals: typing.List[typing.Tuple[bool, np.array, object]] = [_calibrate(in_path, nx, ny) for in_path in in_paths]
    imgpoints = [c for (ret, c, shape) in cals if ret]

    # Udacity's code... how on earth do you read this numpy stuff?
    # objp = np.zeros((ny * nx, 3), np.float32)
    # objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    # This is equivalent and easier to read
    objp = np.array([(x, y, 0) for x in range(0, nx) for y in range(0, ny)], np.float32)
    objpoints = [objp for _ in imgpoints]

    shape = cals[0][2]  # shape of all calibration images should be the same, so take the first
    calibration = Calibration(cv2.calibrateCamera(objpoints, imgpoints, shape, None, None))

    if debug_out:
        shutil.rmtree(debug_out, ignore_errors=True)
        os.makedirs(debug_out)
        for i, in_path in enumerate(in_paths):
            img = cv2.imread(in_path)
            basename, ext = os.path.splitext(os.path.basename(in_path))
            cc_path = os.path.join(debug_out, basename + ext)
            unwarp_path = os.path.join(debug_out, basename + '_unwarp' + ext)

            ret, corners, _ = cals[i]
            if corners is not None:
                cc_img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                unwarp_img = undistort(cc_img, calibration)

                print(in_path, "(drawcc)->", [cc_path, unwarp_path])
                cv2.imwrite(cc_path, cc_img)
                cv2.imwrite(unwarp_path, unwarp_img)
            else:
                cc_img = img
                unwarp_img = undistort(cc_img, calibration)
                print(in_path, "->", [cc_path, unwarp_path])
                cv2.imwrite(cc_path, cc_img)
                cv2.imwrite(unwarp_path, unwarp_img)
    return calibration


def load_or_calibrate(*args, **kwargs) -> Calibration:
    """Store/load calibration data as a pickle file for faster reruns."""
    load_path = kwargs.pop('load_path')
    try:
        # raise FileNotFoundError()  # uncomment to force recalibration
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        c = calibrate(*args, **kwargs)
        with open(load_path, 'wb') as f:
            pickle.dump(c, f)
        return c


DEFAULT_CALIBRATION_DIR = 'assets/camera_cal'
NX = 9
NY = 6
LOAD_PATH = 'output_images/calibration.pickle'
DEBUG_OUT = 'output_images/cal'


def default_calibration_paths() -> typing.List[str]:
    return [os.path.join(DEFAULT_CALIBRATION_DIR, f) for f in os.listdir(DEFAULT_CALIBRATION_DIR)]


def load_or_calibrate_default(debug: bool = False) -> Calibration:
    return load_or_calibrate(default_calibration_paths(), NX, NY, load_path=LOAD_PATH,
                             debug_out=DEBUG_OUT if debug else None)


def undistort(img: np.array, calibration: Calibration) -> np.array:
    _, mtx, dist, _, _ = calibration
    return cv2.undistort(img, mtx, dist, None, mtx)


if __name__ == '__main__':
    load_or_calibrate_default(debug=True)
    # load_or_calibrate(CALIBRATION_PATHS, NX, NY, load_path=LOAD_PATH)
    # calibrate(CALIBRATION_PATHS, NX, NY, debug_out=DEBUG_OUT)
    # calibrate(CALIBRATION_PATHS, NX, NY)
