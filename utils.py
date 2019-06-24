import numpy as np
from occipital import StructureCamera
import cv2 as cv

def bytescaling(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


class InfraredCamera(StructureCamera):
    def __init__(self):
        super(InfraredCamera, self).__init__()
        self.infrared_auto_exposure = True
        self.infrared_enabled = True
        self.infrared_mode = StructureCamera.SC_INFRARED_MODE_RIGHT

        self.start()

    def release(self):
        self.stop()

    def read(self):
        return True, bytescaling(self.last_infrared_frame())

class DepthCamera(StructureCamera):
    def __init__(self):
        super(DepthCamera, self).__init__()
        self.infrared_auto_exposure = True
        self.depth_range = StructureCamera.SC_DEPTH_RANGE_LONG
        self.calibration_mode = StructureCamera.SC_CALIBRATION_ONESHOT
        self.infrared_mode = StructureCamera.SC_INFRARED_MODE_RIGHT
        self.depth_resolution = StructureCamera.SC_RESOLUTION_SXGA

        self.start()

    def release(self):
        self.stop()

    def read(self):
        return True, self.last_depth_frame()

def to_bgr_gray(cv_image):
    if len(cv_image.shape) == 2:
        bgr_image = cv.cvtColor(cv_image, cv.COLOR_GRAY2BGR)
        return bgr_image, cv_image
    return cv_image, cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY)

def to_rgba(cv_image):
    if len(cv_image.shape) == 2:
        return cv.cvtColor(cv_image, cv.COLOR_GRAY2RGBA)
    return cv.cvtColor(cv_image, cv.COLOR_BGR2RGBA)


def get_chessboard_corners(cv_gray, chessboard_size):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_ok, corners = cv.findChessboardCorners(cv_gray, chessboard_size)
    if not corners_ok:
        return False, corners
    refined_corners = cv.cornerSubPix(cv_gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined_corners

def resize(cv_image, size):
    if cv_image.shape[0] != size[1] or cv_image.shape[1] != size[0]:
        return cv.resize(cv_image, size, interpolation=cv.INTER_LINEAR)
    return cv_image

# not yet
def paste(src, dst, pos=(0, 0), opacity=0.5):
    plate = np.copy(dst)
    sh, sw = src.shape[:2]

    yd = min((pos[0] + src.shape[0], plate.shape[0]))
    xd = min((pos[1] + src.shape[1], plate.shape[1]))
    ys = yd - pos[0]
    xs = xd - pos[1]

    if pos[0] > yd or pos[1] > xd:
        return plate

    plate[pos[0]:yd, pos[1]:xd] = src[0]
