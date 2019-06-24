import numpy as np
import os
import cv2 as cv
import utils


def calibrate_stereo(lcam, rcam, mtx1, mtx2, dist1, dist2, size=(640, 480), cell_size=59):
    chessboard_size = (6, 9)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(
        -1, 2
    )
    objp = objp*cell_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    limgpoints = []  # 2d points in left image plane.
    rimgpoints = []  # 2d points in right image plane.

    n_captured_images = 0
    total_images = 30
    while n_captured_images < total_images:
        # LEFT
        is_ok, l_bgr_image = lcam.read()
        if not is_ok:
            continue

        if l_bgr_image.shape[0] != size[1] or l_bgr_image.shape[1] != size[0]:
            l_bgr_image = cv.resize(l_bgr_image, size, interpolation=cv.INTER_LINEAR)

        if len(l_bgr_image.shape) == 2:
            l_gray = l_bgr_image
            l_bgr_image = cv.cvtColor(l_bgr_image, cv.COLOR_GRAY2BGR)
        else:
            l_gray = cv.cvtColor(l_bgr_image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        l_is_found, l_corners = cv.findChessboardCorners(l_gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        if l_is_found:
            l_corners2 = cv.cornerSubPix(
                l_gray, l_corners, (11, 11), (-1, -1), criteria
            )
            # Draw and display the corners
            cv.drawChessboardCorners(
                l_bgr_image, chessboard_size, l_corners2, l_is_found
            )
        # Draw counter
        cv.putText(
            l_bgr_image,
            f"{n_captured_images}/{total_images}",
            (0, l_bgr_image.shape[0]),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
        )
        cv.imshow("left", l_bgr_image)

        # RIGHT
        is_ok, r_bgr_image = rcam.read()
        if not is_ok:
            continue

        if r_bgr_image.shape[0] != size[1] or r_bgr_image.shape[1] != size[0]:
            r_bgr_image = cv.resize(r_bgr_image, size, interpolation=cv.INTER_LINEAR)

        if len(r_bgr_image.shape) == 2:
            r_gray = r_bgr_image
            r_bgr_image = cv.cvtColor(r_bgr_image, cv.COLOR_GRAY2BGR)
        else:
            r_gray = cv.cvtColor(r_bgr_image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        r_is_found, r_corners = cv.findChessboardCorners(r_gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        if r_is_found:
            r_corners2 = cv.cornerSubPix(
                r_gray, r_corners, (11, 11), (-1, -1), criteria
            )
            # Draw and display the corners
            cv.drawChessboardCorners(
                r_bgr_image, chessboard_size, r_corners2, r_is_found
            )
        # Draw counter
        cv.putText(
            r_bgr_image,
            f"{n_captured_images}/{total_images}",
            (0, r_bgr_image.shape[0]),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
        )
        cv.imshow("right", r_bgr_image)

        # all
        wait_time = 10
        if l_is_found and r_is_found:
            wait_time = 30000

        key = cv.waitKey(wait_time)
        if key == 32 and l_is_found and r_is_found:
            n_captured_images += 1
            objpoints.append(objp)
            limgpoints.append(l_corners2)
            rimgpoints.append(r_corners2)

    cv.destroyAllWindows()

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
        objpoints, limgpoints, rimgpoints, mtx1, dist1, mtx2, dist2, None, flags=cv.CALIB_FIX_INTRINSIC
    )

    return retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

def stereo_rectify(mtx1, dist1, mtx2, dist2, size, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, size, R, T, alpha=1, flags=cv.CALIB_ZERO_DISPARITY)
    return R1, R2, P1, P2, Q

def recify(cam1, cam2, mtx1, dist1, R1, P1, mtx2, dist2, R2, P2, size):
    mapx1, mapy1 = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, size, cv.CV_32F)
    mapx2, mapy2 = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, size, cv.CV_32F)
    mapy1 = mapy1 - 500

    while True:
        is_ok, bgr_image = cam1.read()
        if is_ok:
            if bgr_image.shape[0] != size[1] or bgr_image.shape[1] != size[0]:
                bgr_image = cv.resize(bgr_image, size, interpolation=cv.INTER_LINEAR)
            # reproject
            projected_bgr = cv.remap(bgr_image, mapx1, mapy1, cv.INTER_LINEAR)
            cv.imshow('cam1', projected_bgr)

        is_ok, bgr_image = cam2.read()
        if is_ok:
            if bgr_image.shape[0] != size[1] or bgr_image.shape[1] != size[0]:
                bgr_image = cv.resize(bgr_image, size, interpolation=cv.INTER_LINEAR)
            # reproject
            projected_bgr = cv.remap(bgr_image, mapx2, mapy2, cv.INTER_LINEAR)
            cv.imshow('cam2', projected_bgr)

        key = cv.waitKey(10)
        if key == 27:
            break

    cv.destroyAllWindows()

def load_intrinsic(instrinsic_file):
    data = np.load(instrinsic_file)
    mtx = data["camera_matrix"]
    dist = data["distortion"]
    return mtx, dist

def load_recify(path):
    data = np.load(path)
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q = data["Q"]
    R = data['R']
    T = data['T']
    return R1, R2, P1, P2, Q, R, T

if __name__ == "__main__":
    src_mtx, src_dist = load_intrinsic("d_intrinsic.npz")
    dst_mtx, dst_dist = load_intrinsic("rgb_intrinsic.npz")
    recify_file = 'recify.npz'

    dst_cam = cv.VideoCapture(0)
    src_cam = utils.InfraredCamera()

    if os.path.isfile(recify_file):
        R1, R2, P1, P2, Q, R, T = load_recify(recify_file)
    else:
        retval, src_mtx, src_dist, dst_mtx, dst_dist, R, T, E, F = calibrate_stereo(
            src_cam, dst_cam, src_mtx, dst_mtx, src_dist, dst_dist
        )

        
        np.savez(recify_file, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, R=R, T=T)

    R1, R2, P1, P2, Q = stereo_rectify(src_mtx, src_dist, dst_mtx, dst_dist, (640, 480), R, T)

    # print(f'R={R} T={T}')
    recify(src_cam, dst_cam, src_mtx, src_dist, R1, P1, dst_mtx, dst_dist, R2, P2, (640, 480))

    dst_cam.release()
    src_cam.release()


