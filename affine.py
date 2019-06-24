import os
import cv2 as cv
import numpy as np
import utils

def calibrate_affine(
    src_cam: cv.VideoCapture,
    dst_cam: cv.VideoCapture,
    n_images=10,
    chessboard_size=(6, 9),
    size=(640, 480),
):
    src_points = []
    dst_points = []
    n_captured_images = 0
    while n_captured_images < n_images:
        # read images from both cameras
        is_ok, src_image = src_cam.read()
        if not is_ok:
            continue

        is_ok, dst_image = dst_cam.read()
        if not is_ok:
            continue

        # resize
        src_resized = utils.resize(src_image, size)
        dst_resized = utils.resize(dst_image, size)

        # get BGR and Gray
        src_bgr, src_gray = utils.to_bgr_gray(src_resized)
        dst_bgr, dst_gray = utils.to_bgr_gray(dst_resized)

        # get corners
        src_corners_ok, src_corners = utils.get_chessboard_corners(src_gray, chessboard_size)
        dst_corners_ok, dst_corners = utils.get_chessboard_corners(dst_gray, chessboard_size)

        # draw
        cv.drawChessboardCorners(src_bgr, chessboard_size, src_corners, src_corners_ok)
        cv.drawChessboardCorners(dst_bgr, chessboard_size, dst_corners, dst_corners_ok)
        cv.putText(
            src_bgr,
            f"{n_captured_images}/{n_images}",
            (0, src_bgr.shape[0]),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
        )

        cv.putText(
            dst_bgr,
            f"{n_captured_images}/{n_images}",
            (0, dst_bgr.shape[0]),
            cv.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 0, 0),
        )

        cv.imshow('src', src_bgr)
        cv.imshow('dst', dst_bgr)

        # save if ok
        both_ok = src_corners_ok and dst_corners_ok
        wait_time = 10
        if both_ok:
            wait_time = -1

        key = cv.waitKey(wait_time)
        if key == 32 and both_ok:
            n_captured_images += 1
            src_points.append(src_corners)
            dst_points.append(dst_corners)

    cv.destroyAllWindows()
    # calc affine transform
    np_src_points = np.reshape(src_points, (-1, 2))
    np_dst_points = np.reshape(dst_points, (-1, 2))
    trans_mat, inliers = cv.estimateAffinePartial2D(
        np_src_points, np_dst_points, method=cv.RANSAC
    )

    return trans_mat


def test(src_cam: cv.VideoCapture, dst_cam: cv.VideoCapture, affine_mat, size=(640, 480)):
    while True:
        # read images from both cameras
        is_ok, src_image = src_cam.read()
        if not is_ok:
            continue

        is_ok, dst_image = dst_cam.read()
        if not is_ok:
            continue

        # resize
        src_resized = utils.resize(src_image, size)
        dst_resized = utils.resize(dst_image, size)

        # transform
        src_resized = cv.warpAffine(src_resized, affine_mat, dst_resized.shape[::-1], flags=cv.INTER_LINEAR)

        # get BGR and Gray
        src_bgr, src_gray = utils.to_bgr_gray(src_resized)
        dst_bgr, dst_gray = utils.to_bgr_gray(dst_resized)



        # combine
        combine = (0.5*src_bgr + 0.5*dst_bgr).astype(np.uint8)

        # draw
        cv.imshow('combine', combine)
        
        key = cv.waitKey(10)
        if key == 27:
            break

    cv.destroyAllWindows()

def load_affine(path):
    if os.path.isfile(path):
        return np.load(path)

    return None

if __name__ == "__main__":
    rgb_cam = cv.VideoCapture(0)
    infrared_cam = utils.InfraredCamera()

    affine = load_affine('affine.npy')
    if affine is None:
        affine = calibrate_affine(rgb_cam, infrared_cam, 1)
        np.save('affine.npy', affine)

    test(rgb_cam, infrared_cam, affine)

    rgb_cam.release()
    infrared_cam.release()
