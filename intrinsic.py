import numpy as np
import os
import cv2 as cv
import utils

def calibrate_intrinsic(cam, size=(640, 480), cell_size=59):
    chessboard_size = (6, 9)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    # cell size
    objp = objp*cell_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    n_captured_images = 0
    total_images = 15
    while n_captured_images < total_images:
        is_ok, bgr_image = cam.read()
        if not is_ok:
            continue

        if bgr_image.shape[0] != size[1] or bgr_image.shape[1] != size[0]:
                bgr_image = cv.resize(bgr_image, size, interpolation=cv.INTER_LINEAR)

        if len(bgr_image.shape) == 2:
            gray = bgr_image
            bgr_image = cv.cvtColor(bgr_image, cv.COLOR_GRAY2BGR)
        else:
            gray = cv.cvtColor(bgr_image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        is_found, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        wait_time = 10
        if is_found:
            n_captured_images += 1
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(bgr_image, chessboard_size, corners2, is_found)
            wait_time = 100000
        # Draw counter
        cv.putText(bgr_image, f'{n_captured_images}/{total_images}', (0, bgr_image.shape[0]), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
        cv.imshow("img", bgr_image)
        
        cv.waitKey(wait_time)

    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def show_undistorted(mtx, dist, cam, size=(640, 480)):
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)

    while True:
        is_ok, bgr_image = cam.read()
        if not is_ok:
            continue

        # undistort
        if bgr_image.shape[0] != size[1] or bgr_image.shape[1] != size[0]:
            bgr_image = cv.resize(bgr_image, size, interpolation=cv.INTER_LINEAR)
        dst = cv.undistort(bgr_image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imshow('undistorted', dst)
        key = cv.waitKey(10)
        if key == 27:
            break

    cam.release()
    cv.destroyAllWindows()
        

if __name__ == "__main__":
    # rgb_instrinsic_file = 'rgb_intrinsic.npz'
    # cam = cv.VideoCapture(0)
    # if os.path.isfile(rgb_instrinsic_file):
    #     data = np.load(rgb_instrinsic_file)
    #     mtx = data['camera_matrix']
    #     dist = data['distortion']
    # else:
    #     ret, mtx, dist, rvecs, tvecs = calibrate_intrinsic(cam)
    #     np.savez(rgb_instrinsic_file, camera_matrix=mtx, distortion=dist)

    d_instrinsic_file = 'd_intrinsic.npz'
    cam = utils.InfraredCamera()
    if os.path.isfile(d_instrinsic_file):
        data = np.load(d_instrinsic_file)
        mtx = data['camera_matrix']
        dist = data['distortion']
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate_intrinsic(cam)
        np.savez(d_instrinsic_file, camera_matrix=mtx, distortion=dist)

    print(f'mtx={mtx} dist={dist}')
    show_undistorted(mtx, dist, cam, size=(640, 480))
    cam.release()


