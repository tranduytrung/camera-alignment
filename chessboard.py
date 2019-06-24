import numpy as np
import cv2, sys
from occipital import StructureCamera
import utils

def show_rgb2infrared(h, invert=False):
    # capture from camera
    infrared_cam = utils.InfraredCamera()
    rgb_cam = cv2.VideoCapture(1)
    # rgb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # rgb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (6, 6)

    while True:
        # RGB Camera
        ret_rgb, rgb_image = rgb_cam.read()
        if not ret_rgb:
            continue
        # Infrared Camera
        ret_infrared, infrared_image = infrared_cam.read()
        if not ret_infrared:
            continue
        infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)

        # transform
        if invert:
            height, width = rgb_image.shape[:2]
            transformed = cv2.warpPerspective(infrared_image, h, (width, height), flags=cv2.WARP_INVERSE_MAP)

            # combine
            mask = transformed.sum(axis=2) != 0
            combined = rgb_image.copy()
            combined[mask] = transformed[mask]*0.5 + rgb_image[mask]*0.5
        else:
            height, width = infrared_image.shape[:2]
            transformed = cv2.warpPerspective(rgb_image, h, (width, height))

            # combine
            mask = transformed.sum(axis=2) != 0
            combined = infrared_image.copy()
            combined[mask] = transformed[mask]*0.5 + infrared_image[mask]*0.5

        # show
        cv2.imshow("calibrated", combined)
        cv2.imshow('rgb', rgb_image)
        cv2.imshow('infrared', infrared_image)

        wait_time = 10
        key = cv2.waitKey(wait_time)
        if key == 27:
            break

    rgb_cam.release()
    infrared_cam.release()
    cv2.destroyAllWindows()


def find_chessboard(show_infrared=True, show_rgb=True):
    assert show_infrared or show_rgb

    # capture from camera
    if show_infrared:
        infrared_cam = utils.InfraredCamera()

    if show_rgb:
        rgb_cam = cv2.VideoCapture(1)
        # rgb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # rgb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_size = (6, 9)

    while True:
        # RGB Camera
        if show_rgb:
            ret_vc, rgb_image = rgb_cam.read()
            if ret_vc:
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret_rgb, corners = cv2.findChessboardCorners(
                    gray, chessboard_size, None
                )
                # If found, add object points, image points (after refining them)
                if ret_rgb == True:
                    rgb_corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )

                    # Draw and display the corners
                    cv2.drawChessboardCorners(
                        rgb_image, chessboard_size, rgb_corners, ret_rgb
                    )
                cv2.imshow("RGB Camera", rgb_image)

        if show_infrared:
            ret_vc, gray = infrared_cam.read()
            if ret_vc:
                infrared_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Find the chess board corners
                ret_infrared, corners = cv2.findChessboardCorners(
                    gray, chessboard_size, None
                )
                # If found, add object points, image points (after refining them)
                if ret_infrared == True:
                    infrared_corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )

                    # Draw and display the corners
                    cv2.drawChessboardCorners(
                        infrared_image, chessboard_size, infrared_corners, ret_rgb
                    )
                cv2.imshow("Infrared Camera", infrared_image)

        wait_time = 10
        if ret_infrared and ret_rgb:
            wait_time = 10000
        key = cv2.waitKey(wait_time)
        if key == 27:
            break


    if show_rgb:
        rgb_cam.release()
    if show_infrared:
        infrared_cam.release()
    cv2.destroyAllWindows()
    return rgb_corners, infrared_corners, rgb_image, infrared_image

def load_transform():
    return np.load(r'rgb2infrared.npy')

if __name__ == "__main__":
    # rgb_corners, infrared_corners, rgb_image, infrared_image = find_chessboard()
    # h, mask = cv2.findHomography(rgb_corners, infrared_corners, cv2.RANSAC)
    # np.save(r"rgb2infrared", h)

    h = load_transform()

    show_rgb2infrared(h, invert=True)
    cv2.destroyAllWindows()
