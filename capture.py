import os
import cv2 as cv
import numpy as np
import utils
import PIL.Image


def save_manual(src_cam: cv.VideoCapture, dst_cam: cv.VideoCapture, out_dir='.'):
    c = 0
    while True:
        # read images from both cameras
        is_ok, src_image = src_cam.read()
        if not is_ok:
            continue

        is_ok, dst_image = dst_cam.read()
        if not is_ok:
            continue

        # resize
        if src_image.shape[0] != 480 or src_image.shape[1] != 640:
            src_image = cv.resize(src_image, (640, 480))
        if dst_image.shape[0] != 480 or dst_image.shape[1] != 640:
            dst_image = cv.resize(dst_image, (640, 480))

        cv.imshow("src", src_image)
        cv.imshow("dst", dst_image)

        key = cv.waitKey(10)
        if key == 27:
            break
        elif key == 32:
            save_image(src_image, dst_image, os.path.join(out_dir, str(c)))
            c += 1

    cv.destroyAllWindows()

def save_image(src_image, dst_image, prefix=""):
    cv.imwrite(f'{prefix}.src.png', src_image)
    cv.imwrite(f'{prefix}.dst.png', dst_image)


if __name__ == "__main__":
    rgb_cam = cv.VideoCapture(0)
    infrared_cam = utils.InfraredCamera()
    save_manual(rgb_cam, infrared_cam, out_dir='./test')

    rgb_cam.release()
    infrared_cam.release()
