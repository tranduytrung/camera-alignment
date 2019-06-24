import os
import cv2 as cv
import numpy as np
import utils
import PIL.Image


def calibrate_manual(
    src_cam: cv.VideoCapture, dst_cam: cv.VideoCapture, S=(1.0, 1.0), T=(0, 0)
):
    S = list(S)
    T = list(T)
    keep_ratio = True

    while True:
        # read images from both cameras
        is_ok, src_image = src_cam.read()
        if not is_ok:
            continue

        is_ok, dst_image = dst_cam.read()
        if not is_ok:
            continue

        # Scale
        src_image = cv.resize(src_image, None, fx=S[0], fy=S[1])

        # get BGR and Gray
        src_rgba = utils.to_rgba(src_image)
        dst_rgba = utils.to_rgba(dst_image)
        # combine
        src_rgba[:, :, 3] = 170
        src_pil = PIL.Image.fromarray(src_rgba)
        dst_pil = PIL.Image.fromarray(dst_rgba)
        dst_pil.paste(src_pil, (T[0], T[1]), src_pil)
        dst_bgr = np.array(dst_pil)

        cv.imshow("dst", dst_bgr[..., [2,1,0,3]])

        key = cv.waitKey(10)
        if key == 27:
            break
        elif key != -1:
            if key == ord("a"):
                S[0] -= 0.01
                if keep_ratio:
                    S[1] -= 0.01
            elif key == ord("d"):
                S[0] += 0.01
                if keep_ratio:
                    S[1] += 0.01
            elif key == ord("s"):
                S[1] -= 0.01
                if keep_ratio:
                    S[0] -= 0.01
            elif key == ord("w"):
                S[1] += 0.01
                if keep_ratio:
                    S[0] += 0.01
            elif key == ord("j"):
                T[0] -= 1
            elif key == ord("l"):
                T[0] += 1
            elif key == ord("i"):
                T[1] -= 1
            elif key == ord("k"):
                T[1] += 1
            elif key == ord("q"):
                keep_ratio = not keep_ratio

            print(f'S={S}, T={T}, keep_ratio={keep_ratio}')

    cv.destroyAllWindows()

    return S, T


def load_manual(path):
    if os.path.isfile(path):
        data = np.load(path)
        return data["S"], data["T"]

    return (1.0, 1.0), (0, 0)

def save_manual(path, S, T):
    np.savez(path, S=S, T=T)

if __name__ == "__main__":
    rgb_cam = cv.VideoCapture(0)
    infrared_cam = utils.InfraredCamera()

    S, T = load_manual("manual.npz")
    S, T = calibrate_manual(rgb_cam, infrared_cam, S, T)
    save_manual("manual.npz", S ,T)

    rgb_cam.release()
    infrared_cam.release()
