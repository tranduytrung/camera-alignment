import os
import cv2 as cv
import numpy as np
import utils
import numba

@numba.njit(numba.float32[:](numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32))
def project_to_3d(x, y, d, cx, cy, fx, fy):
    x3d = d * (x - cx) / fx
    y3d = d * (y - cy) / fy
    z3d = d
    return np.array([x3d, y3d, z3d])

@numba.njit(numba.typeof((0.0, 0.0))(numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32, numba.float32))
def project_to_2d(x, y, z, cx, cy, fx, fy):
    if z == 0:
        z = 0.001
    x2d = fx * x / z + cx
    y2d = fy * y / z + cy
    return x2d, y2d

@numba.njit(numba.int32(numba.float32, numba.float32, numba.float32))
def clip_round(value, minvalue, maxvalue):
    return int(round((min(max((value, minvalue)), maxvalue))))

@numba.njit(numba.float32[:, :](numba.float32[:, :], numba.float32[:, :], numba.float32[:, :], numba.float32[:,:], numba.float32[:], numba.typeof((0,0))))
def reproject(d_image, rgb_mat, d_mat, R, T, shape):
    buffer = np.zeros(shape, np.float32)
    h, w = d_image.shape
    dfx = d_mat[0, 0]
    dfy = d_mat[1, 1]
    dcx = d_mat[0, 2]
    dcy = d_mat[1, 2]

    vfx = rgb_mat[0, 0]
    vfy = rgb_mat[1, 1]
    vcx = rgb_mat[0, 2]
    vcy = rgb_mat[1, 2]

    for i in range(h):
        for j in range(w):
            d_value = d_image[i, j]
            if d_value <= 0.1:
                continue

            p3d = project_to_3d(j, i, d_value, dcx, dcy, dfx, dfy)
            p3d = np.dot(R, p3d) + T
            p2d = project_to_2d(p3d[0], p3d[1], p3d[2], vcx, vcy, vfx, vfy)

            x2d = clip_round(p2d[0], 0, shape[1] - 1)
            y2d = clip_round(p2d[1], 0, shape[0] - 1)

            if d_value < buffer[y2d, x2d] or buffer[y2d, x2d] == 0:
                buffer[y2d, x2d] = d_value

    return buffer


def show(rgb_cam: cv.VideoCapture, rgb_mat, rgb_dist, d_cam: utils.DepthCamera, d_mat, d_dist, R, T):
    while True:
        # read images from both cameras
        is_ok, bgr_image = rgb_cam.read()
        if not is_ok:
            continue

        is_ok, d_image = d_cam.read()
        if not is_ok:
            continue

        # clean inf
        nan_mask = np.isnan(d_image)
        d_image[nan_mask] = -1

        # scale
        d_image = cv.resize(d_image, (bgr_image.shape[1], bgr_image.shape[0]))

        # undistort
        bgr_image = cv.undistort(bgr_image, rgb_mat, rgb_dist)
        d_image = cv.undistort(d_image, d_mat, d_dist)        

        # reproject
        depth = reproject(d_image, rgb_mat, d_mat, R, T, tuple(bgr_image.shape[:2]))

        # convert
        depth = np.clip(depth, 300, 1000)
        d_map = utils.bytescaling(depth)
        
        d_image = np.clip(d_image, 300, 1000)
        d_gray = utils.bytescaling(d_image)

        # combine
        combine = (cv.cvtColor(d_map, cv.COLOR_GRAY2BGR)*0.7 + bgr_image*.3 ).astype(np.uint8)
        mask = d_map <= 0
        combine[mask] = 0

        # draw
        cv.imshow('rgb', bgr_image)
        cv.imshow('d', d_gray)
        cv.imshow('mapped', d_map)
        cv.imshow('combine', combine)
        
        key = cv.waitKey(10)
        if key == 27:
            break

    cv.destroyAllWindows()

def load_intrinsic(instrinsic_file):
    data = np.load(instrinsic_file)
    mtx = data["camera_matrix"].astype(np.float32)
    dist = data["distortion"].astype(np.float32)
    return mtx, dist

def load_RT(path):
    data = np.load(path)
    R = data["R"].astype(np.float32)
    T = data["T"].astype(np.float32)
    return R, T

if __name__ == "__main__":
    d_mtx, d_dist = load_intrinsic("d_intrinsic.npz")
    rgb_mtx, rgb_dist = load_intrinsic("rgb_intrinsic.npz")
    R, T = load_RT('recify.npz')
    rgb_cam = cv.VideoCapture(0)
    d_cam = utils.DepthCamera()

    T = T.flatten()
    # T[2] = 0
    # T = -T
    show(rgb_cam, rgb_mtx, rgb_dist, d_cam, d_mtx, d_dist, R, T)    

    rgb_cam.release()
    d_cam.release()
