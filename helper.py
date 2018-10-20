import numpy as np
from scipy.interpolate import RectBivariateSpline

def indice_from_rect(rect, row_sample_num=None, col_sample_num=None):
    # rect: tuple, (l, t, r, b)
    # return: np float arrays of shape (N, 1), row_ids, col_ids 
    # fractional coordinates will be handled by interpolation
    l, t, r, b = rect
    # inclusive
    row_min = t
    row_max = b
    col_min = l
    col_max = r
    if row_min > row_max or col_min > col_max:
        return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.int32)

    if row_sample_num is None:
        row_ids = np.arange(row_min, row_max+1, 1, dtype=np.float32)
    else:
        assert row_sample_num > 2
        step = (row_max - row_min) / (row_sample_num - 1.0)
        row_ids = np.arange(row_min, row_max+step/2.0, step, dtype=np.float32)

    if col_sample_num is None:
        col_ids = np.arange(col_min, col_max+1, 1, dtype=np.float32)
    else:
        assert col_sample_num > 2
        step = (col_max - col_min) / (col_sample_num - 1.0)
        col_ids = np.arange(col_min, col_max+step/2.0, step, dtype=np.float32)

    col_ids_all, row_ids_all = np.meshgrid(col_ids, row_ids)
    row_ids_all, col_ids_all = row_ids_all.reshape(-1), col_ids_all.reshape(-1)
    return row_ids_all, col_ids_all


def get_intp_img(img):
    # img: np array, (h, w)
    # return: RectBivariateSpline, able to access
    # pixel at fractional coordinate; Out-of-bound
    # pixel values equal to nearest boundary values.
    h, w = img.shape[:2]
    row_ids = np.arange(0, h, 1, dtype=np.float32)
    col_ids = np.arange(0, w, 1, dtype=np.float32)
    img_intp = RectBivariateSpline(row_ids, col_ids, img, kx=2, ky=2)
    return img_intp