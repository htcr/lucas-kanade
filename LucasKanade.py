import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import numpy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.patches as patches

eps = 0.00001

def svd(A):
    u, s, vh = la.svd(A)
    S = np.zeros(A.shape)
    S[:s.shape[0], :s.shape[0]] = np.diag(s)
    return u, S, vh

def inverse_sigma(S):
    inv_S = S.copy().transpose()
    for i in range(min(S.shape)):
        if abs(inv_S[i, i]) > eps :
            inv_S[i, i] = 1.0/inv_S[i, i]
    return inv_S

def svd_solve(A, b):
    U, S, Vt = svd(A)
    inv_S = inverse_sigma(S)
    svd_solution = Vt.transpose() @ inv_S @ U.transpose() @ b
    return svd_solution


def indice_from_rect(rect):
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
	row_ids = np.arange(row_min, row_max+1, 1, dtype=np.float32)
	col_ids = np.arange(col_min, col_max+1, 1, dtype=np.float32)
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


def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top-left, bot-right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    # stop when ||delta_p||^2 < thr
	thr = 0.001
	# p is initial guess
	# (2, 1)
	p = p0.reshape(-1, 1)
	dp = np.zeros((2, 1), dtype=np.float32)
	
	h, w = It.shape[:2]
	h1, w1 = It.shape[:2]
	
	# pre-calculate template
	It_intp = get_intp_img(It)
	template_row_ids, template_col_ids = indice_from_rect(rect)
	# (N, )
	template_pixels = It_intp.ev(template_row_ids, template_col_ids)

	It1_intp = get_intp_img(It1)

	#It1_dx = cv2.Sobel(It1, cv2.CV_32F, 1, 0)
	#It1_dy = cv2.Sobel(It1, cv2.CV_32F, 0, 1)
	
	#It1_dx_intp = get_intp_img(It1_dx)
	#It1_dy_intp = get_intp_img(It1_dy)
	
	while True:
		p += dp
		search_row_ids = template_row_ids + p[1, 0]
		search_col_ids = template_col_ids + p[0, 0]

		#A_c0 = It1_dx_intp.ev(search_row_ids, search_col_ids)
		#A_c1 = It1_dy_intp.ev(search_row_ids, search_col_ids)
		
		A_c0 = It1_intp.ev(search_row_ids, search_col_ids, dx=0, dy=1)
		A_c1 = It1_intp.ev(search_row_ids, search_col_ids, dx=1, dy=0)

		# (N, 2)
		A = np.stack((A_c0, A_c1), axis=1)

		b = template_pixels - It1_intp.ev(search_row_ids, search_col_ids)
		# (N, 1)
		b = b.reshape(-1, 1)

		# (2, 1)
		dp_svd = svd_solve(A, b)
		
		print(np.sum(np.abs(A @ dp_svd - b)))

		dp_diff = np.sum((dp_svd)**2)
		if dp_diff < thr:
			break
		dp = dp_svd

	# (2,)
	return p.reshape(-1)


