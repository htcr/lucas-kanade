import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from svd import svd_solve
from helper import indice_from_rect, get_intp_img


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

	while True:
		p += dp
		search_row_ids = template_row_ids + p[1, 0]
		search_col_ids = template_col_ids + p[0, 0]
		
		# here, x means row, y means col
		A_c0 = It1_intp.ev(search_row_ids, search_col_ids, dx=0, dy=1)
		A_c1 = It1_intp.ev(search_row_ids, search_col_ids, dx=1, dy=0)

		# (N, 2)
		A = np.stack((A_c0, A_c1), axis=1)

		b = template_pixels - It1_intp.ev(search_row_ids, search_col_ids)
		# (N, 1)
		b = b.reshape(-1, 1)

		# (2, 1)
		dp_svd = svd_solve(A, b)
		
		print('difference with template: %f' % np.sum(np.abs(A @ dp_svd - b)))

		dp_diff = np.sum((dp_svd)**2)
		if dp_diff < thr:
			break
		dp = dp_svd

	# (2,)
	return p.reshape(-1)


