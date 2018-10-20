import numpy as np
from scipy.interpolate import RectBivariateSpline
from helper import get_intp_img, indice_from_rect
from svd import svd_solve

def LucasKanadeBasis(It, It1, rect, bases, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # stop when ||delta_p||^2 < thr
	thr = 0.001
	# p is initial guess
	# (2, 1)
	p = p0.reshape(-1, 1)
	dp = np.zeros((2, 1), dtype=np.float32)

	# pre-calculate template
	It_intp = get_intp_img(It)
	template_row_ids, template_col_ids = indice_from_rect(rect, bases.shape[0], bases.shape[1])
	# (N, )
	template_pixels = It_intp.ev(template_row_ids, template_col_ids)

	It1_intp = get_intp_img(It1)

	# prepare basis
	B = bases.reshape(-1, bases.shape[2])
	BBt = B @ B.transpose()
	B_multiplier = np.eye(BBt.shape[0]) - BBt

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

		# add bases
		A = B_multiplier @ A
		b = B_multiplier @ b

		# (2, 1)
		dp_svd = svd_solve(A, b)
		
		print('difference with template: %f' % np.sum(np.abs(A @ dp_svd - b)))

		dp_diff = np.sum((dp_svd)**2)
		if dp_diff < thr:
			break
		dp = dp_svd

	# (2,)
	return p.reshape(-1)