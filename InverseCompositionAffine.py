import numpy as np
from scipy.interpolate import RectBivariateSpline
from svd import svd_solve
from helper import get_intp_img

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    
	p = np.zeros((6, 1), dtype=np.float32)
	dp = np.zeros((6, 1), dtype=np.float32)
	thr = 0.0001

	M = np.eye(3, dtype=np.float32) # (3, 3) get initial M from initial p 
	
	# get It_intp
	It_intp = get_intp_img(It)

	# get It_xs, It_ys
	It_h, It_w = It.shape[0:2]
	It_xs = np.arange(0, It_w, 1, dtype=np.float32)
	It_ys = np.arange(0, It_h, 1, dtype=np.float32)
	It_xs, It_ys = np.meshgrid(It_xs, It_ys)
	It_xs, It_ys = It_xs.reshape(-1), It_ys.reshape(-1)
	It_ones = np.ones((It_xs.shape[0],), dtype=np.float32)

	template_points = np.stack((It_xs, It_ys, It_ones), axis=0) # all points in It, homogeneous, (3, 1)

	template_row_ids, template_col_ids = It_ys, It_xs

	# get dIt
	It_dx = It_intp.ev(template_row_ids, template_col_ids, dx=0, dy=1) # (N, )
	It_dy = It_intp.ev(template_row_ids, template_col_ids, dx=1, dy=0) # (N, )

	# (2N, 1) Construct from It1_dx and It1_dy
	A_part1 = np.stack((It_dx, It_dy), axis=1) # (N, 2)
	A_part1 = A_part1.reshape(-1, 1) # (2N, 1) 

	# (2N, 6) Construct from template points
	A_part2_zeros = np.zeros((template_points.shape[1], template_points.shape[0]), dtype=np.float32) # (N, 3)
	A_part2 = np.concatenate((template_points.transpose(), A_part2_zeros, A_part2_zeros, template_points.transpose()), axis=1) # (N, 12)
	A_part2 = A_part2.reshape(-1, 6) # (2N, 6)

	# assemble A (N, 6)
	A = A_part1 * A_part2 # (2N, 6)
	A = A[0::2, :] + A[1::2, :] # (N, 6)

	# get template pixels (N, 1)
	template_pixels = It_intp.ev(template_row_ids, template_col_ids).reshape(-1, 1)

	# get It1_intp
	It1_intp = get_intp_img(It1)

	while True:
		# update M
		dM = np.eye(3, dtype=np.float32)
		dM[0, 0:3] += dp[0:3, 0]
		dM[1, 0:3] += dp[3:6, 0]
		M = M @ np.linalg.inv(dM)
		
		# get search points (3, N), homogeneous
		search_points = M @ template_points
		search_points = search_points / search_points[2, :]

		# points to row and col ids
		search_row_ids, search_col_ids = search_points[1, :], search_points[0, :]

		# get search pixels (N, 1)
		search_pixels = It1_intp.ev(search_row_ids, search_col_ids).reshape(-1, 1)

		# evaluate difference and print
		pixel_diff = np.sum((template_pixels - search_pixels)**2)
		print('difference with template: %f' % pixel_diff)

		# (N, 1) It(search points) - It1(template points)
		b = search_pixels - template_pixels # (N, 1)
		
		dp_LSE = np.linalg.inv((A.transpose() @ A)) @ A.transpose() @ b
		
		if np.sum((dp_LSE - dp)**2) < thr:
			break
		dp = dp_LSE

	# remove last row from M
	return M[0:2, :]