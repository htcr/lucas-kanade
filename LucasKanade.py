import numpy as np
from scipy.interpolate import RectBivariateSpline

def indice_from_rect(rect, img_h, img_w):
	# rect: tuple, (l, t, r, b)
	# img_h, img_w: int, image boundary
	# return: np int arrays of shape (N, 1), row_ids, col_ids 
	l, t, r, b = rect
	# inclusive
	row_min = int(max(0, t))
	row_max = int(min(img_h-1, b))
	col_min = int(max(0, l))
	col_max = int(min(img_w-1, r))
	if row_min > row_max or col_min > col_max:
		return np.zeros((0, 1), dtype=np.int32), np.zeros((0, 1), dtype=np.int32)
	row_ids = np.arange(row_min, row_max+1, 1, dtype=np.int32)
	col_ids = np.arange(col_min, col_max+1, 1, dtype=np.int32)
	col_ids_all, row_ids_all = np.meshgrid(col_ids, row_ids)
	row_ids_all, col_ids_all = row_ids_all.reshape(-1), col_ids_all.reshape(-1)
	return row_ids_all, col_ids_all

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
	thr = 4.0
	# p is initial guess
	p = p0
	
	while True:

    return p
