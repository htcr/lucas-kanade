import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    
	thr = 50.0

	M_23 = LucasKanadeAffine(image1, image2)
	M_33 = np.eye(3, dtype=np.float32)
	M_33[0:2, 0:3] = M_23 # (3, 3)

	M_inv = np.linalg.inv(M_33)

	image1_warped = affine_transform(image1, M_inv, output_shape=image2.shape)

	image_diff = np.abs(image2 - image1)

	mask = image_diff > thr
    return mask
