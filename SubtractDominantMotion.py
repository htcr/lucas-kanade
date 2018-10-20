import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
import cv2

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    
	thr = 0.1

	M_23 = LucasKanadeAffine(image1, image2)
	M_33 = np.eye(3, dtype=np.float32)
	M_33[0:2, 0:3] = M_23 # (3, 3)

	#M_inv = np.linalg.inv(M_33)
	M_inv = cv2.invertAffineTransform(M_23)

	#image1_warped = affine_transform(image1, M_inv, output_shape=image2.shape)
	image1_warped = cv2.warpAffine(image1, M_23, (image2.shape[1], image2.shape[0]))

	'''
	cv2.imwrite('i1.jpg', image1*255)
	cv2.imwrite('i1w.jpg', image1_warped*255)
	cv2.imwrite('i2.jpg', image2*255)
	'''

	image_diff = np.abs(image2 - image1_warped)

	mask = image_diff > thr
	return mask

if __name__=='__main__':
	vid = np.load('../data/aerialseq.npy')
	vid = vid.astype(np.float32)

	i1 = vid[:, :, 0]
	i2 = vid[:, :, 0].copy()
	i2[:, 10:] = i2[:, :-10]
	mask = SubtractDominantMotion(i1, i2)