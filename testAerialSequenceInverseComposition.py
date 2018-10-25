import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import os
from LucasKanadeAffine import LucasKanadeAffine
from SubtractDominantMotion import SubtractDominantMotionInverseComposition
import cv2

vid = np.load('../data/aerialseq.npy')
vid = vid.astype(np.float32)
print(vid.shape)
print(np.mean(vid))

output_dir = '../aerial_output_inverse_composition'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_num = vid.shape[2]

for i in range(1, frame_num):
    
    print('frame %d' % i)
    
    It = vid[:, :, i-1]
    It1 = vid[:, :, i]

    mask = SubtractDominantMotionInverseComposition(It, It1)

    vis = It1.copy()
    vis = np.stack((vis, vis, vis), axis=2)*255.0
    vis[:, :, 2] += (mask.astype(np.float32))*100.0

    vis = np.clip(vis, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, ('4_1_frame_%d.jpg' % i)), vis)
    
    
