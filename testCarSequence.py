import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import os
import cv2

car_vid = np.load('../data/carseq.npy')
car_vid = car_vid.astype(np.float32)
init_rect = (59, 116, 145, 151)
all_rects = [init_rect]
prev_frame = car_vid[:, :, 0]
frame_num = car_vid.shape[2]

output_dir = '../car_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, frame_num):
    print('tracking frame %d' % (i))
    cur_frame = car_vid[:, :, i]
    prev_rect = all_rects[-1]
    p = LucasKanade(prev_frame, cur_frame, prev_rect)
    l, t, r, b = prev_rect
    cur_rect = (l+p[0], t+p[1], r+p[0], b+p[1])
    all_rects.append(cur_rect)

    # visualize
    cur_frame_show = np.stack((cur_frame, cur_frame, cur_frame), axis=2)*255.0
    cur_frame_show = cur_frame_show.astype(np.uint8).copy()
        
    l, t, r, b = cur_rect
    cv2.rectangle(cur_frame_show, (int(l), int(t)), (int(r), int(b)), (0, 255, 255), 2)        
    cv2.imwrite((os.path.join(output_dir, '1_3_frame_%d.png') % i), cur_frame_show)
    # visualize

    prev_frame = cur_frame

rects = np.array(all_rects)
print(rects.shape)
np.save('carseqrects.npy', rects)
