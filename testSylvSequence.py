import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis
import os
import cv2

vid = np.load('../data/sylvseq.npy')
vid = vid.astype(np.float32)
init_rect = (101, 61, 155, 107)

all_rects_naive = [init_rect]
all_rects_bases = [init_rect]

prev_frame = vid[:, :, 0]
frame_num = vid.shape[2]

bases = np.load('../data/sylvbases.npy')

output_dir = '../sylv_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(1, frame_num):
    print('tracking frame %d' % (i))
    cur_frame = vid[:, :, i]
    
    prev_rect_naive = all_rects_naive[-1]
    prev_rect_bases = all_rects_bases[-1]

    print('---naive---')
    p_naive = LucasKanade(prev_frame, cur_frame, prev_rect_naive)
    print('---bases---')
    p_bases = LucasKanadeBasis(prev_frame, cur_frame, prev_rect_bases, bases)

    l, t, r, b = prev_rect_naive
    cur_rect_naive = (l+p_naive[0], t+p_naive[1], r+p_naive[0], b+p_naive[1])
    all_rects_naive.append(cur_rect_naive)

    l, t, r, b = prev_rect_bases
    cur_rect_bases = (l+p_bases[0], t+p_bases[1], r+p_bases[0], b+p_bases[1])
    all_rects_bases.append(cur_rect_bases)

    # visualize
    cur_frame_show = np.stack((cur_frame, cur_frame, cur_frame), axis=2)*255.0
    cur_frame_show = cur_frame_show.astype(np.uint8).copy()

    l, t, r, b = map(int, cur_rect_naive)
    l1, t1, r1, b1 = map(int, cur_rect_bases)
    cv2.rectangle(cur_frame_show, (l, t), (r, b), (0, 255, 255), 2)        
    cv2.rectangle(cur_frame_show, (l1, t1), (r1, b1), (0, 255, 0), 2)        
    cv2.imwrite((os.path.join(output_dir, '2_3_frame_%d.png') % i), cur_frame_show)
    # visualize
    
    prev_frame = cur_frame

rects = np.array(all_rects_bases)
print(rects.shape)
np.save('sylvseqrects.npy', rects)
