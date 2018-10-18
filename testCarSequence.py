import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

car_vid = np.load('../data/carseq.npy')
car_vid = car_vid.astype(np.float32)
init_rect = (59, 116, 145, 151)
all_rects = [init_rect]
prev_frame = car_vid[:, :, 0]
frame_num = car_vid.shape[2]
for i in range(1, 10):
    print('tracking frame %d' % (i))
    cur_frame = car_vid[:, :, i]
    prev_rect = all_rects[-1]
    p = LucasKanade(prev_frame, cur_frame, prev_rect)
    l, t, r, b = prev_rect
    cur_rect = (l+p[0], t+p[1], r+p[0], b+p[1])
    all_rects.append(cur_rect)
    prev_frame = cur_frame

rects = np.array(all_rects)
