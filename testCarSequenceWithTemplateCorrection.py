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
init_frame = car_vid[:, :, 0]
frame_num = car_vid.shape[2]

naive_track_result = np.load('carseqrects.npy')

for i in range(1, frame_num):
    print('tracking frame %d' % (i))
    cur_frame = car_vid[:, :, i]
    prev_rect = all_rects[-1]
    p = LucasKanade(prev_frame, cur_frame, prev_rect)
    l, t, r, b = prev_rect
    cur_rect = (l+p[0], t+p[1], r+p[0], b+p[1])
    
    # align again with initial template
    # to tackle drift
    print('refining')
    p_from_init_x, p_from_init_y = cur_rect[0] - init_rect[0], cur_rect[1] - init_rect[1]
    pa = LucasKanade(init_frame, cur_frame, init_rect, 
        np.array((p_from_init_x, p_from_init_y), dtype=np.float32))
    l, t, r, b = init_rect
    cur_rect_refined = (l+pa[0], t+pa[1], r+pa[0], b+pa[1])
    
    all_rects.append(cur_rect_refined)

    if i in (1, 100, 200, 300, 400):
        fig, ax = plt.subplots(1)
        ax.imshow(cur_frame, cmap='gray')
        l, t, r, b = naive_track_result[i, :]
        l1, t1, r1, b1 = cur_rect_refined
        rect_patch = patches.Rectangle((l, t), r-l, b-t, linewidth=1, edgecolor='y', facecolor='none')
        rect_patch_refined = patches.Rectangle((l1, t1), r1-l1, b1-t1, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_patch)
        ax.add_patch(rect_patch_refined)
        #plt.show()
        plt.savefig(('1_4_frame_%d.png' % i))

    prev_frame = cur_frame

rects = np.array(all_rects)
print(rects.shape)
np.save('carseqrects-wcrt.npy', rects)
