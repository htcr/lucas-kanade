import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from LucasKanadeBasis import LucasKanadeBasis

vid = np.load('../data/sylvseq.npy')
vid = vid.astype(np.float32)
init_rect = (101, 61, 155, 107)

all_rects_naive = [init_rect]
all_rects_bases = [init_rect]

prev_frame = vid[:, :, 0]
frame_num = vid.shape[2]

bases = np.load('../data/sylvbases.npy')

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

    if i in (1, 100, 200, 300, 350, 400):
        fig, ax = plt.subplots(1)
        ax.imshow(cur_frame, cmap='gray')
        
        l, t, r, b = cur_rect_naive
        rect_patch_naive = patches.Rectangle((l, t), r-l, b-t, linewidth=1, edgecolor='y', facecolor='none')
        
        l1, t1, r1, b1 = cur_rect_bases
        rect_patch_bases = patches.Rectangle((l1, t1), r1-l1, b1-t1, linewidth=1, edgecolor='g', facecolor='none')

        ax.add_patch(rect_patch_naive)
        ax.add_patch(rect_patch_bases)
        
        #plt.show()
        plt.savefig(('2_3_frame_%d.png' % i))

    prev_frame = cur_frame

rects = np.array(all_rects_bases)
print(rects.shape)
np.save('sylvseqrects.npy', rects)
