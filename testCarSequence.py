import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

car_vid = np.load('../data/carseq.npy')
car_vid = car_vid.astype(np.float32)
init_rect = (59, 116, 145, 151)
f0 = car_vid[:, :, 0]
f1 = car_vid[:, :, 1]
p = LucasKanade(f0, f1, init_rect)
print(p)