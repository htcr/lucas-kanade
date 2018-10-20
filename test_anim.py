import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Video(object):
    def __init__(self, vid_data):
        # vid_data: np array, [h, w, N_frames]
        self.vid_data = vid_data
        self.frame_num = vid_data.shape[2]
        self.fig, self.ax = plt.subplots()
        
    def frame_ids(self):
        cnt = 0
        while cnt < self.frame_num:
            yield cnt
            cnt += 1

    def draw_frame(self, frame_id):
        cur_frame = self.vid_data[:, :, frame_id]
        if frame_id == 0:
            self.buff = self.ax.imshow(cur_frame, cmap='gray')
        else:
            self.buff.set_array(cur_frame)
        # draw other stuff here
        
    def play(self):
        anim = animation.FuncAnimation(self.fig, self.draw_frame, self.frame_ids, blit=False, interval=5,
                              repeat=True)
        plt.show()


car_vid = np.load('../data/aerialseq.npy')*255
car_anim = Video(car_vid)
car_anim.play()