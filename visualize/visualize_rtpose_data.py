import numpy as np
import os
import os.path as osp
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

rtpose_hm_path_tail = "radar/npy_DZYX_real"

x_min  = 0
x_max  = 10.0235484375

y_min  = -4.9064531250000005
y_max  = 4.9064531250000005

z_min  = -2.5965
z_max  = 2.5965

line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

class RTVisualizer:
    def __init__(self, gt_json, dir, draw_doppler=0):
        self.paused = [False]
        self.gt_json = gt_json
        self.dir = dir
        self.draw_doppler = draw_doppler
        self.plt_holders = []
        self.fontsize=15
        self.view_dist=9

        self.first_frame = True
        self.fig = plt.figure(figsize=(12,10))
        self.fig.tight_layout(pad=0.5)

        self.ax = []
        self.ax.append(plt.subplot(221, projection="3d"))
        self.ax[0].view_init(elev=10, azim=-165)
        self.ax[0].dist = self.view_dist
        self.ax[0].set_xlabel('X-metre', fontsize=self.fontsize)
        self.ax[0].set_ylabel('Y-metre', fontsize=self.fontsize)
        self.ax[0].set_zlabel('Z-metre', fontsize=self.fontsize)
        self.ax[0].set_xlim(2, 7.0)
        self.ax[0].set_ylim(-1.5, 1.5)
        self.ax[0].set_zlim(-1.0, 1.0)

        self.ax.append(plt.subplot(222))
        self.ax[1].set_title('Bird Eyes View Heatmap (YX)', fontsize=self.fontsize)
        self.ax[1].set_xlabel('Y-metre', fontsize=self.fontsize)
        self.ax[1].set_ylabel('X-metre', fontsize=self.fontsize)

        self.ax.append(plt.subplot(223))
        self.ax[2].set_title('Front View Heatmap (YZ)')
        self.ax[2].set_xlabel('Y-metre', fontsize=self.fontsize)
        self.ax[2].set_ylabel('Z-metre', fontsize=self.fontsize)

        self.ax.append(plt.subplot(224))
        self.ax[3].set_title('Side View Heatmap (XZ)')
        self.ax[3].set_xlabel('X-meter', fontsize=self.fontsize)
        self.ax[3].set_ylabel('Z-meter', fontsize=self.fontsize)
    
    def on_key(self, event):
        if event.key == " ":
            self.paused[0] = not self.paused[0]
        
            if (self.paused[0]):
                self.ani.pause()
            else:
                self.ani.resume()

    def load_frame(self, frame):
        # Load frame gt
        frame_gt = None
        if (self.gt_json is not None):
            frame_gt = np.asarray(self.gt_json[frame]["pose"])
        else:
            frame = frame.replace(".npy","")

        # Load frame data
        hm_path = osp.join(self.dir, rtpose_hm_path_tail, frame+".npy")
        hm = np.load(hm_path)[self.draw_doppler, :, :, :].astype(np.float32) # ZYX
        hm = 2**hm

        # Parsing into views
        xy_view = np.transpose(np.mean(hm, axis=0), (1, 0))
        xy_view[xy_view < 1] = 1
        xy_view = np.log2(xy_view)
        # xy_view[xy_view < 1.] = 1.

        # Y-Z view
        zy_view = np.mean(hm,axis=2)
        zy_view[zy_view < 1] = 1
        zy_view = np.log2(zy_view)

        # Z-X view
        zx_view = np.mean(hm,axis=1)
        zx_view[zx_view < 1] = 1
        zx_view = np.log2(zx_view)

        return frame_gt, xy_view, zy_view, zx_view

    def make_update(self):
        def update(frame):
            if self.paused[0]:
                return []
            
            st = time.time()
            frame_gts, xy_view, zy_view, zx_view = self.load_frame(frame)
            if (self.first_frame):
                self.first_frame = False

                # Draw gt
                if (frame_gts is not None):
                    for frame_gt in frame_gts:
                        for id in line_idx:
                            self.ax[0].plot([frame_gt[id[0],0], frame_gt[id[1],0]],
                                        [frame_gt[id[0],1], frame_gt[id[1],1]],
                                        [frame_gt[id[0],2], frame_gt[id[1],2]], c='blue')

                # Draw heatmap
                # X-Y view
                self.plt_holders.append(self.ax[1].imshow(xy_view[::-1, ::-1], 
                                                          cmap='jet',vmin=0,vmax=20))
                self.ax[1].set_aspect(0.25)
                x_plot_loc = np.linspace(0, xy_view.shape[1], 9).astype(int)
                y_plot_loc = np.linspace(0, xy_view.shape[0], 11).astype(int)
                self.ax[1].set_xticks(x_plot_loc, -(x_plot_loc/xy_view.shape[1] * (y_max - y_min) + y_min).astype(int))
                self.ax[1].set_yticks(y_plot_loc, (-y_plot_loc/xy_view.shape[0] * (x_max - x_min) + x_max).astype(int))
                
                # ZY view
                self.plt_holders.append(self.ax[2].imshow(zy_view[::-1, :], 
                                                          cmap='jet',vmin=0,vmax=20))
                self.ax[2].set_aspect(2)
                x_plot_loc = np.linspace(0, zy_view.shape[1], 9).astype(int)
                y_plot_loc = np.linspace(0, zy_view.shape[0], 5).astype(int)
                self.ax[2].set_xticks(x_plot_loc, -( x_plot_loc/zy_view.shape[1] * (y_max - y_min) + y_min).astype(int))
                self.ax[2].set_yticks(y_plot_loc, (-y_plot_loc/zy_view.shape[0] * (z_max - z_min) + z_max).astype(int))
                
                # ZX view
                self.plt_holders.append(self.ax[3].imshow(zx_view[::-1, :], 
                                                          cmap='jet',vmin=0,vmax=20))
                self.ax[3].set_aspect(8)
                x_plot_loc = np.linspace(0, zx_view.shape[1], 12).astype(int)
                y_plot_loc = np.linspace(0, zx_view.shape[0], 5).astype(int)
                self.ax[3].set_xticks(x_plot_loc, ( x_plot_loc/zx_view.shape[1] * (x_max - x_min) + x_min).astype(int))
                self.ax[3].set_yticks(y_plot_loc, (-y_plot_loc/zx_view.shape[0] * (z_max - z_min) + z_max).astype(int))
            else:
                # for id in line_idx:
                #     self.plt_holders[0].set_xdata([frame_gt[id[0],0], frame_gt[id[1],0]])
                #     self.plt_holders[0].set_ydata([frame_gt[id[0],1], frame_gt[id[1],1]])
                #     self.plt_holders[0].set_3d_properties([frame_gt[id[0],2], frame_gt[id[1],2]])

                # Draw gt
                if (frame_gts is not None):
                    self.ax[0].cla()
                    for frame_gt in frame_gts:
                        for id in line_idx:
                            self.ax[0].plot([frame_gt[id[0],0], frame_gt[id[1],0]],
                                        [frame_gt[id[0],1], frame_gt[id[1],1]],
                                        [frame_gt[id[0],2], frame_gt[id[1],2]], c='blue')

                self.plt_holders[0].set_data(xy_view[::-1, ::-1])
                self.plt_holders[1].set_data(zy_view[::-1, :])
                self.plt_holders[2].set_data(zx_view[::-1, :])

            self.ax[0].set_xlabel('X-metre')
            self.ax[0].set_ylabel('Y-metre')
            self.ax[0].set_zlabel('Z-metre')
            self.ax[0].set_xlim(0, 10)
            self.ax[0].set_ylim(-3, 3)
            self.ax[0].set_zlim(-2.5, 2.5)

            plt.pause(0.001)
            return []
        return update


    def animation(self, frame_list):
        self.ani = FuncAnimation(self.fig, self.make_update(), frames=frame_list, 
                                 interval=1, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


def main():
    parser = argparse.ArgumentParser(
        prog="visualize_rtpose.py",
        description="Visualize RT-Pose gt for verification"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to data dir"
    )

    parser.add_argument(
        "-st", "--start_frame",
        type=int,
        default=0,
        help="Skip to frame st"
    )

    parser.add_argument(
        "-t","--type",
        type=str,
        default="train"
    )

    parser.add_argument(
        "--no_gt",
        action="store_true",
        help="Show only data no need for gt"
    )

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="rt",
        help="Visualize mode. s/single, rt/real-time"
    )

    args = parser.parse_args()
    gt_json = None
    doppler = 0

    if (not args.no_gt):
        rtpose_gt_path_tail = "gt/train.json"
        if (args.type == "test"):
            rtpose_gt_path_tail = "gt/test.json"
        if (args.type == "val"):
            rtpose_gt_path_tail = "gt/val.json"

        with open(osp.join(args.dir, rtpose_gt_path_tail), "r") as f:
            gt_json = json.load(f)

        # Remove seq name
        gt_json = gt_json[list(gt_json.keys())[0]]

    # Visualizer
    rt_viz = RTVisualizer(gt_json, args.dir, doppler)

    # Animation
    if (args.no_gt):
        frame_list = sorted(os.listdir(osp.join(args.dir, rtpose_hm_path_tail)))
    else:
        frame_list = list(gt_json.keys())

    if (args.mode == "s"):
        frame_gts, xy_view, zy_view, zx_view = rt_viz.load_frame(frame_list[args.start_frame])

        # Draw gt
        if (frame_gts is not None):
            for frame_gt in frame_gts:
                for c, id in enumerate(line_idx):
                    label = ""
                    if (c == 0):
                        label = "GT"
                    rt_viz.ax[0].plot([frame_gt[id[0],0], frame_gt[id[1],0]],
                                [frame_gt[id[0],1], frame_gt[id[1],1]],
                                [frame_gt[id[0],2], frame_gt[id[1],2]], c='blue')#, label=label)
        rt_viz.ax[0].view_init(elev=25, azim=165)
        rt_viz.ax[0].dist = rt_viz.view_dist
        rt_viz.ax[0].tick_params('x', labelsize=rt_viz.fontsize)
        rt_viz.ax[0].tick_params('y', labelsize=rt_viz.fontsize)
        rt_viz.ax[0].tick_params('z', labelsize=rt_viz.fontsize)
        rt_viz.ax[0].set_title("GT", fontsize=15)

        rt_viz.ax[1].imshow(xy_view[::-1, ::-1].T, cmap='jet',vmin=0,vmax=20)
        rt_viz.ax[1].set_aspect(3.0)
        x_plot_loc = np.linspace(0, xy_view.shape[0], 11).astype(int)
        y_plot_loc = np.linspace(0, xy_view.shape[1], 9).astype(int)
        # print(x_plot_loc, (x_plot_loc/xy_view.shape[0] * (x_max - x_min) + x_min))
        rt_viz.ax[1].set_xticks(x_plot_loc[::-1], np.round(x_plot_loc/xy_view.shape[0] * (x_max - x_min) + x_min).astype(int))
        rt_viz.ax[1].set_yticks(y_plot_loc[::-1], (-y_plot_loc/xy_view.shape[1] * (y_max - y_min) + y_max).astype(int))
        rt_viz.ax[1].tick_params('x', labelsize=rt_viz.fontsize)
        rt_viz.ax[1].tick_params('y', labelsize=rt_viz.fontsize)
        rt_viz.ax[1].set_xlabel('X-metre', fontsize=rt_viz.fontsize)
        rt_viz.ax[1].set_ylabel('Y-metre', fontsize=rt_viz.fontsize)

        # ZY view
        rt_viz.ax[2].imshow(zy_view[::-1, :], cmap='jet',vmin=0,vmax=20)
        rt_viz.ax[2].set_aspect(2)
        x_plot_loc = np.linspace(0, zy_view.shape[1], 9).astype(int)
        y_plot_loc = np.linspace(0, zy_view.shape[0], 5).astype(int)
        rt_viz.ax[2].set_xticks(x_plot_loc, -( x_plot_loc/zy_view.shape[1] * (y_max - y_min) + y_min).astype(int))
        rt_viz.ax[2].set_yticks(y_plot_loc, (-y_plot_loc/zy_view.shape[0] * (z_max - z_min) + z_max).astype(int))
        rt_viz.ax[2].tick_params('x', labelsize=rt_viz.fontsize)
        rt_viz.ax[2].tick_params('y', labelsize=rt_viz.fontsize)

        # ZX view
        rt_viz.ax[3].imshow(zx_view[::-1, :], cmap='jet',vmin=0,vmax=20)
        rt_viz.ax[3].set_aspect(8)
        # print(zx_view.shape[0])
        x_plot_loc = np.linspace(0, zx_view.shape[1], 14).astype(int)
        y_plot_loc = np.linspace(0, zx_view.shape[0], 5)#.astype(int)
        rt_viz.ax[3].set_xticks(x_plot_loc, ( x_plot_loc/zx_view.shape[1] * (x_max - x_min) + x_min).astype(int))
        rt_viz.ax[3].set_yticks(y_plot_loc, (-y_plot_loc/zx_view.shape[0] * (z_max - z_min) + z_max).astype(int))
        rt_viz.ax[3].tick_params('x', labelsize=rt_viz.fontsize)
        rt_viz.ax[3].tick_params('y', labelsize=rt_viz.fontsize)
        # plt.margins(0.2)
    else:
        rt_viz.animation(frame_list[args.start_frame:])
    plt.show()

if __name__ == "__main__":
    main()