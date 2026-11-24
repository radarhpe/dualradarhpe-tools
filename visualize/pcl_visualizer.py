import numpy as np
import os
import os.path as osp
import argparse
from pypcd4 import PointCloud
import time
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.cluster import DBSCAN

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, \
                                         create_transformation_matrix, \
                                         src2dst_transformation

line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

class PCLVisualizer():
    def __init__(self, n_sets=1, calib=False, T_sen1_sen2=None, grouping=False, gt_csv=None):
        self.paused = [False]
        self.picked_points = {"a": {}, "b": {}}
        self.n_sets = n_sets
        self.T_sen1_sen2 = T_sen1_sen2
        print(self.T_sen1_sen2)
        self.grouping = grouping if self.n_sets == 1 else False
        if (self.grouping):
            self.db_scan = DBSCAN(eps=1.0, min_samples=10)
        self.gt_csv = gt_csv
        self.calib = calib
        if (self.gt_csv is not None):
            self.calib = True

        # Init plot entity
        self.fig = plt.figure(figsize=(12,10))
        self.fig.tight_layout()

        if (self.calib or self.n_sets==1):
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(121, projection="3d")
            self.ax2 = self.fig.add_subplot(122, projection="3d")

        self.elev = 25
        self.azim = 165
        self.view_dist = 9
        self.fontsize = 15

        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.ax.dist = self.view_dist
        self.ax.set_xlabel('X-metre', fontsize=self.fontsize)
        self.ax.set_ylabel('Y-metre', fontsize=self.fontsize)
        self.ax.set_zlabel('Z-metre', fontsize=self.fontsize)
        self.ax.set_xlim(2, 7.0)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_zlim(-1.0, 1.0)
        self.ax.tick_params('x', labelsize=self.fontsize)
        self.ax.tick_params('y', labelsize=self.fontsize)
        self.ax.tick_params('z', labelsize=self.fontsize)


        if (not self.calib and self.n_sets == 2):
            self.ax2.view_init(elev=self.elev, azim=self.azim)
            self.ax2.dist = self.view_dist
            self.ax2.set_xlabel('X-metre', fontsize=self.fontsize)
            self.ax2.set_ylabel('Y-metre', fontsize=self.fontsize)
            self.ax2.set_zlabel('Z-metre', fontsize=self.fontsize)
            self.ax2.set_xlim(2, 7.0)
            self.ax2.set_ylim(-1.5, 1.5)
            self.ax2.set_zlim(-1.0, 1.0)
            self.ax2.tick_params('x', labelsize=self.fontsize)
            self.ax2.tick_params('y', labelsize=self.fontsize)
            self.ax2.tick_params('z', labelsize=self.fontsize)

        self.first_frame = True

        # Initial empty plots

    def load_pcl(self, path, type):
        cloud = None
        if (type == "lidar"):
            pcd = PointCloud.from_path(path)
            cloud = pcd.numpy(("x", "y", "z"))
            cloud = cloud[cloud[:, 0] <= 8.0]
            cloud = cloud[cloud[:, 0] >= 0.0]
            cloud = cloud[cloud[:, 1] <= 6.0]
            cloud = cloud[cloud[:, 1] >= -3.0]
            cloud = cloud[cloud[:, 2] <= 0.0]
            cloud = cloud[cloud[:, 2] >= -2.0]
        elif (type == "radarpcl" or type == "pcl"):
            pcd = np.loadtxt(path, ndmin=2)
            cloud = pcd
            cloud[:, [0, 1]] = cloud[:, [1, 0]]

            # cloud = cloud[cloud[:, 0] <= 8.0]
            # cloud = cloud[cloud[:, 0] >= 1.0]
            # cloud = cloud[cloud[:, 1] <= 4.0]
            # cloud = cloud[cloud[:, 1] >=-4.0]
            # cloud = cloud[cloud[:, 2] <= 2.0]
            # cloud = cloud[cloud[:, 2] >= -2.0]

            # cloud = cloud[:,:3]
        elif (type == "bosch"):
            pcd = np.loadtxt(path, ndmin=2)
            cloud = pcd
            # cloud = cloud[:,:3]

        return cloud
    
    def load_gt(self, index):
        gt_line = self.gt_csv[index].split(",")[4:-8]
        gt = np.array(gt_line, dtype=np.float32).reshape(-1, 15, 3)
        return gt

    def load_data(self, frame):
        idx, file_01, sen1_fmt, file_02, sen2_fmt = frame

        # Load sensor pcl 1
        cloud_01 = self.load_pcl(file_01, sen1_fmt)

        # Load sensor pcl 2
        cloud_02 = None
        if (file_02 != "" or sen2_fmt != ""):
            if (sen2_fmt == "gt"):
                cloud_02 = self.load_gt(idx)
            else:
                cloud_02 = self.load_pcl(file_02, sen2_fmt)
                if (self.calib):
                    cloud_02[:,:3] = src2dst_transformation(self.T_sen1_sen2, cloud_02[:,:3])

        return cloud_01, cloud_02

    def make_update(self):
        def update(frame):
            if self.paused[0]:
                return []
            
            st = time.time()
            points_a, points_b = self.load_data(frame)

            # Clear previous scatter plots and re-plot
            self.ax.collections.clear()
            self.ax.lines.clear()
            if (points_a.shape[1] > 3 and points_b is None):
                if (self.grouping):
                    db = self.db_scan.fit(points_a[:,:3])
                    labels = db.labels_
                    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
                    print("Number of cluster: ", n_labels)
                    for n_label in range(n_labels):
                        indices = (labels == n_label)
                        self.ax.scatter(points_a[indices,0], points_a[indices,1], points_a[indices,2], 
                            label='', s=5)
                else:
                    scat = self.ax.scatter(points_a[:,0], points_a[:,1], points_a[:,2], c=points_a[:,3], 
                                label='', s=5, cmap="jet")
            else:
                scat = self.ax.scatter(points_a[:,0], points_a[:,1], points_a[:,2], c="blue", 
                                label='', s=5, cmap="jet")

            if (points_b is not None):
                if (self.calib or self.n_sets == 1):
                    if (self.gt_csv is None):
                        scat = self.ax.scatter(points_b[:,0], points_b[:,1], points_b[:,2], c='red', 
                                label='', s=5, cmap="jet")
                    else:
                        for gt in points_b:
                            if (np.sum(np.isnan(gt))):
                                continue
                            for id in line_idx:
                                scat = self.ax.plot([gt[id[0],0], gt[id[1],0]],
                                        [gt[id[0],1], gt[id[1],1]],
                                        [gt[id[0],2], gt[id[1],2]], c='blue')

                else:
                    self.ax2.collections.clear()
                    scat = self.ax2.scatter(points_b[:,0], points_b[:,1], points_b[:,2], c='red', 
                                label='', s=5, cmap="jet")
                    # self.ax2.set_title(f"Frame {frame[0]}")
                    self.ax2.set_title(f"Bosch point cloud")
                    # self.ax2.legend()

            # self.ax.set_title(f"Frame {frame[0]}")
            self.ax.set_title(f"TI point cloud")
            self.ax.legend()
            print(frame[0])

            # if (self.first_frame):
            #     self.first_frame = False
            #     self.fig.colorbar(scat, ax=self.ax)

            plt.pause(0.001)
            return []
        return update

    def on_key(self, event):
        if event.key == " ":
            self.paused[0] = not self.paused[0]
        
            if (self.paused[0]):
                self.ani.pause()
            else:
                self.ani.resume()

    def on_click(self, event, points_a, points_b, ax):
        if (event.button != 2):
            return

        # Project 3D points to 2D
        if (points_b is not None):
            points = np.append(points_a, points_b, axis=0)
        else:
            points = points_a
        x_proj, y_proj, _ = proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
        screen_coords = ax.transData.transform(np.vstack([x_proj, y_proj]).T)

        # Find closest point to click
        mouse_xy = np.array([event.x, event.y])
        dists = np.linalg.norm(screen_coords - mouse_xy, axis=1)
        idx = np.argmin(dists)
        selected_point = points[idx]
        set_id = "a" if idx < points_a.shape[0] else "b"

        if (idx in self.picked_points[set_id]):
            self.picked_points[set_id].pop(idx)
            ax.scatter(*selected_point, c='k', s=1.0)
        else:
            self.picked_points[set_id][idx] = selected_point
            ax.scatter(*selected_point, c='k', marker='x', s=50)

        print(selected_point)
    
    def animation(self, frame_list):
        self.ani = FuncAnimation(self.fig, self.make_update(), frames=frame_list, 
                                 interval=1, blit=False)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

