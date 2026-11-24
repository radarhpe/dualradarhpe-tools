import numpy as np
import os
import os.path as osp
import argparse
import time
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import sys
import os
from pcl_visualizer import PCLVisualizer

line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_pointcloud.py",
        description="For showing pointcloud with Matplotlib"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        default="",
        help="Path to sequence root directory"
    )

    parser.add_argument(
        "-s", "--start_frame",
        type=int,
        default=0,
        help="Starting frame"
    )

    # parser.add_argument(
    #     "-m", "--mode",
    #     type=str,
    #     default="rt",
    #     help="Mode: rt-realtime, fbf-frame by frame, s-single frame at start frame"
    # )

    args = parser.parse_args()
    seq = args.dir[args.dir.rfind("/")+1:]

    # Get mapping data if visualize two pcl sets and gt
    ti2gt_map = np.loadtxt(osp.join(args.dir, "calib/radarpcl2gt_mapping.txt"), dtype=str)
    ti2bo_map = np.loadtxt(osp.join(args.dir, "calib/radarpcl2bosch_mapping.txt"), dtype=str)
    ti2bo2gt_map = np.hstack((ti2bo_map,ti2gt_map[:,1:2]))

    # Load gtcsv
    gt_csv = None
    with open(osp.join(args.dir, f"gt/{seq}_filtered.csv"), "r") as f:
        gt_csv = f.read().split("\n")[1:-1] 

    # Load frame
    tipcl = np.loadtxt(osp.join(args.dir, f"radar/radarpcl/{ti2bo2gt_map[args.start_frame][0]}.xyzdi"),dtype=np.float32)
    tipcl[:,(0,1)] = tipcl[:,(1,0)]
    bopcl = np.loadtxt(osp.join(args.dir, f"bosch/{ti2bo2gt_map[args.start_frame][1]}.xyzdi"),dtype=np.float32)
    gt    = np.array(gt_csv[args.start_frame].split(",")[4:-8], dtype=np.float32)
    gt    = np.reshape(gt, (-1, 3))

    # Clear previous scatter plots and re-plot
    fig = plt.figure()
    fig.tight_layout()
    ax = []
    ax.append(plt.subplot(131, projection="3d"))
    ax.append(plt.subplot(132, projection="3d"))
    ax.append(plt.subplot(133, projection="3d"))
    view_dist = 9
    elevation = 25
    azimuth = 165
    font_size = 20
    for i in range(len(ax)):
        ax[i].view_init(elev=elevation, azim=azimuth)
        ax[i].dist = view_dist
        # ax[i].set_xlabel('X-metre', fontsize=font_size)
        # ax[i].set_ylabel('Y-metre', fontsize=font_size)
        # ax[i].set_zlabel('Z-metre', fontsize=font_size)
        ax[i].set_xlim(2, 7.0)
        ax[i].set_ylim(-1.5, 1.5)
        ax[i].set_zlim(-1.0, 1.0)
        ax[i].tick_params('x', labelsize=font_size)
        ax[i].tick_params('y', labelsize=font_size)
        ax[i].tick_params('z', labelsize=font_size)

    ax[0].set_title("TI point cloud", fontsize=font_size)
    ax[0].scatter(tipcl[:,0], tipcl[:,1], tipcl[:,2], s=5, c='blue')
    
    ax[1].set_title("Bosch point cloud", fontsize=font_size)
    ax[1].scatter(bopcl[:,0], bopcl[:,1], bopcl[:,2], s=5, c='red')
    
    ax[2].set_title("Ground truth", fontsize=font_size)
    for id in line_idx:
        ax[2].plot([gt[id[0],0], gt[id[1],0]],
                [gt[id[0],1], gt[id[1],1]],
                [gt[id[0],2], gt[id[1],2]], c='blue')

    plt.show()

if __name__ == "__main__":
    main()
