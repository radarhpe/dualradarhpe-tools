import numpy as np
import os
import os.path as osp
import argparse
import time
import json
import matplotlib.pyplot as plt
import sys
import os
from pcl_visualizer import PCLVisualizer, line_idx

# # Get the parent directory of the current file
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add it to the sys.path
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)
# from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, T_lidar_bosch, T_calib_radar

def create_gt_ts2id(gt_csv):
    gt_ts2id = {}
    for line in gt_csv:
        line = line.split(",")
        gt_ts2id[line[1]] = int(line[0])
    return gt_ts2id

def load_gt(gt_csv, frame_idx):
    gt_line = gt_csv[frame_idx].split(",")
    frame_id, gt_ts, gt = gt_line[0], gt_line[1], np.array(gt_line[2:-8], dtype=np.float32)
    box_T = np.array(gt_line[-8:-1], dtype=np.float32)
    box_rot, box_pos = box_T[:4], box_T[4:]
    return frame_id, gt_ts, gt, box_rot, box_pos

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
        "-p", "--pcl",
        type=str,
        default="radar/radarpcl",
        help="Path suffix to pcl dir from root dir",
    )

    parser.add_argument(
        "-s", "--start_frame",
        type=int,
        default=0,
        help="Starting frame"
    )

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="rt",
        help="Mode: rt-realtime, fbf-frame by frame, s-single frame at start frame"
    )

    args = parser.parse_args()

    # Get gt and mapping path
    seq = osp.basename(osp.normpath(args.dir))
    pcl_type = args.pcl[args.pcl.rfind("/")+1:]
    if (pcl_type == "radarpcl"):
        gt_file_path = osp.join(args.dir, f"gt/{seq}_filtered.csv")
    else:
        gt_file_path = osp.join(args.dir, f"gt/{seq}_filtered_bosch.csv")
    # ts_map_path = osp.join(args.dir, f"calib/{pcl_type}2gt_mapping.txt")

    # Check if gt is exist
    if (not osp.exists(gt_file_path)):
        print("Ground Truth is not exist")
        exit(0)

    # Load GT csv
    gt_csv = None
    with open(gt_file_path, "r") as f:
        gt_csv = f.read().splitlines()[1:]
    # gt_ts2id = create_gt_ts2id(gt_csv)

    # Load ts mapping file
    # ts_map = np.loadtxt(ts_map_path, dtype=str)

    # Get mapping data if visualize two pcl sets
    n_sets = 2
   
    viz = PCLVisualizer(n_sets, True, T_sen1_sen2=np.eye(4), gt_csv=gt_csv)
   
    # forming frame list
    frame_list = []
    for i, gt_line in enumerate(gt_csv):
        gt_line = gt_line.split(",")
        src2dst = [gt_line[1], gt_line[3]]
        if (src2dst[0] == "nan" or src2dst[1] == "nan"):
            continue

        file_01 = src2dst[0] + ".xyzdi"
        file_01 = os.path.join(args.dir, args.pcl, file_01)

        frame_list.append([i, file_01, pcl_type, "", "gt"])

    # Start the animation
    if (args.mode == "s"):
        points_a, points_b = viz.load_data(frame_list[args.start_frame])

        # Clear previous scatter plots and re-plot
        viz.ax.set_title(f"Frame {frame_list[args.start_frame][0]}")
        viz.ax.legend()
        viz.ax.scatter(points_a[:,0], points_a[:,1], points_a[:,2], 
                     c='blue', label='Set A', s=1.0, picker=True)
        if (points_b is not None):
            if (gt_csv is None):
                viz.ax.scatter(points_b[:,0], points_b[:,1], points_b[:,2], 
                            c='red', label='Set B', s=1.0, picker=True)
                viz.fig.canvas.mpl_connect('button_press_event', 
                    lambda event: viz.on_click(event, points_a[:,:3], points_b[:,:3], viz.ax))
            else:
                for gt in points_b:
                    if (np.sum(np.isnan(gt))):
                        continue
                    for c, id in enumerate(line_idx):
                        scat = viz.ax.plot([gt[id[0],0], gt[id[1],0]],
                                [gt[id[0],1], gt[id[1],1]],
                                [gt[id[0],2], gt[id[1],2]], c='blue')
        else:
            viz.fig.canvas.mpl_connect('button_press_event', 
                lambda event: viz.on_click(event, points_a[:,:3], None, viz.ax))
        
            
    else: 
        viz.animation(frame_list)

    plt.show()
    for pa in list(viz.picked_points["a"].values()): 
        print(f"[({pa[0]:0.2f}, {pa[1]:0.2f}, {pa[2]:0.2f}), ", end="")
    for pb in list(viz.picked_points["b"].values()):
        print(f"({pb[0]:0.2f}, {pb[1]:0.2f}, {pb[2]:0.2f})],")

if __name__ == "__main__":
    main()
