import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path as osp
import argparse
import time
import json
from sklearn.cluster import DBSCAN

radpcl_path_tail = "radarpcl_multiper"

j_select = ["WaistLFront", "WaistRFront", "WaistLBack", "WaistRBack", "BackTop",
            "HeadTop", "LShoulderTop", "LElbowOut", "LWristOut", "RShoulderTop","RElbowOut", 
            "RWristOut", "LKneeOut", "LAnkleOut", "RKneeOut", "RAnkleOut"]

line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_pointcloud.py",
        description="For calculating centroid of reflector on lidar"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to data dir",
    )

    parser.add_argument(
        "--viz_raw",
        action="store_true"
    )

    parser.add_argument(
        "-s", "--start_frame",
        type=int,
        default=0,
        help="Starting frame"
    )

    parser.add_argument(
        "-t", "--type",
        type=str,
        default="outdoor"
    )

    args = parser.parse_args()

    # seq name
    seq = args.dir[args.dir.rfind("/")+1:]
    ft_path_tail = f"radarpcl_multiper/featuremap_{seq}_{args.type}.npy"
    lb_path_tail = f"radarpcl_multiper/labels_{seq}_{args.type}.npy"
    # rd_path_tail = f"radarpcl_multiper/labels_{seq}_{args.type}.npy"
    if (args.type != "outdoor"):
        gt_path_tail = f"radarpcl_multiper/{seq}_{args.type}.json"
    else:
        gt_path_tail = f"gt/{seq}.json"

    fts = np.load(osp.join(args.dir, ft_path_tail))
    fts = fts.reshape(-1, fts.shape[1]**2, 5)
    lbs = np.load(osp.join(args.dir, lb_path_tail)).reshape(-1, 15, 3)
    with open(osp.join(args.dir, gt_path_tail), "r") as f:
        gt_json = json.load(f)        

    # 
    fig = plt.figure(figsize=(12,10))
    fig.tight_layout()

    ax11 = fig.add_subplot(111, projection="3d")
    ax11.view_init(elev=10, azim=-165)
    ax11.set_xlabel('X-metre')
    ax11.set_ylabel('Y-metre')
    ax11.set_zlabel('Z-metre')

    ax11.set_xlim(0, 8)
    ax11.set_ylim(-3, 3)
    ax11.set_zlim(-2, 2)

    line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])
    first_frame = True
    # db_scan = DBSCAN(eps=0.5, min_samples=40)
    frame_list = sorted(list(gt_json.keys()))
    for k, (frame_ts, frame) in enumerate(gt_json.items()):
        print(frame_ts)
        frame_ind = frame["frame_id"]
        gts = np.asarray(frame["poses"])
        ax11.collections.clear()
        ax11.lines.clear()
        for i, ind in enumerate(frame_ind):
            # print(144*5, np.sum(cloud != np.array([0.,0.,0.,0.,0.])))
            # cloud[cloud[:,4] < 0.25] = 0.0

            gt = gts[i]

            if (not args.viz_raw):
                cloud = fts[ind]
                cloud[:,:3] = cloud[:,:3] + gt[0]
                lb = lbs[ind] + gt[0]
                # print(cloud[:,4])
                sc = ax11.scatter(cloud[:,0], cloud[:,1], cloud[:,2], c = cloud[:,4], s=5, vmin=-1.0, vmax=1.0)
            else:
                lb = gt

            for id in line_idx:
                ax11.plot([lb[id[0],0], lb[id[1],0]],
                        [lb[id[0],1], lb[id[1],1]],
                        [lb[id[0],2], lb[id[1],2]], c='blue')
            

        if (args.viz_raw):
            for j in range(1):
                frame_ts = frame_list[k+j]
                print(frame_ts)
                cloud = np.loadtxt(osp.join(args.dir, "radar/radarpcl", f"{frame_ts}.xyzdi"), 
                                   dtype=np.float32)
                cloud[:,(0,1)] = cloud[:,(1,0)]
                sc = ax11.scatter(cloud[:,0], cloud[:,1], cloud[:,2], c = cloud[:,3], s=5, vmin=-1.0, vmax=1.0)

        if (first_frame):
            plt.colorbar(sc, ax=ax11)
            first_frame = False
        
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    main()
