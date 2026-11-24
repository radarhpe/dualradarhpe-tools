# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 14:59:44 2025

@author: trung.l.nguyen@tuni.fi
"""

"""
- Visualize ground truth and prediction as skeleton 
  with the synced camera view and radar pcl
"""
import numpy as np 
from sklearn import metrics
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import os.path as osp
import argparse
import time
import json
import sys
import cv2

# Get the parent directory of the current file
parent_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, T_lidar_cam, T_cam_ti, \
                                         create_transformation_matrix, \
                                         src2dst_transformation, left_cam_info

image_path_tail = "images"
rad2img_calib_path_tail = "calib/radarpcl2images_mapping.txt"
rad2lid_calib_path_tail = "calib/radarpcl2lidar_mapping.txt"
rad_mars_path_tail = "radarpcl_mars"
lidar_path_tail = "lidar"
IMG_SIZE = (576, 1024)

def point2pixel(point, cam_info):
    # Define the 3D points (N, 3)
    points_3d = np.array(point, np.float32) 

    # Transform point from custom coordinator to original
    T_org_custom = np.array([[0., -1., 0., 0.],
                             [0., 0., -1., 0.],
                             [1., 0., 0., 0.],
                             [0., 0., 0., 1.]])
    # T_org_custom = np.eye(4, 4)
    
    # (4, N)
    points_3d = T_org_custom @ np.hstack((points_3d, np.ones((points_3d.shape[0],1)))).T
    
    # (N, 3)
    points_3d = points_3d.T[:,:3]
    
    # Define the rotation and translation vectors, Extrinsic to world coordination
    rvec = np.zeros((3, 1), np.float32) 
    tvec = np.zeros((3, 1), np.float32)
    
    # Map the 3D point to 2D point
    points_2d, _ = cv2.projectPoints(points_3d, 
                                    rvec, tvec, 
                                    np.array(cam_info["k"]).reshape(3, 3), 
                                    None)
    return points_2d

def create_argparse():
    parser = argparse.ArgumentParser(
        prog="visualize_hpe3d_result.py",
        description="Visualize 3D human pose estimation result"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to model's estimation",
    )
    
    parser.add_argument(
        "--data",
        default="",
        type=str,
        help="Path to data"
    )
    
    parser.add_argument(
        "-gt",
        default="",
        type=str,
        help="Path to gt's estimation",
    )

    parser.add_argument(
        "--use_lidar",
        action="store_true",
        help="Decide to show lidar instead of skeleton"
    )

    parser.add_argument(
        "--savefig",
        action="store_true",
        help="Decide to save figure"
    )

    parser.add_argument(
        "-st", "--start_frame",
        type=int,
        default=0
    )

    parser.add_argument(
        "--use_motion",
        action="store_true"
    )

    parser.add_argument(
        "--multi_person",
        action="store_true"
    )

    parser.add_argument(
        "-fbf", "--frame_by_frame",
        action="store_true"
    )

    return parser

def main():
    parser = create_argparse()
    args = parser.parse_args()

    # Get radarpcl and gt
    motion_type = "with_motion" if args.use_motion else "no_motion"
    suffix = "_multi_person" if args.multi_person else ""
    
    if (args.data == ""):
        radpcls = np.load(f"../evaluation/groundtruth/{motion_type}/featuremap_test{suffix}.npy")
    else:
        radpcls = np.load(args.data)
    
    if (args.gt == ""):
        gts = np.load(f"../evaluation/groundtruth/{motion_type}/labels_test{suffix}.npy")
    else:
        gts = np.load(args.gt)
    preds = np.load(args.file)
    print(radpcls.shape, gts.shape, preds.shape)
    if (radpcls.shape[0] != preds.shape[0]):
        print("Not the same data")
        exit(0)

    # Get radarpcl and images sync
    radpcl_img_ts = None
    if (args.data != ""):
        data_folder = args.data[:args.data.rfind("/")]
        radpcl_img_ts = np.loadtxt(osp.join(data_folder, "../../calib/radarpcl2images_mapping.txt"), dtype=str)

    # Joints line
    n_joints = 15
    line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

    # Plotting operator
    fig = plt.figure()
    fig.tight_layout()
    ax_im = fig.add_subplot(121)
    ax_im.set_title("ZED2i, left view", fontsize=20)
    ax_im.get_xaxis().set_ticks([])
    ax_im.get_yaxis().set_ticks([])

    ax = fig.add_subplot(122, projection="3d")
    ax.set_title("TI AWR2243 cascaded radar, point cloud", 
                 fontsize=20, y=1.0, pad=-70.)
    ax.set_xlim(3, 6)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1.5, 1.5)
    ax.legend(loc=(1, -70))
    ax.dist = 1
    ax.view_init(elev=25, azim=165)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()

    plt.subplots_adjust(wspace=0.0)

    show_img = None
    circ = []
    for i, (radpcl, gt, pred) in enumerate(zip(radpcls, gts, preds)):
        # if (i < 1647):
            # continue
        if (i < args.start_frame):
            continue
        print(i)

        pcl = radpcl.reshape(-1, 5)
        gt = gt.reshape(-1, 3)
        pred = pred.reshape(-1, 3)

        # # Draw radar pcl
        # ax11.cla()
        # ax11.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c=pcl[:,4], s=5, cmap="jet")
        
        # # Draw prediction as line
        # for id in line_idx:
        #     ax11.plot([gt[id[0],0], gt[id[1],0]],
        #             [gt[id[0],1], gt[id[1],1]],
        #             [gt[id[0],2], gt[id[1],2]], c='blue')
        # ax11.set_xlabel('X-meter')
        # ax11.set_ylabel('Y-meter')
        # ax11.set_zlabel('Z-meter')

        # ax11.set_xlim(0, 10)
        # ax11.set_ylim(-3, 3)
        # ax11.set_zlim(-2.5, 2.5)

        # Draw radar pcl
        ax.collections.clear()
        ax.lines.clear()

        # Draw prediction as line
        is_labeled = False
        for id in line_idx:
            label=""
            if (not is_labeled):
                label = "GT"
            ax.plot([gt[id[0],0], gt[id[1],0]],
                    [gt[id[0],1], gt[id[1],1]],
                    [gt[id[0],2], gt[id[1],2]], c='blue', label=label)
            if (not is_labeled):
                label = "Pred"
            ax.plot([pred[id[0],0], pred[id[1],0]],
                    [pred[id[0],1], pred[id[1],1]],
                    [pred[id[0],2], pred[id[1],2]], c='red', label=label)
            is_labeled = True
        ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c=pcl[:,4], s=5, cmap="jet")
        ax.legend(loc=(0.75, 0.65))

        # Show RGB view if available
        # face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if (radpcl_img_ts[i,1] != "nan"):
            # Transform gt from radar coord to cam coord
            # points_cam = src2dst_transformation(T_cam_ti, gt)

            # Project 3D to 2D point
            # points_img = point2pixel(points_cam, left_cam_info).squeeze()
            # points_img[:,1] = IMG_SIZE[0] - points_img[:,1] + 25
            # points_img[:,0] = IMG_SIZE[1] - points_img[:,0] - 25

            image = plt.imread(osp.join(data_folder, "../..", image_path_tail, radpcl_img_ts[i,1]+".png"))
            image = np.flip(image, axis=1)
            # print(image)
            # face_data = face_detect.detectMultiScale(np.array(image*255, dtype=np.uint8), 
            #                                          1.3, 5)

            with open(osp.join(data_folder, "../../openpose", radpcl_img_ts[i,1]+"_keypoints.json"),"r") as f:
                skeletons = json.load(f)["people"]
            
            heads = []
            for s, skeleton in enumerate(skeletons):
                org_points = np.asarray(skeleton["pose_keypoints_2d"]).reshape(-1,3)
                head_points = org_points[[0,16,17,18]]
                index = np.where(np.all(head_points[:,:2] != np.array([0.,0.]), axis=1))
                head_points = np.mean(head_points[index], axis=0)
                # if (head_points[-1] >= ):
                heads.append(head_points)
            heads = heads[:2]
            # heads = np.array(heads)
            # heads = heads[np.argsort(heads[:, -1])]
            # heads = np.array([heads[0], ])
            ax_im.add_artist(Circle([IMG_SIZE[1]-847, 259], radius=40))

            # for (x, y, w, h) in face_data:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     roi = image[y:y+h, x:x+w]
            #     # applying a gaussian blur over this new rectangle area
            #     roi = cv2.GaussianBlur(roi, (23, 23), 30)
            #     # impose this blurred image on original image to get final image
            #     image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

            if (show_img is None):
                show_img = ax_im.imshow(image)
                for head in heads:
                    head[0] = IMG_SIZE[1] - head[0]
                    circ.append(Circle(head[:2], radius=30))
                    ax_im.add_artist(circ[-1])
            else:
                show_img.set_data(image)
                for h, head in enumerate(heads):
                    head[0] = IMG_SIZE[1] - head[0]
                    if (h >= len(circ)):
                        circ.append(Circle(head[:2], radius=30))
                        ax_im.add_artist(circ[-1])
                    else:
                        circ[h].center = head[:2]

        # else:
        #     scat.set_array()
        # if (args.savefig):
        #     fig.canvas.draw()
        #     buf = fig.canvas.tostring_rgb()
        #     width, height = fig.canvas.get_width_height()
        #     img = Image.frombytes("RGB", (width, height), buf)
        #     if (args.no_pred):
        #         img.save(osp.join(args.dir, "radpcl_viz",
        #                  "radpcl_viz_"+str(i).zfill(4)+".png"))
        #     else:
        #         img.save(osp.join(args.dir, "hpe_result", 
        #                              "hpe_result_"+str(i).zfill(4)+".png"))
        # else:
        plt.draw()
        if (args.frame_by_frame):
            plt.waitforbuttonpress()
        else:
            plt.pause(1./30)

if __name__ == "__main__":
    main()
