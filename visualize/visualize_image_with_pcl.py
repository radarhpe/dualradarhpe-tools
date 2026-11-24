import numpy as np
import os
import os.path as osp
import argparse
import json
from pypcd4 import PointCloud
import sys
import cv2
import matplotlib.pyplot as plt

# Get the parent directory of the current file
parent_dir = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, T_lidar_cam, T_cam_ti, \
                                         create_transformation_matrix, \
                                         src2dst_transformation, left_cam_info

# openpose order: 0-Head, 1-Neck, 2-left shoulder, 3-left elbow, 4-left wrist, 
#                 5-right shoulder, 6-right elbow, 7-right wrist, 8-pelvic,
#                 9-left waist, 10-left knee, 11-left ankle,
#                 12-right waist, 13-right knee, 14-right ankle

# j_select = ["WaistLFront", "WaistRFront", "WaistLBack", "WaistRBack", "BackTop",
#             "HeadTop", "LShoulderTop", "LElbowOut", "LWristOut", "RShoulderTop","RElbowOut", 
#             "RWristOut", "LKneeOut", "LAnkleOut", "RKneeOut", "RAnkleOut"]

line_idx = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[1,4],
                     [1,9],[4,12],[0,7],[9,12],[8,7],
                     [7,9],[9,10],[10,11],[7,12],[12,13],[13,14]])

joint_ind = [8, 12, 13, 14, 9, 10, 11, 1, 0, 5, 6, 7, 2, 3, 4]

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

def opti2sensor_gt(T_opti_box, T_box_sensor, points):
    points = np.hstack((points, np.ones((points.shape[0], 1)))).T
    
    # (4,4) @ (4,4) @ (4, N) => (4, N)
    return np.linalg.inv(T_box_sensor) @ np.linalg.inv(T_opti_box) @ points

def load_extrinsic(path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_3D_from_2D(points_img, lidar_2d, lidar_3d):
    points_3d = np.zeros((15, 3))

    # print(points_img)
    # print(lidar_2d.shape)
    
    # Get body center depth
    center_dist = np.linalg.norm(lidar_2d - points_img[0], axis=1)
    sort_ind = np.argsort(center_dist)
    points_3d[0] = np.mean(lidar_3d[sort_ind[:3]], axis=0)

    # filter based on center depth
    dist_to_center = np.linalg.norm(lidar_3d - points_3d[0], axis=1)
    in_range_ind = np.argwhere(dist_to_center <= 1.0)
    lidar_2d = lidar_2d[in_range_ind].squeeze(1)
    lidar_3d = lidar_3d[in_range_ind].squeeze(1)
    # print(lidar_2d.shape)
    # print(lidar_3d.shape)
    
    # Get distance from image points and lidar 2d
    for i, point_img in enumerate(points_img[1:]):
        # point_img_ = np.array([point_img[1], point_img[0]])
        dist_arr = np.linalg.norm(lidar_2d - point_img, axis=1)
        min_arg = np.argmin(dist_arr)
        points_3d[i+1] = lidar_3d[min_arg]

    return points_3d

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_gt_on_image.py",
        description="For visualizing gt on image"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to sequence root directory"
    )

    parser.add_argument(
        "-p", "--pcl",
        type=str,
        help="Path to point cloud dir from root. Can be radar pcl, lidar or gt"
    )

    parser.add_argument(
        "-s", "--start_frame",
        type=int,
        default=0,
        help="Starting frame"
    )

    parser.add_argument(
        "-m", "--frame_by_frame_mode",
        action="store_true",
        help="Load frame by frame"
    )

    args = parser.parse_args()

    # Get mapping file
    pcl_type = args.pcl[args.pcl.rfind("/")+1:]
    mapping_file = osp.join(args.dir, "calib", f"images2{pcl_type}_mapping.txt")
    ts_map = np.loadtxt(mapping_file, dtype=str)

    fig = plt.figure()
    fig.tight_layout()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, 10)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-2, 2)
    ax.view_init(elev=45, azim=-165)


    for i, img2gt in enumerate(ts_map):
        if (i < args.start_frame):
            continue
        if (img2gt[1] == "nan"):
            continue
        # print("Timestamp: ", img2gt[0], "-", gt_csv[gt_idx[img2gt[1]]].split(",")[1])
        print("Timestamp: ", img2gt[0], img2gt[1], abs(float(img2gt[0]) - float(img2gt[1])))
        # try:

        # Load images
        img = cv2.imread(osp.join(args.dir, "images", img2gt[0]+".png"))

        # Load pcl
        pcd = PointCloud.from_path(osp.join(args.dir, args.pcl, img2gt[1]+".pcd"))
        points_world = pcd.numpy(("x", "y", "z"))
        points_world = points_world[~np.isnan(points_world).any(axis=1)]
        points_cam = src2dst_transformation(np.linalg.inv(T_lidar_cam), points_world)

        x_criter = np.logical_and((points_cam[:, 0] <= 9.0), (points_cam[:, 0] >= 1.0))
        y_criter = np.logical_and((points_cam[:, 1] <= 5.0), (points_cam[:, 1] >= -5.0))
        z_criter = np.logical_and((points_cam[:, 2] <= 1.75), (points_cam[:, 2] >= -2.0))
        pnt_indices = np.logical_and(x_criter, np.logical_and(y_criter, z_criter))
                      
        points_cam = points_cam[pnt_indices]
        points_world = points_world[pnt_indices]

        # print(points_cam[0])

        points_img = point2pixel(points_cam[:,:3], left_cam_info).squeeze()
        # print(points_img)
        points_img[:,1] = img.shape[0] - points_img[:,1]

        # Load openpose estimation
        hpe2d_json = None
        with open(osp.join(args.dir, "gt/json", f"{img2gt[0]}_keypoints.json")) as f:
            hpe2d_json = json.load(f)["people"]
        
        # # Show lidar
        # for point_2d in points_img:
        #     cv2.circle(img, point_2d.astype(int), 3, (0, 0, 255), -1)

        ax.collections.clear()
        ax.lines.clear()

        # Show openpose prediction
        point_3d_hpe = []
        for person in hpe2d_json:
            points = np.asarray(person["pose_keypoints_2d"]).reshape(-1,3)[joint_ind]
            avg_score = np.mean(points[:,2])
            zero_point = (points == np.array([0.0,0.0,0.0])).any()
            if (avg_score > 0.5 and not zero_point):
                for i, point in enumerate(points):
                    point = point[:2].astype(int)
                    cv2.circle(img, point, 3, (0, 0, 255), -1)
                    cv2.putText(img, str(i), point, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

                point_3d_hpe = get_3D_from_2D(points[:,:2], points_img, points_cam)

                # Transform to TI radar coord
                
                point_3d_hpe = src2dst_transformation(np.linalg.inv(T_cam_ti), point_3d_hpe)

                # print(point_3d_hpe.shape)
                # ax.scatter(point_3d_hpe[:,0],point_3d_hpe[:,1], point_3d_hpe[:,2])
                for id in line_idx:
                    ax.plot([point_3d_hpe[id[0],0], point_3d_hpe[id[1],0]],
                            [point_3d_hpe[id[0],1], point_3d_hpe[id[1],1]],
                            [point_3d_hpe[id[0],2], point_3d_hpe[id[1],2]], c='blue')


        # cv2.imshow("Test", img)
        plt.draw()
        plt.pause(0.0001)
        if (args.frame_by_frame_mode):
            key = cv2.waitKey(0) 
        else:
            key = cv2.waitKey(30)
        if (key == ord("q")):
            break
        # except:
        #     print("Point cloud missing")
    

if __name__ == "__main__":
    main()

