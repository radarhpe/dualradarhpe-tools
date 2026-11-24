import numpy as np
import os
import cv2
import argparse
import json
from scipy.spatial.transform import Rotation
from pypcd4 import PointCloud

left_cam_info = {
    "height": 576,
    "width": 1024,
    "distortion_model": "rational_polynomial",
    "d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "k": [756.9138793945312,               0.0, 534.088134765625,
                        0.0, 756.9138793945312, 288.1670227050781,
                        0.0,               0.0,               1.0],
    "r": [1.0, 0.0, 0.0,
          0.0, 1.0, 0.0,
          0.0, 0.0, 1.0],
    "p": [756.9138793945312,               0.0, 534.088134765625, 0.0,
                        0.0, 756.9138793945312, 288.1670227050781, 0.0, 
                        0.0,               0.0,               1.0, 0.0]
}

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

def create_transformation_matrix(R, t):
    T_mat = np.eye(4, 4)
    R = np.asarray(R)
    t = np.asarray(t)

    R = Rotation.from_quat(R).as_matrix()
    T_mat[:3,:3] = R
    T_mat[:3,3] = t
    return T_mat

def load_extrinsic(path):
    data = None
    with open(path, "r") as f:
        data = json.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(
        prog="bbox2d_generation.py",
        description="Generate 2D human bounding box from optitrack ground truth"
    )

    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        help="Path to image directory"
    )

    parser.add_argument(
        "-gt",
        type=str,
        help="Path to ground truth files"
    )

    parser.add_argument(
        "--mapping_file",
        type=str,
        help="Path to timestamp mapping file"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save bounding box files"
    )

    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
    )


    args = parser.parse_args()
    
    # Get image list
    img_list = sorted(os.listdir(args.data_dir))

    # Get pcl list
    gt_csv = None
    with open(args.gt, "r") as f:
        gt_csv = f.read().split("\n")[1:-1] # Load, split by row and remove 1st and last row
    gt_idx = {str(line.split(",")[1]): int(line.split(",")[0]) for line in gt_csv}

    # T_box_cam = create_transformation_matrix([0.5, 0.5, -0.5, 0.5], \
    #                                          [0.151, -0.781, -0.186])
    # T_box_cam = create_transformation_matrix([0.5, 0.5, -0.5, 0.5], \
    #                                          [0.25, -0.761, -0.155])

    # T_box_cam = create_transformation_matrix([0.5, 0.5, -0.5, 0.5], \
    #                                          [0.275, -0.801, -0.155])
    T_box_cam = create_transformation_matrix([0.5, 0.5, -0.5, 0.5], \
                                             [0.295, -0.8251, -0.0255])


    # Get mapping data
    ts_map = np.loadtxt(args.mapping_file, dtype=str)
    bbox2d_dict = {}

    for i, img2gt in enumerate(ts_map):
        if (img2gt[1] == "nan"):
            continue
        # print("Timestamp: ", img2gt[0], "-", gt_csv[gt_idx[img2gt[1]]].split(",")[1])
        # print("Timestamp: ", img2gt[0], img2gt[1], abs(float(img2gt[0]) - float(img2gt[1])))
        # try:
        img = cv2.imread(os.path.join(args.data_dir, img2gt[0]+".png"))

        gt = gt_csv[gt_idx[img2gt[1]]].split(",")[2:-1]
        if ("" in gt):
            continue
        gt = np.array(gt, np.float32)
        T_opti_box = create_transformation_matrix(gt[-7:-3], gt[-3:])
        gt = gt[:-7]

        #point in world coordinate, reshape to (N, 3)
        points_world = gt.reshape(-1, 3)
        points_cam = opti2sensor_gt(T_opti_box, T_box_cam, points_world).T

        #project point in camera to image and flip it upside down
        points_img = point2pixel(points_cam[:,:3], left_cam_info).squeeze()
        points_img[:,1] = img.shape[0] - points_img[:,1]

        # Get x_top, y_top, x_bot, y_bot
        x_top, y_top = int(min(points_img[:,0])), int(min(points_img[:,1]))
        x_bot, y_bot = int(max(points_img[:,0])), int(max(points_img[:,1]))
        margin = 10
        # print(x_top, y_top, x_bot, y_bot)

        if (args.visualize):
            for point_2d in points_img:

                # TODO: Hard code here
                # point_2d[0] -= int(90.88066101074219/2)
                cv2.circle(img, point_2d.astype(int), 3, (0, 0, 255), -1)

            cv2.rectangle(img, (x_top-margin, y_top-margin), 
                            (x_bot+margin, y_bot+margin), (255,0,0), 1)
            cv2.imshow("Test", img)
            key = cv2.waitKey(1)
            if (key == ord("q")):
                break
        
        bbox2d_dict[img2gt[0]] = [x_top, y_top, x_bot, y_bot]
        
        # except:
        #     print("Point cloud missing")
    
    with open(os.path.join(args.output, "bbox2d.json"), 'w') as f:
        json.dump(bbox2d_dict, f)

if __name__ == "__main__":
    main()

