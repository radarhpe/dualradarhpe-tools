import numpy as np
import os
import os.path as osp
import argparse
import json
from pypcd4 import PointCloud
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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
# joint_ind = [0, 16, 17, 18]

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

def get_3D_from_2D(points_img, lidar_2d, lidar_3d):
    points_3d = np.zeros((points_img.shape[0], 3))

    # print("Point", points_img)
    # print("Shape", lidar_2d.shape)
    
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

    if (lidar_2d.shape[0] == 0 or lidar_3d.shape[0] == 0):
        return points_3d
    
    # Get distance from image points and lidar 2d
    for i, point_img in enumerate(points_img[1:]):
        # point_img_ = np.array([point_img[1], point_img[0]])
        dist_arr = np.linalg.norm(lidar_2d - point_img, axis=1)
        min_arg = np.argsort(dist_arr)
        points_3d[i+1] = np.mean(lidar_3d[min_arg[:3]], axis=0)

    return points_3d

def slice_featuremap(featuremap, gt, cut_radius=1.0):
    # Get Pelvic coord as box center
    center = gt[0]

    # Calculate distance from all featuremap points to center
    fm_dist = np.linalg.norm(featuremap[:,:3] - center, axis=1)
    indices = (fm_dist <= cut_radius)
    return featuremap[indices]

def adding_ankle(points_2d):
    left_knee, right_knee = points_2d[2], points_2d[5]
    left_ankle = left_knee.copy()
    left_ankle[1] += 30
    right_ankle = right_knee.copy()
    right_ankle[1] += 30
    points_2d[3] = left_ankle
    points_2d[6] = right_ankle
    return points_2d

def main():
    parser = argparse.ArgumentParser(
        prog="visualize_gt_on_image.py",
        description="For visualizing gt on image"
    )

    parser.add_argument("-d", "--dir", type=str, help="Path to sequence root directory")
    parser.add_argument("-p", "--pcl", type=str, help="Path to point cloud dir from root. \
                                                       Can be radar pcl, lidar or gt")
    parser.add_argument("--seq", type=int, default=5, help="Number of consecutive frame to \
                                                            combine into one")
    parser.add_argument("--size", type=int, default=12, help="Size of 2D point cloud matrix")
    parser.add_argument("--norm_type", default="sequence", help="Normalization type: by frame, \
                                                                 by sequence or global")
    parser.add_argument("-s", "--start_frame", type=int, default=0, help="Starting frame")
    parser.add_argument("-m", "--mode", type=str, default="rt", help="Viz mode")
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()
    viz = args.viz
    if (viz):
        mode = args.mode
    mat_size = args.size
    n_frames = args.seq
    output_dir = osp.join(args.dir, f"radarpcl_multiper/")
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    # Get mapping file
    pcl_type = args.pcl[args.pcl.rfind("/")+1:]
    mapping_file = osp.join(args.dir, "calib", f"images2{pcl_type}_mapping.txt")
    img2lid = np.loadtxt(mapping_file, dtype=str)

    mapping_file = osp.join(args.dir, "calib", f"radarpcl2images_mapping.txt")
    rad2img = np.loadtxt(mapping_file, dtype=str)
    img2rad = {img: rad for rad, img in rad2img}

    if (viz):
        fig = plt.figure()
        fig.tight_layout()

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, 10)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-2, 2)
        ax.view_init(elev=45, azim=-165)
    
    """
    Generate groundtruth json with format
    {
        "radar_timestamp": {
            "frame_id": [frame_01, frame_02, ...] # List of frame id based on number of skeleton
            "poses": [[[x, y, z],[...]]] # List of joints. Shape (#skeleton, 15, 3)
        }
    }
    """
    gt_json = {}
    seq_name = args.dir[args.dir.rfind("/")+1:]
    skeleton_cnt = -1
    for i, img2gt in enumerate(img2lid):
        if (i < args.start_frame):
            continue
        if (img2gt[1] == "nan" or img2gt[0] not in img2rad):
            continue
        # print("Timestamp: ", img2gt[0], "-", gt_csv[gt_idx[img2gt[1]]].split(",")[1])
        if (i % 50 == 0):
            print("Timestamp: ", img2gt[0], img2gt[1])
        # try:

        # Load images
        if (viz):
            img = cv2.imread(osp.join(args.dir, "images", img2gt[0]+".png"))

        # Load pcl
        pcd = PointCloud.from_path(osp.join(args.dir, args.pcl, img2gt[1]+".pcd"))
        points_world = pcd.numpy(("x", "y", "z"))
        points_world = points_world[~np.isnan(points_world).any(axis=1)]
        points_cam = src2dst_transformation(np.linalg.inv(T_lidar_cam), points_world)

        x_criter = np.logical_and((points_cam[:, 0] <= 6.8), (points_cam[:, 0] >= 1.0))
        y_criter = np.logical_and((points_cam[:, 1] <= 5.0), (points_cam[:, 1] >= -5.0))
        z_criter = np.logical_and((points_cam[:, 2] <= 2.0), (points_cam[:, 2] >= -2.0))
        pnt_indices = np.logical_and(x_criter, np.logical_and(y_criter, z_criter))
                      
        points_cam = points_cam[pnt_indices]
        points_world = points_world[pnt_indices]

        # print(points_cam[0])

        points_img = point2pixel(points_cam[:,:3], left_cam_info).squeeze()
        # print(points_img)
        points_img[:,1] = IMG_SIZE[0] - points_img[:,1]

        # Load openpose estimation
        hpe2d_json = None
        with open(osp.join(args.dir, "openpose", f"{img2gt[0]}_keypoints.json")) as f:
            hpe2d_json = json.load(f)["people"]
        
        if (viz):
            # Show lidar
            for point_2d in points_img:
                cv2.circle(img, point_2d.astype(int), 3, (0, 0, 255), -1)

            ax.collections.clear()
            ax.lines.clear()

        # Show openpose prediction
        points_3d_hpe = []
        skeletons_id_arr = []
        for person in hpe2d_json:
            org_points = np.asarray(person["pose_keypoints_2d"]).reshape(-1,3)
            head_points = org_points[[0,16,17,18]]
            index = np.where(np.all(head_points[:,:2] != np.array([0.,0.]), axis=1))
            head_points = np.mean(head_points[index], axis=0)
            # print(head_points)

            points = np.asarray(person["pose_keypoints_2d"]).reshape(-1,3)[joint_ind]
            points[8] = head_points
            for ankle_index in [3, 6]:
                if (points[ankle_index,:2] == np.array([0.,0.])).all():
                    points = adding_ankle(points)
                    # print(points[3], points[6])

            # bad_score = (points[:,2] < 0.5).any()
            avg_score = np.mean(points[:,2]) #+ 1.0
            zero_point = (points == np.array([0.0,0.0,0.0])).any()# and False
            if (avg_score >= 0.25 and not zero_point):
                if (viz):
                    for i, point in enumerate(points):
                        point = point[:2].astype(np.int32)
                        cv2.circle(img, point, 3, (0, 0, 255), -1)
                        cv2.putText(img, str(i), point, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

                point_3d_hpe = get_3D_from_2D(points[:,:2], points_img, points_cam)
                if ((point_3d_hpe == np.array([0.0,0.0,0.0])).any()):
                    continue

                # Transform to TI radar coord
                
                point_3d_hpe = src2dst_transformation(np.linalg.inv(T_cam_ti), point_3d_hpe)
                person_height = np.max(point_3d_hpe[:,2]) - np.min(point_3d_hpe[:,2])
                if (person_height < 1.0):
                    continue

                points_3d_hpe.append(point_3d_hpe.tolist())

                skeleton_cnt += 1
                skeletons_id_arr.append(int(skeleton_cnt))

                # print(point_3d_hpe.shape)
                # ax.scatter(point_3d_hpe[:,0],point_3d_hpe[:,1], point_3d_hpe[:,2])
                if (viz):
                    for id in line_idx:
                        ax.plot([point_3d_hpe[id[0],0], point_3d_hpe[id[1],0]],
                                [point_3d_hpe[id[0],1], point_3d_hpe[id[1],1]],
                                [point_3d_hpe[id[0],2], point_3d_hpe[id[1],2]], c='blue')

        gt_json[img2rad[img2gt[0]]] = {}
        gt_json[img2rad[img2gt[0]]]["frame_id"] = skeletons_id_arr
        gt_json[img2rad[img2gt[0]]]["poses"] = points_3d_hpe
        # if (len(skeletons_id_arr) != len(points_3d_hpe)):
        #     print("Problem at ", img2gt[0])

        if (viz):
            # cv2.imshow("Test", img)
            if (args.mode == "s"):
                key = cv2.waitKey(0)
                plt.show()
                exit(0)
            else:
                plt.draw()
                plt.pause(0.0001)
                key = cv2.waitKey(30)

            if (key == ord("q")):
                break
            # except:
            #     print("Point cloud missing")


    # Create data holder for feature map, gt and data type
    ft_tensor = np.empty((0, mat_size**2, 5))
    gt_tensor = np.empty((0, 45))
    # 
    count = 0
    frame_list = sorted(list(gt_json.keys()))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_xlim(0, 8)
    # ax.set_ylim(-4, 4)
    # ax.set_zlim(-2, 2)

    skeleton_cnt = 0
    for i in range(len(frame_list[:-n_frames+1])):
        rad_ts = frame_list[i]
        gts = np.asarray(gt_json[rad_ts]["poses"], dtype=np.float32)
        if (gts.shape[0] == 0):
            del gt_json[rad_ts]
            continue
        # print(gts.shape)
        ft_stack = np.empty((0, 5))

        # ax.collections.clear()
        # ax.lines.clear()

        for j in range(n_frames):
            # Load radar point cloud
            tmp_rad_ts = frame_list[i+j]
            pcd_file = tmp_rad_ts + ".xyzdi"
            # Shape (N, 5) - (x, y, z, doppler, intensity)
            radar_pcl = np.loadtxt(osp.join(args.dir, "radar/radarpcl", pcd_file))
            radar_pcl[:,[0,1]] = radar_pcl[:,[1,0]]
            radar_pcl[:,4] = 20 * np.log(radar_pcl[:,4])
            ft_stack = np.append(ft_stack, radar_pcl, axis=0)

        has_data = False
        center_list = []
        frame_id_list = []
        pop_list = []
        for ske_id, abs_gt in enumerate(gts):
            # Cut part of featuremap based on body center (Pelvic)
            sliced_ft = slice_featuremap(ft_stack, abs_gt, cut_radius=1.0)
            # print(sliced_ft.shape[0])
            if (sliced_ft.shape[0] < 0.25 * mat_size**2):
                # Remove skeleton from gt_json
                pop_list.append(ske_id)
                continue
            else:
                # Update frame id based on skeleton count
                gt_json[rad_ts]["frame_id"][ske_id] = skeleton_cnt
                skeleton_cnt += 1
            has_data = True

            # Align gt and the slice to body center
            sliced_ft[:,:3] -= abs_gt[0]
            gt = abs_gt - abs_gt[0]

            # ax.scatter(sliced_ft[:,0]+abs_gt[0][0],sliced_ft[:,1]+abs_gt[0][1],
            #            sliced_ft[:,2]+abs_gt[0][2], s=5)
            # for id in line_idx:
            #     test_gt = gt + abs_gt[0]
            #     ax.plot([test_gt[id[0],0], test_gt[id[1],0]],
            #             [test_gt[id[0],1], test_gt[id[1],1]],
            #             [test_gt[id[0],2], test_gt[id[1],2]], c='blue')
                
            # break
            # Feature map place holder
            ft_holder = np.zeros((mat_size**2, 5))
            if (sliced_ft.shape[0] > mat_size**2):
                ft_holder = sliced_ft[:mat_size**2]
            else:
                ft_holder[:sliced_ft.shape[0]] = sliced_ft

            # Sort by x for all points (mat_size**2)
            ft_holder = ft_holder[ft_holder[:, 0].argsort()]

            # Sort by y for each size number of points
            for k in range(mat_size):
                sub_radar_pcl = ft_holder[mat_size*k:mat_size*(k+1)]
                ft_holder[mat_size*k:mat_size*(k+1)] = sub_radar_pcl[sub_radar_pcl[:,1].argsort()]

            # Put sliced_ft into final ft_tensor
            ft_tensor = np.append(ft_tensor, np.expand_dims(ft_holder, axis=0), axis=0)
            gt_tensor = np.append(gt_tensor, np.expand_dims(gt.flatten(), axis=0), axis=0)
        
        pop_list = sorted(pop_list, reverse=True)
        for pop_id in pop_list:
            gt_json[rad_ts]["frame_id"].pop(pop_id)
            gt_json[rad_ts]["poses"].pop(pop_id)

        # Clean trash in gt_json
        if (len(gt_json[rad_ts]["frame_id"]) == 0):
            del gt_json[rad_ts]
            # print(gt_json[rad_ts])

        # plt.draw()
        # plt.pause(0.1)


    ft_tensor = ft_tensor.reshape((-1, mat_size, mat_size, 5))
    print(ft_tensor.shape, gt_tensor.shape)
    # Normalize data        
    if (args.norm_type == "sequence"):
        mean_val, std_val = np.mean(ft_tensor[:,:,:,4]), np.std(ft_tensor[:,:,:,4])
        ft_tensor[:,:,:,4] = (ft_tensor[:,:,:,4] - mean_val) / (std_val + 1e-10)

    # Save radar pcl to file
    with open(output_dir + f"/featuremap_{seq_name}_outdoor.npy", "wb") as f:
        np.save(f, ft_tensor)
    # Save optitrack gt to file
    with open(output_dir + f"/labels_{seq_name}_outdoor.npy", "wb") as f:
        np.save(f, gt_tensor)
    # Save gt json
    with open(osp.join(args.dir, f"gt/{seq_name}.json"), "w") as f:
        json.dump(gt_json, f)

    print(count)

if __name__ == "__main__":
    main()

