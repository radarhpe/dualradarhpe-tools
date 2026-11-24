import numpy as np
import open3d as o3d
import os
import os.path as osp
import argparse
import time
import json
from scipy.spatial.transform import Rotation
from data_split import data_info
import sys

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from calibration.extrinsic_calib import T_lidar_ti, T_box_lidar, T_calib_radar, T_lidar_bosch, \
    create_transformation_matrix, src2dst_transformation

# ROI (x_min, x_max, y_min, y_max, z_min, z_max)
ROI = [0.5, 8.0, -2.5, 2.5, -np.inf, np.inf]

def is_pelvis_in_roi(gt):
    gt_x, gt_y, gt_z = gt[0,0], gt[0,1], gt[0,2]
    is_gt_x_valid = np.all(gt_x >= ROI[0]) and np.all(gt_x <= ROI[1])
    is_gt_y_valid = np.all(gt_y >= ROI[2]) and np.all(gt_y <= ROI[3])
    is_gt_z_valid = np.all(gt_z >= ROI[4]) and np.all(gt_z <= ROI[5])

    return (is_gt_x_valid and is_gt_y_valid and is_gt_z_valid)

def is_gt_in_roi(gt):
    gt_x, gt_y, gt_z = gt[:,0], gt[:,1], gt[:,2]
    is_gt_x_valid = np.all(gt_x >= ROI[0]) and np.all(gt_x <= ROI[1])
    is_gt_y_valid = np.all(gt_y >= ROI[2]) and np.all(gt_y <= ROI[3])
    is_gt_z_valid = np.all(gt_z >= ROI[4]) and np.all(gt_z <= ROI[5])

    return (is_gt_x_valid and is_gt_y_valid and is_gt_z_valid)

def extract_radar_pose(csv_path, T_box_radar):
    position = np.empty((0, 3))
    orientation = np.empty((0, 3))
    T_ti0_ti_arr = np.empty((0, 4, 4))
    T_opti_ti0 = None

    full_csv = None
    with open(csv_path, "r") as f:
        full_csv = f.read().split("\n")[1:-1] # Remove header
    for line_csv in full_csv:
        line = line_csv.split(",")
        box_ori, box_pos = line[-8:-4], line[-4:-1]
        T_opti_box = create_transformation_matrix(box_ori, box_pos)
        T_opti_ti = T_opti_box @ T_box_radar
        if (T_opti_ti0 is None):
            T_opti_ti0 = T_opti_ti

        T_ti0_ti = np.linalg.inv(T_opti_ti0) @ T_opti_ti
        T_ti0_ti_arr = np.append(T_ti0_ti_arr, np.expand_dims(T_ti0_ti, axis=0), axis=0)
        position = np.append(position, np.expand_dims(T_ti0_ti[:3,3], axis=0), axis=0)
        rotation = Rotation.from_matrix(T_ti0_ti[:3,:3]).as_euler("xyz")
        orientation = np.append(orientation, np.expand_dims(rotation, axis=0), axis=0)

    return position, orientation, T_ti0_ti_arr

def compensate_doppler(radar_pcl, delta_xyz):
    # v_comp = v_org - v_stat
    # v_stat = -(v_senx*cos(P_angle) + v_seny*sin(P_angle))
    P_angle = np.arctan2(radar_pcl[:,1], radar_pcl[:,0])
    v_stat = -(delta_xyz[0] * np.cos(P_angle) + delta_xyz[1] * np.sin(P_angle))
    v_comp = radar_pcl[:, 3] - v_stat
    return v_comp
    
def slice_featuremap(featuremap, gt, cut_radius=1.0):
    # Get Pelvic coord as box center
    center = gt[0]

    # Calculate distance from all featuremap points to center
    fm_dist = np.linalg.norm(featuremap[:,:3] - center, axis=1)
    indices = (fm_dist <= cut_radius)

    return featuremap[indices]

def filter_gt(gt, n_pers=1):
    final_gt = np.empty((0, 15, 3))
    gt = np.reshape(gt, (-1, 15, 3))
    for i in range(n_pers):
        if (is_gt_in_roi(gt[i])):
            final_gt = np.append(final_gt, np.expand_dims(gt[i], axis=0), axis=0)
    if (final_gt.shape[0] > 1):
        # Check occlusion
        P1_pelvis = final_gt[0,0]
        P2_pelvis = final_gt[1,0]

        origin = np.asarray([0,0,0])

        P1O = np.linalg.norm(origin-P1_pelvis)
        P2O = np.linalg.norm(origin-P2_pelvis)
        P1P2 = np.linalg.norm(P2_pelvis-P1_pelvis)
        delta = abs(np.arccos((P1O**2 + P2O**2 - P1P2**2)/(2*P1O*P2O)))

        if (delta <= 3*np.pi/180):
            if (P1_pelvis[0] < P2_pelvis[0]):
                final_gt = final_gt[:1]
            else:
                final_gt = final_gt[1:]

    return final_gt
    

def main():
    parser = argparse.ArgumentParser(
        prog="radarpcl_multiperson",
        description="Create radar pcl data for top-down multi person pose estimation"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to radarpcl directory"
    )

    parser.add_argument(
        "--seq",
        type=int,
        default=5,
        help="Number of consecutive frame to combine into one"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=12,
        help="Size of 2D point cloud matrix"
    )

    parser.add_argument(
        "--sel",
        default="",
        help="Select sequence to process instead of processing all sequences"
    )

    parser.add_argument(
        "--no_use_log",
        action="store_true",
        help="perform log transform for intensity"
    )

    parser.add_argument(
        "--norm_type",
        default="sequence",
        help="Normalization type: by frame, by sequence or global"
    )

    parser.add_argument(
        "--pcl_type",
        default="ti",
        help="Define radar sensor (ti/bosch)"
    )

    parser.add_argument(
        "--no_comp",
        action="store_true",
        help="Deactivate doppler compensation"
    )


    # parser.add_argument(
    #     "--use_transform",
    #     action="store_true",
    #     help="Decide to transform point cloud back to the odom frame of the radar"
    # )

    args = parser.parse_args()
    print(args)

    ignore_seqs = ["civit_norm_07", "civit_easy_prashant_01"]
    # ignore_dates = ["april", "april2", "april3", "april4"]
    ignore_dates = []

    select_seq = []
    if (args.sel != ""):
        select_seq = [args.sel]

    if (len(select_seq) != 0):
        seqs = select_seq
    else:
        seqs = sorted(os.listdir(args.dir))

    if (args.pcl_type == "ti"):
        radarpcl_path_tail = "radar/radarpcl" # bosch
        prefix = "radarpcl"
    else:
        radarpcl_path_tail = "bosch" # bosch
        prefix = "bosch"

    # For counting good frame from each set
    good_frames = {
        "train": 0,
        "val": 0,
        "test" : 0
    }

    # Feature and Label collection
    mat_size = args.size
    ft_dict = {}
    gt_dict = {}
    type_dict = {}
    combine_dict = {}

    for seq in seqs:
        if (seq not in data_info or seq in ignore_seqs):
            print(f"Cannot find {seq} or it is ignored")
            continue
        if (data_info[seq]["date"] in ignore_dates):
            continue

        # Read config and calibration matrix
        T_ti2calib = T_calib_radar[data_info[seq]["date"]]
        data_dir = osp.join(args.dir, seq, radarpcl_path_tail)
        if (args.pcl_type == "ti"):
            groundtruth = osp.join(args.dir, seq, "gt", seq[seq.rfind("/")+1:] + "_filtered.csv")
        else:
            groundtruth = osp.join(args.dir, seq, "gt", seq[seq.rfind("/")+1:] + "_filtered_bosch.csv")
        output_dir = osp.join(args.dir, seq, prefix + "_multiper/")
        if (not os.path.exists(output_dir)):
            os.makedirs(output_dir)

        # 
        seq_date = data_info[seq]["date"]
        T_box_radar = T_box_lidar[seq_date] @ T_lidar_ti @ T_calib_radar[seq_date]
        position, orientation, T_ti0_ti_arr = extract_radar_pose(groundtruth, T_box_radar)

        # mapping_file = seq + "/calib/" + prefix + "2gt_mapping.txt"

        # Load GT
        # GT line format: rad_id, rad_ts, gt_id, gt_ts, j_00_x, j_00_y, j_00_z, ..., data_type (train/val/test)
        # j_**_x/y/z: 3D position of joint **
        n_pers = 1 if "multi" not in seq else 2
        gt_csv = None
        with open(groundtruth, "r") as f:
            gt_csv = f.read().split("\n")[1:-1] # Load, split by row and remove 1st and last row
        total_frame = len(gt_csv)
        print("Seq: ", seq)
        print("Total frame: ", total_frame)
        print("position list: ", position.shape[0])

        # Create data holder for feature map, gt and data type
        ft_tensor = np.empty((0, mat_size**2, 5))
        gt_tensor = np.empty((0, 3*15))
        type_list = np.empty(0, dtype=str)
        mapping_json = {}
        # good_data = [] # Frame index that has more than 50% point cloud
        # good_rad_id = [] # Radar frame index that has more than 50% point cloud


        # For Doppler compensation
        delta_rad_ts = [0.0]
        delta_xyz = [np.array([0.0, 0.0, 0.0])]
        prev_rad_ts = float(gt_csv[0].split(",")[1])
        for i in range(1, total_frame):
            # Update delta rad ts
            rad_ts = float(gt_csv[i].split(",")[1])
            delta_rad_ts.append(rad_ts - prev_rad_ts)
            prev_rad_ts = rad_ts

            # Update delta xyz
            # print(position[i]-position[i-1])
            delta_xyz.append((position[i]-position[i-1])/(delta_rad_ts[-1]*1e-9))
        delta_rad_ts = np.asarray(delta_rad_ts)
        delta_xyz = np.asarray(delta_xyz) 
        
        # 
        avg_point_cnt = 0.0
        frame_cnt = {}   # Count number of actual frame (each frame contains one skeleton)
        for i in range(0, total_frame-args.seq):
            n_frames = args.seq
            point_cnt = 0

            # Load GT from csv file
            gt_line = np.array(gt_csv[i].split(","))
            # print(gt_line)
            box_rot = np.array(gt_line[-8:-4], dtype=np.float32)
            box_pos = np.array(gt_line[-4:-3], dtype=np.float32)
            rad_id, rad_ts, gt_id, gt_ts = gt_line[:4]
            gt = np.array(gt_line[4:-8], dtype=np.float32)
            data_type = gt_line[-1]

            # Stacking featuremap
            ft_stack = np.empty((0, 5))
            for j in range(n_frames):
                # Load radar point cloud
                pcd_file = rad_ts + ".xyzdi"
                # Shape (N, 5) - (x, y, z, doppler, intensity)
                radar_pcl = np.loadtxt(os.path.join(data_dir, pcd_file))
                # print(radar_pcl.shape[0])
                if (prefix == "radarpcl"):
                    radar_pcl[:,[0,1]] = radar_pcl[:,[1,0]]
                
                # Compensate Doppler
                # radar_doppler = np.array(radar_pcl[:,3])
                if (delta_rad_ts[j] != 0.0 and not args.no_comp):
                    radar_pcl[:,3] = compensate_doppler(radar_pcl, delta_xyz[j])
                    # print("Total difference in Doppler: ", np.mean(np.abs(radar_doppler-radar_pcl[:,3])))

                # Calib radar to align with the ground
                radar_pcl[:,:3] = src2dst_transformation(np.linalg.inv(T_ti2calib), 
                                                         radar_pcl[:,:3])

                # Filter out range < 1.0 or range > 10.0
                # radar_pcl = radar_pcl[radar_pcl[:, 0] >= ROI[0]]
                # radar_pcl = radar_pcl[radar_pcl[:, 0] <= ROI[1]]
                # radar_pcl = radar_pcl[radar_pcl[:, 1] >= ROI[2]]
                # radar_pcl = radar_pcl[radar_pcl[:, 1] <= ROI[3]]
                # radar_pcl = radar_pcl[radar_pcl[:, 2] >= ROI[4]]
                # radar_pcl = radar_pcl[radar_pcl[:, 2] <= ROI[5]]
                if (not args.no_use_log and args.pcl_type == "ti"):
                    radar_pcl[:,4] = 20 * np.log(radar_pcl[:,4])

                ft_stack = np.append(ft_stack, radar_pcl, axis=0)

            # Filter GT
            final_gts = filter_gt(gt, n_pers)

            # Slice featuremap based on gt
            if (final_gts.shape[0] != 0):
                # Assign data type
                # print(final_gts.shape)
                has_data = False
                poses_list = []
                frame_id_list = []

                # Filtering gt and featuremap based on human center
                for final_gt in final_gts:
                    # Cut part of featuremap based on body center (Pelvic)
                    sliced_ft = slice_featuremap(ft_stack, final_gt, cut_radius=1.0)
                    if (sliced_ft.shape[0] < 0.5 * mat_size**2):
                        continue
                    has_data = True
                    avg_point_cnt += sliced_ft.shape[0]

                    # Save center position for later trasnlation
                    poses_list.append(final_gt.tolist())
                    
                    # Align gt and the slice to body center
                    sliced_ft[:,:3] -= final_gt[0]
                    final_gt -= final_gt[0]
                    
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
                    gt_tensor = np.append(gt_tensor, np.expand_dims(final_gt.flatten(), axis=0), axis=0)
                    type_list = np.append(type_list, data_type)
                    if (data_type not in frame_cnt):
                        frame_cnt[data_type] = 0
                    else:
                        frame_cnt[data_type] += 1
                    frame_id_list.append(frame_cnt[data_type])

                if (has_data):
                    if (data_type in mapping_json):
                        mapping_json[data_type][rad_ts] = {
                            "frame_id": frame_id_list,
                            "poses": poses_list,
                        }
                    else:
                        mapping_json[data_type] = {
                            rad_ts: {
                                "frame_id": frame_id_list,
                                "poses": poses_list,
                            }
                        }

            # if (args.norm_type == "frame"):
            #     # max_int, min_int = np.max(ft_tensor[i][:,4]), np.min(ft_tensor[i][:,4])
            #     # ft_tensor[i][:,4] = (ft_tensor[i][:,4] - min_int) / (max_int - min_int + 1e-10)
            #     mean_int, std_int = np.mean(ft_tensor[i][:,4]), np.std(ft_tensor[i][:,4])
            #     ft_tensor[i][:,4] = (ft_tensor[i][:,4] - mean_int) / (std_int + 1e-10)
            # print(point_cnt)
            # print(ft_tensor[i][:,4])
            # good_data.append(i)
            # good_frames[type_list[i]] += 1
            # if (point_cnt < 0.5*mat_size**2):
            #     print(point_cnt)

        ft_tensor = ft_tensor.reshape((-1, mat_size, mat_size, 5))
        print(ft_tensor.shape, gt_tensor.shape)
        print("Average number of point per person: ", avg_point_cnt / ft_tensor.shape[0])
        # Normalize data        
        if (args.norm_type == "sequence"):
            if (args.pcl_type != "bosch"):
                # max_int, min_int = np.max(ft_tensor[:,:,:,4]), np.min(ft_tensor[:,:,:,4])
                # ft_tensor[:,:,:,4] = (ft_tensor[:,:,:,4] - min_int) / (max_int - min_int + 1e-10)
                # print(np.max(ft_tensor[:,:,:,4]), np.min(ft_tensor[:,:,:,4]))
                mean_int, std_int = np.mean(ft_tensor[:,:,:,4]), np.std(ft_tensor[:,:,:,4])
                ft_tensor[:,:,:,4] = (ft_tensor[:,:,:,4] - mean_int) / (std_int + 1e-10)
            else:
                min_val, max_val = np.min(ft_tensor[:,:,:,4]), np.max(ft_tensor[:,:,:,4])
                ft_tensor[:,:,:,4] = (ft_tensor[:,:,:,4] - min_val) / (max_val - min_val + 1e-10)

        # Append to collection
        ft_dict[seq] = ft_tensor
        gt_dict[seq] = gt_tensor
        type_dict[seq] = type_list
        combine_dict[seq] = mapping_json

        # Save data
        for set_type in ["train", "val", "test"]:
            if (ft_tensor[type_list == set_type].shape[0] == 0):
                continue
            # Save radar pcl to file
            with open(output_dir + f"/featuremap_{seq}_{set_type.lower()}.npy", "wb") as f:
                np.save(f, ft_tensor[type_list == set_type])
            
            # Save optitrack gt to file
            with open(output_dir + f"/labels_{seq}_{set_type.lower()}.npy", "wb") as f:
                np.save(f, gt_tensor[type_list == set_type])
        
            # Save gt center and frame id
            if (set_type in mapping_json):
                with open(output_dir + f"/{seq}_{set_type.lower()}.json", "w") as f:
                    json.dump(mapping_json[set_type], f)

        # # Save good data index
        # np.savetxt(output_dir + f"/good_rad_idx_{seq}.txt", good_rad_id)

        # # Save good data index
        # np.savetxt(output_dir + f"/good_idx_{seq}.txt", good_data)

    # # Normalize radar point cloud intensity
    # ft_arr = np.empty((0, mat_size, mat_size, 5))
    # for k, v in ft_dict.items():
    #     ft_arr = np.append(ft_arr, v, axis=0)
    # max_int, min_int = np.max(ft_arr[:,:,:,4]), np.min(ft_arr[:,:,:,4])
    
    # Save data by type
    if (args.sel == ""):
        output_dir = osp.join(args.dir, f"../{prefix}_multiper_data")
        os.makedirs(output_dir, exist_ok=True)
        for motion_type in ["no_motion", "with_motion", "motion_only"]:
            for actor in ["", "_multi_person"]:
                if (actor == "_multi_person" and motion_type == "motion_only"):
                    continue
                os.makedirs(osp.join(output_dir, motion_type), exist_ok=True)

                ft_tensor = np.empty((0, mat_size, mat_size, 5))
                gt_tensor = np.empty((0, 45))
                type_list = np.empty(0)
                mapping_json = {"train": {}, "val": {}, "test": {}}

                for seq in seqs:
                    if (seq not in data_info or seq in ignore_seqs):
                        print(f"Cannot find {seq} or it is ignored")
                        continue
                    if (data_info[seq]["date"] in ignore_dates):
                        continue
                    if (motion_type == "no_motion" and "hard" in seq):
                        continue
                    if (motion_type == "motion_only" and "hard" not in seq):
                        continue
                    if (actor != "_multi_person" and ("prashant" in seq or "nikolay" in seq)):
                        continue
                    
                    ft_tensor = np.append(ft_tensor, ft_dict[seq], axis=0)
                    gt_tensor = np.append(gt_tensor, gt_dict[seq], axis=0)
                    type_list = np.append(type_list, type_dict[seq], axis=0)
                    
                    for type in combine_dict[seq].keys():
                        mapping_json[type][seq] = combine_dict[seq][type]

                for type in ["train", "val", "test"]:
                    if (actor == "_multi_person" and type != "test"):
                        continue
                    ft_name = f"featuremap_{type}{actor}.npy"
                    lb_name = f"labels_{type}{actor}.npy"
                    center_name = f"multiper_{type}{actor}.json"

                    type_idx = type_list == type

                    # Save radar pcl to file
                    with open(os.path.join(output_dir, motion_type, ft_name), "wb") as f:
                        np.save(f, ft_tensor[type_idx])
                    
                    # Save optitrack gt to file
                    with open(os.path.join(output_dir, motion_type, lb_name), "wb") as f:
                        np.save(f, gt_tensor[type_idx])
                    
                    # Save mapping json
                    with open(os.path.join(output_dir, motion_type, center_name), "w") as f:
                        json.dump(mapping_json[type], f)

                    # # Save optitrack gt to file
                    # with open(os.path.join(output_dir, motion_type, seq_idx_name), "w") as f:
                    #     json.dump(seq_frame_count[key], f, indent=2)


    # print("Total good training frame: ", good_frames["train"])
    # print("Total good validating frame: ", good_frames["val"])
    # print("Total good testing frame: ", good_frames["test"])

if __name__ == "__main__":
    main()
