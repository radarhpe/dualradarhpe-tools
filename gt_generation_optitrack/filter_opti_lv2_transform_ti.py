import numpy as np
import os
import os.path as osp
import argparse
import datetime
from joints_order import j_order, gt_select
from scipy.spatial.transform import Rotation
from data_split import data_info

import sys
# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add it to the sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from calibration.extrinsic_calib import src2dst_transformation, create_transformation_matrix,\
                            T_lidar_ti, T_box_lidar, T_calib_radar

calib_path_tail = "calib/radarpcl2gt_mapping.txt"
gt_sel_idx = np.array([j_order[gt_sel] for gt_sel in gt_select])

# train_seq will be splitted into train and validation
# train_seq = ["civit_easy_01", "civit_norm_01", "civit_norm_02", "civit_norm_04", 
#              "civit_norm_05", "civit_norm_06", "civit_norm_08", "civit_multi_01",
#              "civit_hard_01"] 

# test_seq = ["civit_norm_03", "civit_hard_02", "civit_multi_02"]

# Check if gt line has enough data based on gt_select and number of person in scene
# and produce filtered gt
def produce_gt(gt_line, n_per, T_box2lidar, T_ti2calib):
    res_line = np.empty((0))

    # Check sensor box position first
    if (np.isnan(gt_line[-7:]).any()):
        return None

    has_data = False
    for i in range(n_per):
        gt_line_idx = []
        for idx in gt_sel_idx[:-1]:
            gt_line_idx += [3*idx + 0 + 111*i, 3*idx + 1 + 111*i, 3*idx + 2 + 111*i]
        gt_line_data = gt_line[gt_line_idx]
        has_nan = np.isnan(gt_line_data).any()
        if (not has_nan or n_per != 1):
            # Reduce 4 first joints to be 1 joint (each joint has 3 numbers)
            has_data = True
            first_joint = np.mean(gt_line_data[:4*3].reshape(-1, 3), axis=0).flatten()
            gt_line_data = np.concatenate([first_joint, gt_line_data[12:]])
            res_line = np.concatenate([res_line, gt_line_data])

    if (has_data):
        # Put sensor box info into res_line
        res_line = np.concatenate([res_line, gt_line[-7:]])

        # Transform point in world coord to TI coord
        T_opti_box = create_transformation_matrix(res_line[-7:-3], res_line[-3:])

        gt_opti = res_line[:-7].reshape(-1,3)

        # Opti -> Box
        gt_box = src2dst_transformation(np.linalg.inv(T_opti_box), gt_opti)
        
        # Box -> Lidar
        gt_lidar = src2dst_transformation(np.linalg.inv(T_box2lidar), gt_box)
        # Lidar -> TI
        gt = src2dst_transformation(np.linalg.inv(T_lidar_ti), gt_lidar)

        # TI calib (rotate to parallel with ground)
        gt = src2dst_transformation(np.linalg.inv(T_ti2calib), gt)

        # Transform sensor box pose
        res_line[:-7] = gt.flatten()
        return res_line    
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        prog="filter_optitrack_step2.py",
        description="Filter out good gt points for all data seqs,\
                     in radar coord"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path root data directory"
    )

    parser.add_argument(
        "--sel",
        default="",
        help="Select sequence to process instead of processing all sequences"
    )

    args = parser.parse_args()

    # select_seq = ["civit_easy_06"]
    select_seq = []
    if (args.sel != ""):
        select_seq = [args.sel]

    if (len(select_seq) != 0):
        seqs = select_seq
    else:
        seqs = sorted(os.listdir(args.dir))
    
    total_train = 0
    total_test = 0
    total_val = 0

    for seq in seqs:
        if (seq not in data_info):
            print("Cannot find: ", seq)
            continue
        set_type = data_info[seq]["type"]
        print(seq, set_type)

        n_per = 1 if "multi" not in seq else 2

        ts_map = np.loadtxt(osp.join(args.dir, seq, calib_path_tail), dtype=str)
        
        T_box2lidar = T_box_lidar[data_info[seq]["date"]]
        T_ti2calib = T_calib_radar[data_info[seq]["date"]]

        # Get gt list
        gt_csv = None
        with open(osp.join(args.dir, seq, "gt", seq+".csv"), "r") as f:
            gt_csv = f.read().split("\n")[:-1] # Load, split by row and remove last row
            header = gt_csv[0].split(",") # Get header
            gt_csv = gt_csv[1:]
        gt_idx = {str(line.split(",")[1]): int(line.split(",")[0]) for line in gt_csv}

        data_lines = []
        for i in range(len(ts_map)):
            rad_ts, gt_ts = ts_map[i]
            if (gt_ts == "nan" or (i < 150 and data_info[seq]["date"] == "april2")
                               or (i < 50 and seq == "civit_hard_02")):
                continue

            # Get gt at gt_ts
            gt_line = np.asarray(gt_csv[gt_idx[gt_ts]].split(",")[2:-1])
            gt_line[gt_line == ""] = "nan"

            # There are 111*num_person + 7 values in gt
            # 7 last values are sensor box info
            gt = np.array(gt_line, np.float32) 

            # Get gt_sel
            gt_sel = produce_gt(gt, n_per, T_box2lidar, T_ti2calib)
            if (gt_sel is not None):
                data_line = [str(i), rad_ts, str(gt_idx[gt_ts]), gt_ts]
                data_line += list(gt_sel.astype(str))
                data_lines.append(data_line)

        val_start = int(0.6*len(data_lines))
        val_end = int(0.8*len(data_lines))
        if (set_type == "train"):
            total_train += int(0.8*len(data_lines))
            total_val   += int(0.2*len(data_lines))
        else:
            total_test  += len(data_lines)

        with open(osp.join(args.dir, seq, "gt", seq+"_filtered.csv"), "w") as f:
            header_str = ["Rad_id","Rad_time","GT_id","GT_time"]
            # Header still has first two columns (Name and Time)
            for i in range(n_per):
                head_idx = []
                for idx in gt_sel_idx[4:-1]:
                    head_idx += [3*idx + 2 + 111*i, 3*idx + 3 + 111*i, 3*idx + 4 + 111*i]
                header_str += [f"Skeleton 00{i+1}:Pelvis+X", 
                               f"Skeleton 00{i+1}:Pelvis+Y", 
                               f"Skeleton 00{i+1}:Pelvis+Z"]
                header_str += list(np.array(header)[head_idx])
            
            header_str += header[-8:-1]
            f.write(",".join(header_str)+",Type\n")

            for i, data_line in enumerate(data_lines):
                type = set_type
                if (type == "train"):
                    if (i >= val_start and i <= val_end):
                        type = "val"
                f.write(",".join(data_line) + "," + type + "\n")
    print(f"Total train frames: {total_train}\nTotal val frames: {total_val}\nTotal test frames: {total_test}\n")
if __name__ == "__main__":
    main()